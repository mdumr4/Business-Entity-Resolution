#!/usr/bin/env python3
"""
End-to-end evaluation of the Phase-3 ranker on the TRAIN labels.

Why this exists (what train_ranker.py's printout does NOT tell you):
  1. Its recall only counts true matches that survived L1+L2 into the candidate
     set. The leaderboard counts ALL true matches, so lost matches must hurt.
  2. Entities with zero candidates are absent from the feature parquet, so they
     silently vanish from its metric.
  3. Early stopping and threshold selection used the same split (double dipping).
  4. Entities used for design decisions (diagnostic samples, seeds 42/43, and the
     L2-pruner training sample) must never be in the holdout.

Design:
  * Universe = ALL train S1 entities (incl. those with zero candidates).
  * Deterministic hash split: 70% train / 15% tune / 15% holdout.
    Entities from the seed-42/43 design samples are forced into TRAIN.
  * Metric = per-entity F0.5 against the FULL ground truth (leaderboard metric).
  * Ranker trains on `train`, early-stops + picks threshold on `tune`,
    and every reported number comes from `holdout`.
  * Features are rounded to 4 decimals (like generate_submission.py's CSV) so the
    model is trained on the same numeric precision it sees at inference.

Usage:
  python evaluate_pipeline.py                        # main evaluation
  python evaluate_pipeline.py --ablate --transfer    # + feature ablations + US<->India proxy
  python evaluate_pipeline.py --device cpu           # if the GPU is busy (e.g. FAISS build)
"""
import argparse, json, os, time, zlib
import numpy as np
import polars as pl
import xgboost as xgb
from src.models.l3_ranker import (
    BASE_RANKER_FEATURES as BASE,
    RELATIVE_RANKER_FEATURES as REL,
    ALL_RANKER_FEATURES as ALL_FEATS,
)
DESIGN_SEEDS = (42, 43)

THRESHOLDS = np.round(np.arange(0.05, 0.991, 0.01), 2)


# ----------------------------------------------------------------------------- data
def load_entities(s1_path, gt_path):
    s1 = pl.read_csv(s1_path, separator="\t")
    gt = pl.read_csv(gt_path, separator="\t").with_columns(pl.col("matched_entity_ids").fill_null(""))
    n_true = gt.select([
        "source1_entity_id",
        pl.when(pl.col("matched_entity_ids") == "").then(0)
          .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1)
          .cast(pl.Int64).alias("n_true")])
    ent = (s1.select([pl.col("entity_id").alias("source1_entity_id"),
                      pl.col("country").fill_null("unknown")])
             .join(n_true, on="source1_entity_id", how="left")
             .with_columns(pl.col("n_true").fill_null(0)))
    ent = ent.with_columns(pl.Series("eidx", np.arange(ent.height, dtype=np.int64)))
    pairs = (gt.filter(pl.col("matched_entity_ids") != "")
               .select(["source1_entity_id", pl.col("matched_entity_ids").str.split(",").alias("cid")])
               .explode("cid")
               .select(["source1_entity_id", pl.col("cid").str.strip_chars().alias("candidate_entity_id")])
               .with_columns(pl.lit(1).cast(pl.Int8).alias("label")))
    return s1, ent, pairs


def assign_partitions(s1, ent, n_design=50000):
    ids = ent["source1_entity_id"].to_list()
    h = np.fromiter((zlib.crc32(("eval-v1:" + x).encode()) % 100 for x in ids), dtype=np.int16, count=len(ids))
    part = np.where(h < 70, 0, np.where(h < 85, 1, 2)).astype(np.int8)
    bad = set()
    for sd in DESIGN_SEEDS:
        if s1.height >= n_design:
            bad.update(s1.sample(n=n_design, seed=sd)["entity_id"].to_list())
    forced = ent["source1_entity_id"].is_in(list(bad)).to_numpy() if bad else np.zeros(len(ids), bool)
    part[forced] = 0
    print(f"[split] train={int((part==0).sum()):,} tune={int((part==1).sum()):,} "
          f"holdout={int((part==2).sum()):,} | design-sample entities forced to train: {int(forced.sum()):,}")
    return ent.with_columns(pl.Series("part", part))


def load_features(path, ent, pairs, round4=True):
    df = pl.read_parquet(path)
    if round4:
        df = df.with_columns([pl.col(c).round(4) for c in BASE if c != "name_exact"])
    df = df.join(pairs, on=["source1_entity_id", "candidate_entity_id"], how="left")
    df = df.with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
    df = df.join(ent.select(["source1_entity_id", "eidx", "country", "part"]),
                 on="source1_entity_id", how="left")
    df = df.with_columns([
        (pl.col("addr_cos") - pl.col("addr_cos").max().over("source1_entity_id")).alias("addr_cos_gap_from_top"),
        (pl.col("name_cos") - pl.col("name_cos").max().over("source1_entity_id")).alias("name_cos_gap_from_top"),
        pl.col("addr_cos").rank(method="ordinal", descending=True)
          .over("source1_entity_id").cast(pl.Float64).alias("addr_cos_rank"),
    ])
    return df


def build_set(df, ent, k, country=None, train_frac=1.0):
    e = ent.filter(pl.col("part") == k)
    d = df.filter(pl.col("part") == k)
    if country is not None:
        e = e.filter(pl.col("country") == country)
        d = d.filter(pl.col("country") == country)
    if train_frac < 1.0:
        keep = int(train_frac * 100)
        e = e.filter(pl.col("eidx") % 100 < keep)
        d = d.filter(pl.col("eidx") % 100 < keep)
    ids = e["eidx"].to_numpy()
    lut = np.full(int(ent["eidx"].max()) + 1, -1, dtype=np.int64)
    lut[ids] = np.arange(len(ids))
    return {"d": d, "li": lut[d["eidx"].to_numpy()], "y": d["label"].to_numpy().astype(np.int8),
            "n_true": e["n_true"].to_numpy(), "country": e["country"].to_numpy(), "n_ent": len(ids)}


def X_of(S, feats):
    return S["d"].select(feats).to_numpy().astype(np.float32)


# --------------------------------------------------------------------------- metric
def per_entity(probs, S, thr, extra=None):
    """Leaderboard metric, vectorised. Denominator = FULL ground truth per entity."""
    E, li, y, n_true = S["n_ent"], S["li"], S["y"], S["n_true"]
    keep = probs >= thr
    if extra is not None:
        keep = keep & extra
    pred = np.bincount(li[keep], minlength=E)
    tp = np.bincount(li[keep & (y == 1)], minlength=E)
    P = np.where(pred > 0, tp / np.maximum(pred, 1), 0.0)
    R = np.where(n_true > 0, tp / np.maximum(n_true, 1), 0.0)
    den = 0.25 * P + R
    F = np.where(den > 0, 1.25 * P * R / np.where(den > 0, den, 1.0), 0.0)
    F = np.where((n_true == 0) & (pred == 0), 1.0, F)   # correct empty
    F = np.where((n_true == 0) & (pred > 0), 0.0, F)    # false merge on a singleton
    return F, pred, tp


def winner_mask(S, probs):
    """One-record-one-business: a candidate may only be kept for the S1 entity that scores it highest.
    (Verified on train ground truth: 0 of 7,638,365 true S2/S3 ids belong to more than one S1 entity.)"""
    t = pl.DataFrame({"c": S["d"]["candidate_entity_id"], "p": probs})
    return t.with_columns((pl.col("p") >= pl.col("p").max().over("c")).alias("w"))["w"].to_numpy()


def sweep(probs, S, extra=None):
    best_t, best_f, curve = None, -1.0, []
    for t in THRESHOLDS:
        f = float(per_entity(probs, S, t, extra)[0].mean())
        curve.append((float(t), f))
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f, curve


def bootstrap_ci(F, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(F)
    means = np.array([F[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def calibration(probs, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf, acc, w = float(probs[m].mean()), float(y[m].mean()), float(m.mean())
        ece += w * abs(conf - acc)
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "mean_pred": conf, "frac_positive": acc, "n": int(m.sum())})
    return {"ece": float(ece), "bins": rows}


def full_report(probs, S, thr):
    F, pred, tp = per_entity(probs, S, thr)
    n_true, cand = S["n_true"], np.bincount(S["li"], minlength=S["n_ent"])
    pos_present = np.bincount(S["li"][S["y"] == 1], minlength=S["n_ent"])
    has, single = n_true > 0, n_true == 0

    def seg(mask):
        return {"n_entities": int(mask.sum()), "macro_f05": float(F[mask].mean()) if mask.any() else None}

    strata = {f"country={c}": seg(S["country"] == c) for c in np.unique(S["country"])}
    strata.update({
        "true_matches=0 (singleton)": seg(n_true == 0), "true_matches=1": seg(n_true == 1),
        "true_matches=2-3": seg((n_true >= 2) & (n_true <= 3)), "true_matches=4-5": seg((n_true >= 4) & (n_true <= 5)),
        "true_matches>=6": seg(n_true >= 6),
        "candidates=0": seg(cand == 0), "candidates=1-9": seg((cand >= 1) & (cand <= 9)),
        "candidates=10-19": seg((cand >= 10) & (cand <= 19)), "candidates>=20 (fat tail)": seg(cand >= 20)})

    funnel = {"true_pairs_total": int(n_true.sum()),
              "candidate_recall_L1_L2": float(pos_present.sum() / max(1, n_true.sum())),
              "final_recall": float(tp.sum() / max(1, n_true.sum())),
              "final_precision_micro": float(tp.sum() / max(1, pred.sum())),
              "by_country": {str(c): {"candidate_recall": float(pos_present[S["country"] == c].sum() / max(1, n_true[S["country"] == c].sum())),
                                      "final_recall": float(tp[S["country"] == c].sum() / max(1, n_true[S["country"] == c].sum()))}
                             for c in np.unique(S["country"])}}
    return {
        "threshold": thr, "n_entities": int(S["n_ent"]),
        "macro_f05": float(F.mean()), "macro_f05_ci95": bootstrap_ci(F),
        "singleton_correct_empty_rate": float((pred[single] == 0).mean()) if single.any() else None,
        "has_match_predicted_empty_rate": float((pred[has] == 0).mean()) if has.any() else None,
        "floor_all_empty_f05": float(single.mean()),
        "ceiling_perfect_ranker_f05": float(per_entity(S["y"].astype(float), S, 0.5)[0].mean()),
        "funnel": funnel, "strata": strata,
        "parsimony": {"avg_candidates": float(cand.mean()), "median": float(np.median(cand)),
                      "p95": float(np.percentile(cand, 95)), "max": int(cand.max()),
                      "pct_entities_zero_candidates": float((cand == 0).mean() * 100)},
        "calibration": calibration(probs, S["y"]),
        "_per_entity_f05": F,
    }


# --------------------------------------------------------------------------- model
def fit(S_tr, S_tu, feats, device):
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, objective="binary:logistic",
                            tree_method="hist", device=device, eval_metric="logloss",
                            early_stopping_rounds=20, random_state=42)
    clf.fit(X_of(S_tr, feats), S_tr["y"], eval_set=[(X_of(S_tu, feats), S_tu["y"])], verbose=50)
    return clf


def run_ablations(df, ent, device, base_f05, train_frac):
    groups = {"no_relative_features": [f for f in ALL_FEATS if f not in REL],
              "no_dense_embeddings": [f for f in ALL_FEATS if f not in ("name_cos", "addr_cos", "addr_cos_gap_from_top", "name_cos_gap_from_top", "addr_cos_rank")],
              "no_tfidf": [f for f in ALL_FEATS if "tfidf" not in f],
              "no_levenshtein": [f for f in ALL_FEATS if "lev" not in f],
              "no_numeric_jaccard": [f for f in ALL_FEATS if f != "num_jac"]}
    S_tr, S_tu, S_ho = build_set(df, ent, 0, train_frac=train_frac), build_set(df, ent, 1), build_set(df, ent, 2)
    out = {}
    for name, feats in groups.items():
        print(f"\n[ablation] {name} ({len(feats)} features)")
        clf = fit(S_tr, S_tu, feats, device)
        thr, _, _ = sweep(clf.predict_proba(X_of(S_tu, feats))[:, 1], S_tu)
        f = float(per_entity(clf.predict_proba(X_of(S_ho, feats))[:, 1], S_ho, thr)[0].mean())
        out[name] = {"holdout_macro_f05": f, "delta_vs_full": f - base_f05}
        print(f"  holdout F0.5 = {f:.4f}  (delta {f - base_f05:+.4f})")
    return out


def run_transfer(df, ent, device, in_domain, train_frac):
    """Train on one country only, test on the other. A PROXY for the unseen-France case:
    it stresses the ranker only (L1 embeddings and the L2 pruner already saw both countries)."""
    out = {}
    for src, tgt in [("US", "India"), ("India", "US")]:
        S_tr, S_tu = build_set(df, ent, 0, src, train_frac), build_set(df, ent, 1, src)
        S_ho = build_set(df, ent, 2, tgt)
        if S_tr["n_ent"] == 0 or S_ho["n_ent"] == 0:
            continue
        print(f"\n[transfer] train on {src} only -> test on {tgt}")
        clf = fit(S_tr, S_tu, ALL_FEATS, device)
        thr_src, _, _ = sweep(clf.predict_proba(X_of(S_tu, ALL_FEATS))[:, 1], S_tu)
        p = clf.predict_proba(X_of(S_ho, ALL_FEATS))[:, 1]
        f_src = float(per_entity(p, S_ho, thr_src)[0].mean())
        thr_or, f_or, _ = sweep(p, S_ho)
        base = in_domain["strata"].get(f"country={tgt}", {}).get("macro_f05")
        out[f"{src}->{tgt}"] = {"threshold_from_source": thr_src, "f05_with_source_threshold": f_src,
                                "oracle_threshold_on_target": thr_or, "f05_with_oracle_threshold": f_or,
                                "in_domain_model_f05_on_target": base,
                                "generalization_gap": None if base is None else base - f_src,
                                "threshold_shift_cost": f_or - f_src}
        print(f"  F0.5 (source thr {thr_src:.2f}) = {f_src:.4f} | oracle thr {thr_or:.2f} = {f_or:.4f} | in-domain = {base}")
    return out


def make_plots(out_dir, curve_tune, curve_ho, rep):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plots] skipped ({e})")
        return
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].plot(*zip(*curve_tune), label="tune"); ax[0, 0].plot(*zip(*curve_ho), label="holdout")
    ax[0, 0].axvline(rep["threshold"], ls="--", c="gray"); ax[0, 0].set(title="Macro F0.5 vs threshold", xlabel="threshold", ylabel="F0.5"); ax[0, 0].legend()
    cb = rep["calibration"]["bins"]
    ax[0, 1].plot([0, 1], [0, 1], "k--"); ax[0, 1].plot([b["mean_pred"] for b in cb], [b["frac_positive"] for b in cb], "o-")
    ax[0, 1].set(title=f"Reliability (ECE={rep['calibration']['ece']:.3f})", xlabel="predicted P(match)", ylabel="observed")
    keys = [k for k in rep["strata"] if k.startswith("country=") or k.startswith("true_matches")]
    vals = [rep["strata"][k]["macro_f05"] or 0 for k in keys]
    ax[1, 0].barh(keys, vals); ax[1, 0].set(title="Macro F0.5 by segment (holdout)", xlim=(0, 1))
    f = rep["funnel"]
    ax[1, 1].bar(["candidate recall\n(L1+L2 ceiling)", "final recall", "final precision"],
                 [f["candidate_recall_L1_L2"], f["final_recall"], f["final_precision_micro"]])
    ax[1, 1].set(title="Where recall is lost", ylim=(0, 1))
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "eval_summary.png"), dpi=150)
    print(f"[plots] saved {out_dir}/eval_summary.png")


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="output/ranker_features_train.parquet")
    ap.add_argument("--s1", default="dataset/train/train_source1.tsv")
    ap.add_argument("--gt", default="dataset/train/train_ground_truth.tsv")
    ap.add_argument("--out-dir", default="output/eval")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-frac", type=float, default=1.0, help="subsample train entities to save RAM/time")
    ap.add_argument("--no-round", action="store_true", help="do NOT round features to 4 decimals")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--save-model", action="store_true", help="save v2 ranker + threshold to output/")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()

    s1, ent, pairs = load_entities(a.s1, a.gt)
    ent = assign_partitions(s1, ent)
    df = load_features(a.features, ent, pairs, round4=not a.no_round)
    print(f"[data] {df.height:,} candidate pairs | positives present in candidate set: {int(df['label'].sum()):,} "
          f"of {int(ent['n_true'].sum()):,} true pairs")

    S_tr, S_tu, S_ho = build_set(df, ent, 0, train_frac=a.train_frac), build_set(df, ent, 1), build_set(df, ent, 2)
    clf = fit(S_tr, S_tu, ALL_FEATS, a.device)
    p_tu = clf.predict_proba(X_of(S_tu, ALL_FEATS))[:, 1]
    p_ho = clf.predict_proba(X_of(S_ho, ALL_FEATS))[:, 1]
    thr, f_tune, curve_tune = sweep(p_tu, S_tu)
    curve_ho = [(t, float(per_entity(p_ho, S_ho, t)[0].mean())) for t in THRESHOLDS]
    rep = full_report(p_ho, S_ho, thr)
    rep["tune_macro_f05_at_threshold"] = f_tune
    rep["oracle_threshold_on_holdout"] = max(curve_ho, key=lambda x: x[1])
    w_tu, w_ho = winner_mask(S_tu, p_tu), winner_mask(S_ho, p_ho)
    thr_cr, _, _ = sweep(p_tu, S_tu, w_tu)
    F_cr = per_entity(p_ho, S_ho, thr_cr, w_ho)[0]
    rep["conflict_resolution"] = {"threshold": thr_cr, "macro_f05": float(F_cr.mean()),
                                  "delta_vs_plain": float(F_cr.mean() - rep["macro_f05"]),
                                  "note": "LOWER BOUND: only holdout entities compete for a candidate; at test time all S1 entities compete."}
    rep["feature_importance"] = {f: float(v) for f, v in zip(ALL_FEATS, clf.feature_importances_)}

    print("\n" + "=" * 70 + "\nHOLDOUT RESULTS (never used for training, early stopping or threshold)\n" + "=" * 70)
    print(f"Entities: {rep['n_entities']:,} | threshold (chosen on tune): {thr:.2f}")
    print(f"Macro F0.5 : {rep['macro_f05']:.4f}   95% CI {rep['macro_f05_ci95'][0]:.4f} - {rep['macro_f05_ci95'][1]:.4f}")
    print(f"  floor (predict nothing)      : {rep['floor_all_empty_f05']:.4f}")
    print(f"  ceiling (perfect ranker on candidate set): {rep['ceiling_perfect_ranker_f05']:.4f}   <- upper bound imposed by L1+L2 recall")
    f = rep["funnel"]
    print(f"Recall funnel: candidate-set {f['candidate_recall_L1_L2']:.4f} -> final {f['final_recall']:.4f} | precision {f['final_precision_micro']:.4f}")
    print(f"Singletons predicted empty: {rep['singleton_correct_empty_rate']}  | has-match predicted empty: {rep['has_match_predicted_empty_rate']}")
    print(f"Parsimony: {rep['parsimony']}")
    print(f"Calibration ECE: {rep['calibration']['ece']:.4f}")
    cr = rep["conflict_resolution"]
    print(f"One-candidate-one-owner step: F0.5 {cr['macro_f05']:.4f} (delta {cr['delta_vs_plain']:+.4f}, thr {cr['threshold']:.2f}) [lower bound]")
    print("\nSegments:")
    for k, v in rep["strata"].items():
        print(f"  {k:32s} n={v['n_entities']:>8,}  F0.5={v['macro_f05'] if v['macro_f05'] is None else round(v['macro_f05'], 4)}")

    if a.ablate:
        rep["ablations"] = run_ablations(df, ent, a.device, rep["macro_f05"], a.train_frac)
    if a.transfer:
        rep["country_transfer"] = run_transfer(df, ent, a.device, rep, a.train_frac)

    make_plots(a.out_dir, curve_tune, curve_ho, rep)
    rep.pop("_per_entity_f05")
    with open(os.path.join(a.out_dir, "eval_results.json"), "w") as fh:
        json.dump(rep, fh, indent=2, default=float)
    if a.save_model:
        clf.save_model("output/phase3_ranker_xgb_v2.json")
        json.dump({"optimal_threshold": thr}, open("output/phase3_threshold_v2.json", "w"))
        print("Saved output/phase3_ranker_xgb_v2.json + phase3_threshold_v2.json")
    print(f"\nWrote {a.out_dir}/eval_results.json  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
