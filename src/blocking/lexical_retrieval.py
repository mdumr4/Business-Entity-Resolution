import polars as pl
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Import our custom rule-compliant normalizer
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.features.address_normalizer import prepare_dataframe

def build_combined_text(df: pl.DataFrame) -> list:
    """Creates the text field for TF-IDF by joining normalized name and address."""
    return df.with_columns(
        pl.concat_str(
            [pl.col("norm_name"), pl.lit(" "), pl.col("norm_address")]
        ).alias("lexical_text")
    )["lexical_text"].to_list()

def main(data_dir: str, output_dir: str):
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    print("Loading datasets for Lexical Blocking...")
    prefix = "test" if "test" in data_dir.lower() else "train"
    s1 = prepare_dataframe(pl.read_csv(data_path / f"{prefix}_source1.tsv", separator="\t"))
    s2 = prepare_dataframe(pl.read_csv(data_path / f"{prefix}_source2.tsv", separator="\t"))
    s3 = prepare_dataframe(pl.read_csv(data_path / f"{prefix}_source3.tsv", separator="\t"))
    
    candidates_df = pl.concat([s2, s3])
    candidate_ids = candidates_df["entity_id"].to_numpy()
    
    print("Building Lexical text corpus...")
    s1_texts = build_combined_text(s1)
    candidate_texts = build_combined_text(candidates_df)
    
    # Rule confirmation: Computing unsupervised text statistics over the corpus is permitted!
    print("Fitting TF-IDF Vectorizer on full candidate corpus...")
    # Switched to 'word' analyzer to prevent 10GB RAM Out-of-Memory crash on 10M rows.
    vectorizer = TfidfVectorizer(analyzer='word', min_df=2)
    
    candidate_tfidf = vectorizer.fit_transform(candidate_texts)
    print(f"Candidate TF-IDF Matrix shape: {candidate_tfidf.shape}")
    
    print("Transforming S1 Queries...")
    query_tfidf = vectorizer.transform(s1_texts)
    
    # Block batching to avoid RAM OOM on exact sparse matrix multiplication
    TOP_K = 20
    results = []
    
    BATCH_SIZE = 10
    num_queries = query_tfidf.shape[0]
    
    print("Computing Sparse Cosine Similarity in batches...")
    for start_idx in tqdm(range(0, num_queries, BATCH_SIZE), desc="Lexical Search"):
        end_idx = min(start_idx + BATCH_SIZE, num_queries)
        
        # Sparse matrix multiplication for fast exact search
        q_batch = query_tfidf[start_idx:end_idx]
        sim_matrix = q_batch.dot(candidate_tfidf.T)
        
        # NEVER call .toarray() here! It takes 40GB RAM! 
        # Extract top K directly from the sparse CSR format row by row
        for i in range(sim_matrix.shape[0]):
            # Get the sparse row
            row_data = sim_matrix.data[sim_matrix.indptr[i]:sim_matrix.indptr[i+1]]
            row_indices = sim_matrix.indices[sim_matrix.indptr[i]:sim_matrix.indptr[i+1]]
            
            if len(row_data) > 0:
                # Find top K indices within this sparse row
                k = min(TOP_K, len(row_data))
                top_k_local_idx = np.argpartition(row_data, -k)[-k:]
                
                # Sort them by score
                top_k_local_idx = top_k_local_idx[np.argsort(row_data[top_k_local_idx])[::-1]]
                
                # Map back to global candidate indices
                top_indices = row_indices[top_k_local_idx]
                retrieved_ids = candidate_ids[top_indices]
            else:
                retrieved_ids = []
                
            results.append({
                "source1_entity_id": s1["entity_id"][start_idx + i],
                "candidate_entity_ids": ",".join(retrieved_ids)
            })
            
    final_df = pl.DataFrame(results)
    output_file = out_path / "candidate_pairs_lexical.tsv"
    final_df.write_csv(output_file, separator="\t")
    print(f"Success! Saved lexical candidates to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset/train")
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()
    
    main(args.data_dir, args.output_dir)
