import polars as pl
from pathlib import Path

def main():
    print("Testing if numbers are preserved identically across True Matches...\n")
    
    data_dir = Path("dataset/train")
    
    print("Loading data...")
    s1 = pl.read_csv(data_dir / "train_source1.tsv", separator="\t")
    s2 = pl.read_csv(data_dir / "train_source2.tsv", separator="\t")
    s3 = pl.read_csv(data_dir / "train_source3.tsv", separator="\t")
    gt = pl.read_csv(data_dir / "train_ground_truth.tsv", separator="\t")
    
    candidates = pl.concat([s2, s3])
    
    gt_exploded = gt.with_columns(
        pl.col("matched_entity_ids").str.split(",")
    ).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
    
    # Target examples we found in S1
    target_names = [
        "4295 Sixth Blvd Apartments LLC",
        "Plumbers Union Local No 766",
        "72 Hour Massage",
        "77 Datta Mandir Road Bhandup Stores Private Limited"
    ]
    
    target_addrs = [
        "8505 Vista Clara Lane, Albuquerque, NM",
        "42 F. I . E Patparganj Industrial Area, Delhi, East Delhi, Delhi",
        "Kalkaji, New Delhi, Delhi, J-86",
        "Huntertown, 16023 Ballantyne Trail, IN"
    ]
    
    # 1. Check True Matches for specific names
    print("=== TRUE MATCHES: DO THE NUMBERS MATCH EXACTLY? ===")
    
    for name in target_names:
        row = s1.filter(pl.col("business_name") == name)
        if row.shape[0] == 0: continue
        s1_id = row["entity_id"][0]
        
        cands = gt_exploded.filter(pl.col("source1_entity_id") == s1_id)["matched_entity_ids"].to_list()
        print(f"\nS1 Name: {name}")
        for c_id in cands:
            c_row = candidates.filter(pl.col("entity_id") == c_id)
            if c_row.shape[0] > 0:
                print(f" -> True Match Cand Name: {c_row['business_name'][0]}")
                
    for addr in target_addrs:
        row = s1.filter(pl.col("business_address") == addr)
        if row.shape[0] == 0: continue
        s1_id = row["entity_id"][0]
        
        cands = gt_exploded.filter(pl.col("source1_entity_id") == s1_id)["matched_entity_ids"].to_list()
        print(f"\nS1 Addr: {addr}")
        for c_id in cands:
            c_row = candidates.filter(pl.col("entity_id") == c_id)
            if c_row.shape[0] > 0:
                print(f" -> True Match Cand Addr: {c_row['business_address'][0]}")

    # 2. Find False Positives (Numbers match, but they are different businesses)
    print("\n=== FALSE MATCHES: WHEN NUMBERS MATCH BUT BUSINESS IS DIFFERENT ===")
    print("Searching for pairs that share the exact number token 'J-86' or '4295' but are NOT matches...")
    
    s1_token = s1.filter(pl.col("business_address").str.contains("J-86"))
    cand_token = candidates.filter(pl.col("business_address").str.contains("J-86"))
    
    if s1_token.shape[0] > 0 and cand_token.shape[0] > 0:
        s1_example = s1_token.row(0, named=True)
        cand_example = cand_token.row(0, named=True)
        
        # Check if they are a true match
        is_match = gt_exploded.filter(
            (pl.col("source1_entity_id") == s1_example["entity_id"]) & 
            (pl.col("matched_entity_ids") == cand_example["entity_id"])
        ).shape[0] > 0
        
        if not is_match:
            print("\nFound a False Match sharing token 'J-86':")
            print(f"S1:   {s1_example['business_name']} | {s1_example['business_address']}")
            print(f"Cand: {cand_example['business_name']} | {cand_example['business_address']}")
            print("Status: NOT A MATCH (Different Businesses, same building/plot number!)")

if __name__ == "__main__":
    main()
