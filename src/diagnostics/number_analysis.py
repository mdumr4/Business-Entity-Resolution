import polars as pl
from pathlib import Path
import re

def main():
    print("Extracting empirical examples of numbers from the training data...\n")
    
    data_dir = Path("dataset/train")
    
    # Load a small chunk to analyze
    s1 = pl.read_csv(data_dir / "train_source1.tsv", separator="\t", n_rows=100000)
    
    # We want to find distinct tokens (words) that contain numbers.
    # We will extract them using regex.
    def extract_number_tokens(text):
        if not text: return []
        # Find any word that has at least one digit in it
        tokens = re.findall(r'\b\w*\d\w*\b', str(text))
        return tokens
        
    print("Analyzing Business Names...")
    name_numbers = []
    for name in s1["business_name"].to_list():
        if name:
            nums = extract_number_tokens(name)
            if nums: name_numbers.append((name, nums))
            
    print("Analyzing Business Addresses...")
    addr_numbers = []
    for addr in s1["business_address"].to_list():
        if addr:
            nums = extract_number_tokens(addr)
            if nums: addr_numbers.append((addr, nums))
            
    import random
    
    print(f"\nFound numbers in {len(name_numbers):,} Business Names out of 100,000")
    print("--- 10 Random Examples of Numbers in NAMES ---")
    for name, nums in random.sample(name_numbers, min(10, len(name_numbers))):
        print(f"Tokens {nums} found in: '{name}'")
        
    print(f"\nFound numbers in {len(addr_numbers):,} Business Addresses out of 100,000")
    print("--- 10 Random Examples of Numbers in ADDRESSES ---")
    for addr, nums in random.sample(addr_numbers, min(10, len(addr_numbers))):
        print(f"Tokens {nums} found in: '{addr}'")

if __name__ == "__main__":
    main()
