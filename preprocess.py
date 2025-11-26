import pandas as pd
import numpy as np
import os

RAW_PATH = "/data/raw"
OUTPUT_PATH = "/data/processed/combined.csv"
CHUNK_SIZE = 100_000   # safe for Colab (adjust if needed)


def build_master_column_list():
    """Scan first row of each CSV to get all column names."""
    all_cols = set()

    for file in os.listdir(RAW_PATH):
        if file.endswith(".csv"):
            path = os.path.join(RAW_PATH, file)
            print(f"[+] Reading header: {file}")
            try:
                cols = pd.read_csv(path, nrows=1).columns
                all_cols.update(cols)
            except Exception as e:
                print(f"[!] Error: {e}")

    master_cols = sorted(list(all_cols))
    print(f"[✓] Total unique columns: {len(master_cols)}\n")
    return master_cols


def process_file_in_chunks(file, master_columns, write_header):
    """Load CSV in chunks, align columns, clean, and append to output."""
    path = os.path.join(RAW_PATH, file)
    print(f"\n=== Processing {file} in chunks ===")

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):

        # Align columns
        chunk = chunk.reindex(columns=master_columns)

        # Clean chunk
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.fillna(0, inplace=True)

        # Append to CSV
        chunk.to_csv(OUTPUT_PATH, mode="a", index=False, header=write_header)
        write_header = False  # Only first chunk writes header

    print(f"[✓] Finished: {file}")
    return write_header


def main():
    print("\n=== PREPROCESSING CICIDS-2017 (MEMORY-SAFE) ===\n")

    os.makedirs("data/processed", exist_ok=True)

    # Remove old combined file if exists
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    # Step 1: Build column master list
    master_columns = build_master_column_list()

    # Step 2: Process each file safely in chunks
    write_header = True
    for file in sorted(os.listdir(RAW_PATH)):
        if file.endswith(".csv"):
            write_header = process_file_in_chunks(file, master_columns, write_header)

    print("\n[✓] All files processed successfully!")
    print(f"[✓] Output saved to: {OUTPUT_PATH}")
    print("\n=== PREPROCESSING COMPLETE ===")


if __name__ == "__main__":
    main()
