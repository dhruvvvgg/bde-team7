import os
import subprocess
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

states = [
    'Andhra Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Delhi',
    'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
    'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Odisha', 'Punjab',
    'Tamil Nadu', 'Telangana', 'Uttar Pradesh', 'West Bengal'
]

def main():
    os.makedirs('bigquery/exports/chunks', exist_ok=True)
    parquet_path = 'bigquery/exports/final_features.parquet'
    dfs = []
    total_rows = 0
    t0 = time.time()
    
    for i, st in enumerate(states):
        csv_file = f'bigquery/exports/chunks/state_{i}.csv'
        sql = f'SELECT * FROM `river-data-511316-j4.gw_stress.final_features` WHERE state = "{st}" ORDER BY well_id, date;'
        cmd = ['bq', 'query', '--nouse_legacy_sql', '--format=csv', '--max_rows=50000']
        with open(csv_file, 'wb') as f:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=f, stderr=subprocess.PIPE, shell=True)
            stdout, stderr = p.communicate(input=sql.encode('utf-8'))
        if p.returncode != 0:
            raise RuntimeError(f'Failed on {st}: {stderr.decode("utf-8")}')
        
        df = pd.read_csv(csv_file)
        dfs.append(df)
        total_rows += len(df)
        print(f'[{i+1}/19] {st:16s}: {len(df):6,d} rows (cumulative: {total_rows:7,d})', flush=True)
        os.remove(csv_file)
        
    print(f"\nAll 19 states fetched in {time.time()-t0:.2f}s! Concatenating {total_rows:,} rows...", flush=True)
    full_df = pd.concat(dfs, ignore_index=True)
    assert len(full_df) == 253828, f"Row count mismatch: expected 253,828, got {len(full_df)}"
    assert len(full_df.columns) == 84, f"Column count mismatch: expected 84, got {len(full_df.columns)}"
    
    print(f"Writing {parquet_path}...", flush=True)
    full_df.to_parquet(parquet_path, index=False, compression='snappy')
    size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"Successfully generated {parquet_path}: {len(full_df):,} rows x {len(full_df.columns)} cols ({size_mb:.2f} MB)", flush=True)
    assert size_mb < 50, f"File exceeds 50MB: {size_mb:.2f} MB"
    
    os.rmdir('bigquery/exports/chunks')
    print("Export complete!", flush=True)

if __name__ == '__main__':
    main()
