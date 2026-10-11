import json
import subprocess
import time
import pandas as pd

def run_query(sql):
    proc = subprocess.Popen(
        ['bq', 'query', '--nouse_legacy_sql', '--format=json'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True
    )
    stdout, stderr = proc.communicate(input=sql)
    if proc.returncode != 0:
        raise RuntimeError(f"Query failed: {stderr}")
    return json.loads(stdout)

def main():
    print("--- Fetching schema of ref_final_table ---")
    proc = subprocess.run(
        ['bq', 'show', '--schema', '--format=json', 'river-data-511316-j4:gw_stress.ref_final_table'],
        capture_output=True, text=True, shell=True, check=True
    )
    schema = json.loads(proc.stdout)
    columns = [f["name"] for f in schema]
    col_types = {f["name"]: f["type"] for f in schema}
    print(f"Total columns in schema: {len(columns)}")

    results = []

    # Process columns in batches to be fast and cost-effective
    # We can group columns into 6 batches of ~14 columns
    batch_size = 12
    for i in range(0, len(columns), batch_size):
        batch = columns[i:i+batch_size]
        print(f"\nComparing batch {i//batch_size + 1}: {', '.join(batch)}...")

        metrics_clauses = []
        for col in batch:
            ctype = col_types[col]
            if ctype in ("FLOAT", "FLOAT64"):
                # Both null: ok
                # One null: mismatch
                # Both not null: check abs diff and rel diff
                clause = f"""
                  COUNTIF(
                    (a.{col} IS NULL AND b.{col} IS NOT NULL) OR
                    (a.{col} IS NOT NULL AND b.{col} IS NULL) OR
                    (a.{col} IS NOT NULL AND b.{col} IS NOT NULL AND
                     ABS(a.{col} - b.{col}) > 1e-9 AND
                     ABS(a.{col} - b.{col}) / NULLIF(GREATEST(ABS(a.{col}), ABS(b.{col})), 0) > 1e-9)
                  ) AS {col}__mismatches,
                  MAX(CASE WHEN a.{col} IS NOT NULL AND b.{col} IS NOT NULL THEN ABS(a.{col} - b.{col}) END) AS {col}__max_abs,
                  MAX(CASE WHEN a.{col} IS NOT NULL AND b.{col} IS NOT NULL AND GREATEST(ABS(a.{col}), ABS(b.{col})) > 0
                           THEN ABS(a.{col} - b.{col}) / GREATEST(ABS(a.{col}), ABS(b.{col})) END) AS {col}__max_rel
                """
            else:
                # String, Int, Bool, Timestamp
                clause = f"""
                  COUNTIF(
                    (a.{col} IS NULL AND b.{col} IS NOT NULL) OR
                    (a.{col} IS NOT NULL AND b.{col} IS NULL) OR
                    (a.{col} != b.{col})
                  ) AS {col}__mismatches,
                  CAST(NULL AS FLOAT64) AS {col}__max_abs,
                  CAST(NULL AS FLOAT64) AS {col}__max_rel
                """
            metrics_clauses.append(clause)

        sql = f"""
        SELECT
          COUNT(*) AS total_joined_rows,
          {', '.join(metrics_clauses)}
        FROM `river-data-511316-j4.gw_stress.final_features` a
        JOIN `river-data-511316-j4.gw_stress.ref_final_table` b
          ON a.well_id = b.well_id AND a.period_label = b.period_label
        """

        res = run_query(sql)[0]
        total_rows = int(res["total_joined_rows"])

        for col in batch:
            mismatches = int(res[f"{col}__mismatches"])
            max_abs = float(res[f"{col}__max_abs"]) if res[f"{col}__max_abs"] is not None else None
            max_rel = float(res[f"{col}__max_rel"]) if res[f"{col}__max_rel"] is not None else None
            results.append({
                "column": col,
                "type": col_types[col],
                "total_rows": total_rows,
                "mismatches": mismatches,
                "max_abs_diff": max_abs,
                "max_rel_diff": max_rel,
                "status": "EXACT MATCH" if mismatches == 0 else f"{mismatches} MISMATCHES"
            })
            print(f"  - {col} ({col_types[col]}): mismatches={mismatches}, max_abs={max_abs}, max_rel={max_rel}")

    df_res = pd.DataFrame(results)
    df_res.to_csv("parity_final_features.csv", index=False)
    print("\nSaved parity_final_features.csv")
    print(f"Exact match count: {(df_res['mismatches'] == 0).sum()} / {len(df_res)}")

if __name__ == "__main__":
    main()
