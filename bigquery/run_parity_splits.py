import json
import subprocess
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
    print("--- Comparing splits against ref_splits (source = 'main') ---")
    proc = subprocess.run(
        ['bq', 'show', '--schema', '--format=json', 'river-data-511316-j4:gw_stress.ref_splits'],
        capture_output=True, text=True, shell=True, check=True
    )
    schema = json.loads(proc.stdout)
    columns = [f["name"] for f in schema]
    col_types = {f["name"]: f["type"] for f in schema}

    metrics = []
    for col in columns:
        clause = f"""
          COUNTIF(
            (a.{col} IS NULL AND b.{col} IS NOT NULL) OR
            (a.{col} IS NOT NULL AND b.{col} IS NULL) OR
            (a.{col} != b.{col})
          ) AS {col}__mismatches
        """
        metrics.append(clause)

    sql = f"""
    SELECT
      COUNT(*) AS total_joined,
      {', '.join(metrics)}
    FROM `river-data-511316-j4.gw_stress.splits` a
    JOIN `river-data-511316-j4.gw_stress.ref_splits` b
      ON a.well_id = b.well_id AND a.period_label = b.period_label
    WHERE b.source = 'main'
    """
    res = run_query(sql)[0]
    total = int(res["total_joined"])
    print(f"Total joined main rows: {total}")

    results = []
    for col in columns:
        m = int(res[f"{col}__mismatches"])
        print(f"  - {col}: mismatches={m}")
        results.append({"column": col, "type": col_types[col], "mismatches": m, "total": total})

    df = pd.DataFrame(results)
    df.to_csv("parity_splits.csv", index=False)
    print(f"Splits exact matching columns: {(df['mismatches'] == 0).sum()} / {len(df)}")

if __name__ == "__main__":
    main()
