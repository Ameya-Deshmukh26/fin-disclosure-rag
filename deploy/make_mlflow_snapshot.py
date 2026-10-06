"""
Package a read-only snapshot of the local MLflow store for the hosted dashboard.

The local store records artifact locations as absolute Windows paths
(file:///C:/Users/.../mlartifacts/...). Inside the Linux container the store is copied to
/tmp/mlflow, so this rewrites every artifact URI in a COPY of the database to point there.
The original mlflow.db is never modified.

    python deploy/make_mlflow_snapshot.py   ->  mlflow_snapshot/{mlflow.db, mlartifacts/}
"""
import pathlib
import shutil
import sqlite3

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DB, SRC_ART = ROOT / "mlflow.db", ROOT / "mlartifacts"
OUT = ROOT / "mlflow_snapshot"
TARGET = "file:///tmp/mlflow/mlartifacts"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    shutil.copy2(SRC_DB, OUT / "mlflow.db")
    shutil.copytree(SRC_ART, OUT / "mlartifacts")
    local_prefix = SRC_ART.as_uri()          # file:///C:/Users/.../mlartifacts
    con = sqlite3.connect(OUT / "mlflow.db")
    n_exp = con.execute("UPDATE experiments SET artifact_location = REPLACE(artifact_location, ?, ?)",
                        (local_prefix, TARGET)).rowcount
    n_run = con.execute("UPDATE runs SET artifact_uri = REPLACE(artifact_uri, ?, ?)",
                        (local_prefix, TARGET)).rowcount
    con.commit()
    left = con.execute("SELECT COUNT(*) FROM runs WHERE artifact_uri LIKE 'file:///C:%'").fetchone()[0]
    runs = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    con.close()
    print(f"snapshot: {runs} runs, rewrote {n_exp} experiment + {n_run} run artifact URIs, "
          f"{left} still pointing at Windows paths")


if __name__ == "__main__":
    main()
