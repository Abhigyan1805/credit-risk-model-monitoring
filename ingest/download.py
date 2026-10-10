"""Download LendingClub accepted loans via the kaggle CLI. ``python -m ingest.download``

The token is read from ~/.kaggle/access_token at runtime and passed to the CLI through the
environment only; it is never written anywhere. Skips the download when the file already exists.
"""

from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path

from common.paths import raw_dir

DATASET = "wordsforthewise/lending-club"
ACCEPTED = "accepted_2007_to_2018Q4.csv.gz"


def main() -> Path:
    out = raw_dir()
    out.mkdir(parents=True, exist_ok=True)
    target = out / ACCEPTED
    if target.exists():
        print(f"already downloaded: {target}")
        return target
    env = dict(os.environ)
    token_file = Path.home() / ".kaggle" / "access_token"
    if "KAGGLE_API_TOKEN" not in env and token_file.exists():
        env["KAGGLE_API_TOKEN"] = token_file.read_text().strip()
    zpath = out / "lending-club.zip"
    if not zpath.exists():
        subprocess.run(["kaggle", "datasets", "download", "-d", DATASET, "-p", str(out)], check=True, env=env)
    with zipfile.ZipFile(zpath) as z:
        z.extract(ACCEPTED, out)
    zpath.unlink()
    return target


if __name__ == "__main__":
    main()
