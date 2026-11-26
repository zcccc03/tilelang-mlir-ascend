"""One-launch CAModel cycle diagnostics, with actual model metadata retained."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("binaries", nargs="+")
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(args.output)
root.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
camodel = "/usr/local/Ascend/cann/tools/simulator/dav_3510/camodel"
env["LD_LIBRARY_PATH"] = camodel + ":" + env.get("LD_LIBRARY_PATH", "")
summary = []
for item in args.binaries:
    binary = Path(item).resolve()
    tag = binary.parent.parent.name + "-" + binary.parent.name
    directory = root / tag
    directory.mkdir(exist_ok=True)
    command = ["npusim", "record", "-s", "Ascend950", "-o", str(directory / "record"), "-n", "all", str(binary) + " profile"]
    start = time.monotonic()
    with (directory / "profile.log").open("w") as log:
        try:
            rc = subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=240).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    record = {
        "binary": str(binary),
        "command": command,
        "exit_code": rc,
        "host_wall_seconds": time.monotonic() - start,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "kind": "simulator_diagnostic_not_device_performance",
        "numerical_pass": "NUMERICAL_PASS" in (directory / "profile.log").read_text(),
        "warmup": 0,
        "launches": 1,
        "databases": [],
    }
    for database in directory.rglob("*.db"):
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        entry = {"path": str(database), "tables": tables}
        if "KernelInfo" in tables:
            entry["kernel_info"] = [dict(row) for row in connection.execute("SELECT * FROM KernelInfo")]
        connection.close()
        record["databases"].append(entry)
    (directory / "profile.json").write_text(json.dumps(record, indent=2, default=str))
    summary.append(record)
    (root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(record, default=str), flush=True)
    if rc != 0 or not record["numerical_pass"]:
        print((directory / "profile.log").read_text()[-4000:], flush=True)
        raise SystemExit(1)
