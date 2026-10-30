"""Run the frozen, model-free verifier and compare the archived summary."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
from pathlib import Path
import platform
import sys
import time
import traceback

OUT = Path(__file__).resolve().parents[1]
BASE = OUT.parents[1]
sys.path.insert(0, str(BASE / "p4_v1"))
import p4_io as disk
import p4_verify as verify
import numpy as np

started = time.monotonic()
meta = dict(started=disk.stamp(), python=sys.version, numpy=np.__version__,
            platform=platform.platform(), threads=2,
            frozen_verifier_sha256=disk.digest(BASE / "p4_v1/p4_verify.py"),
            language_model_imports=False, language_model_forwards=0)
try:
    result = verify.verify(OUT / "raw", OUT / "recomputed_summary.json")
    original = disk.read(OUT / "raw/summary.json")
    # The launcher adds execution/transport metadata to the scientific summary.
    for key in original:
        if key in result and key not in ("audit", "local_receipt_verified"):
            verify.near(result[key], original[key])
    meta.update(status="PASS", archived_summary_agrees=True,
                audit=result["audit"])
except Exception:
    meta.update(status="FAIL", error=traceback.format_exc())
    raise
finally:
    meta.update(finished=disk.stamp(), elapsed_seconds=time.monotonic() - started)
    disk.write(OUT / "verification_execution.json", meta)
print("Frozen saved-observation verification and summary comparison PASS.", flush=True)
