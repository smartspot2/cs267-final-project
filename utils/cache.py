import os

CACHE_DIR = os.environ.get("SCRATCH", None)
if CACHE_DIR is not None:
    CACHE_DIR = CACHE_DIR.rstrip("/") + "/cache"
