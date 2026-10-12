import os, sys, tempfile
os.environ.update(AVASOC_ENV="dev", AVASOC_PBKDF2_ITERS="1000", AVASOC_DATA_DIR=tempfile.mkdtemp(prefix="avasoc-test-"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
