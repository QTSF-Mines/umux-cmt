"""Execute the example notebooks in order and save them with their outputs.

    python scripts/run_examples.py             # all notebooks
    python scripts/run_examples.py 04 06       # only those whose name starts with 04 or 06

Each notebook runs in its own folder (examples/) with a fresh kernel. The
order matters for a handful of numbers that one notebook records for the next
in examples/results/values.json (see examples/README.md); run everything once
in order, after that any notebook can be rerun on its own.
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
ORDER = sorted(p for p in EXAMPLES.glob("[0-9][0-9]_*.ipynb"))


def run(path, timeout=3600):
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}})
    t0 = time.time()
    client.execute()
    nbformat.write(nb, path)
    return time.time() - t0


if __name__ == "__main__":
    prefixes = tuple(sys.argv[1:])
    todo = [p for p in ORDER if not prefixes or p.name.startswith(prefixes)]
    for path in todo:
        print(f"running {path.name} ...", end=" ", flush=True)
        dt = run(path)
        print(f"done in {dt:.0f} s")
