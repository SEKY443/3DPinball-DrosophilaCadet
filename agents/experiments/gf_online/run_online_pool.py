"""Runs many online chains in parallel (spawn pool, one chain per worker, torch single-threaded in worker init).

Default: circuits real, shuffled_1..3 x 4 chains = 16 chains, 1500 updates each. Resumable: re-running skips finished
chains and resumes the others from their snapshots.

Usage: python run_online_pool.py --out DIR [--updates 1500] [--workers 16]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _init() -> None:
	import torch

	torch.set_num_threads(1)
	import online_gf

	online_gf.G.init_worker()


def _task(t):
	import online_gf

	circuit, chain, updates, out = t
	try:
		return online_gf.run_chain(circuit, chain, updates, out)
	except Exception:
		print(f"[{circuit} c{chain}] FAILED\n{traceback.format_exc()}", flush=True)
		return None


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--out", required=True)
	ap.add_argument("--updates", type=int, default=1500)
	ap.add_argument("--circuits", default="real,shuffled_1,shuffled_2,shuffled_3")
	ap.add_argument("--chains", type=int, default=4)
	ap.add_argument("--workers", type=int, default=None)
	args = ap.parse_args()
	tasks = [(c, k, args.updates, args.out) for k in range(args.chains) for c in args.circuits.split(",")]
	workers = args.workers or len(tasks)
	print(f"{len(tasks)} chains on {workers} workers, {args.updates} updates each, out {args.out}", flush=True)
	with mp.get_context("spawn").Pool(workers, initializer=_init) as pool:
		res = pool.map(_task, tasks, chunksize=1)
	print(f"done: {sum(r is not None for r in res)}/{len(tasks)} chains finished", flush=True)


if __name__ == "__main__":
	main()
