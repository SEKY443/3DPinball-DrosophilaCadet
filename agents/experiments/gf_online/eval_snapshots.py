"""Evaluate the partial Colab online-learning chains locally at a common training point.

The Colab VM running run_online_pool.py was terminated before the end, so only the snapshots downloaded
mid-run exist (results_partial/, 300-500 updates per chain). For a fair comparison every chain is cut back to
the same update count (--at, default 300, the smallest available), its mu/theta_on taken from that update of
its own history, and then run through online_gf.run_chain's normal final evaluation (96 paired lives,
EVAL_SEEDS) without further training. Outputs go to results_eval<at>/; summarize with summarize_online.py.

Usage: .venv/bin/python agents/experiments/gf_online/eval_snapshots.py --at 300 --workers 8
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _init() -> None:
	import torch

	torch.set_num_threads(1)
	import online_gf

	online_gf.G.init_worker()


def _work(task: tuple[str, int, int, str]) -> str:
	import online_gf

	circuit, chain, at, out = task
	return online_gf.run_chain(circuit, chain, at, out)


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--src", default=os.path.join(HERE, "results_partial"))
	ap.add_argument("--at", type=int, default=300)
	ap.add_argument("--workers", type=int, default=os.cpu_count())
	args = ap.parse_args()
	out = os.path.join(HERE, f"results_eval{args.at}")
	os.makedirs(out, exist_ok=True)
	tasks = []
	for f in sorted(glob.glob(os.path.join(args.src, "*_c*.json"))):
		with open(f, encoding="utf-8") as fh:
			st = json.load(fh)
		if st["done_updates"] < args.at:
			print(f"skip {os.path.basename(f)}: only {st['done_updates']} updates", flush=True)
			continue
		h = st["history"][args.at - 1]
		st.update(mu=h["mu"], theta_on=h["theta_on"], done_updates=args.at, history=st["history"][: args.at], eval=None)
		with open(os.path.join(out, os.path.basename(f)), "w", encoding="utf-8") as fh:
			json.dump(st, fh)
		tasks.append((st["circuit"], st["chain"], args.at, out))
	with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
		for p in pool.imap_unordered(_work, tasks):
			print("done", os.path.basename(p), flush=True)


if __name__ == "__main__":
	main()
