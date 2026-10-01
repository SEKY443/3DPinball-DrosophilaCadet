#!/usr/bin/env python3
"""Run this on the Colab VM after uploading and unzipping pinball.zip: applies patches, builds
the native binary, writes a crash-resilient (flock-protected, per docs/colab_cli_notes.md
gotcha #4) watchdog, and launches the one-ball-life / log-score run of the 444-neuron circuit, resuming from the
included checkpoint (centered on the big_circuit generation-280 champion).

Usage on the Colab VM:
    python3 colab_launch.py
"""
import os
import subprocess
import sys

PROJECT_DIR = "/content/3DPinball-DrosophilaCadet"
CKPT = f"{PROJECT_DIR}/agents/checkpoints/big_v2/pinball_circuit_cma.pt"
WATCHDOG_PATH = "/content/watchdog.sh"
WATCHDOG_LOG = "/content/train.log"
WATCHDOG_STDOUT = "/content/watchdog_stdout.log"
TARGET_GEN = 500  # generous headroom - see colab_cli_notes.md gotcha #6 on bumping proactively


def run(cmd, **kwargs):
	print(f"$ {cmd}", flush=True)
	result = subprocess.run(cmd, shell=True, **kwargs)
	if result.returncode != 0:
		print(f"  -> exited with code {result.returncode}", flush=True)
	return result


def main():
	print("=== 1. unzipping project ===", flush=True)
	run("cd /content && unzip -o -q pinball.zip")

	print("\n=== 2. installing system dependencies ===", flush=True)
	run("apt-get update -qq && apt-get install -y -qq cmake build-essential libsdl2-dev libsdl2-mixer-dev git > /dev/null")

	print("\n=== 3. installing python dependencies ===", flush=True)
	run("pip install -q gymnasium pyarrow cma")

	print("\n=== 4. giving vendor/SpaceCadetPinball a self-contained git context ===", flush=True)
	run(f"cd {PROJECT_DIR}/vendor/SpaceCadetPinball && git init -q && git add -A && "
	    f"git -c user.email=colab@local -c user.name=colab commit -q -m 'vendored snapshot'")

	print("\n=== 5. applying patches ===", flush=True)
	patch_result = run(f"cd {PROJECT_DIR} && bash scripts/apply_patches.sh")
	if patch_result.returncode != 0:
		print("patch application failed - stopping here", flush=True)
		sys.exit(1)

	print("\n=== 6. building native binary ===", flush=True)
	build = run(f"cmake -S {PROJECT_DIR}/vendor/SpaceCadetPinball -B {PROJECT_DIR}/build-native -DCMAKE_BUILD_TYPE=Debug")
	if build.returncode != 0:
		sys.exit(1)
	build = run(f"cmake --build {PROJECT_DIR}/build-native --parallel")
	if build.returncode != 0:
		print("build failed - stopping here, do not launch training on a stale binary", flush=True)
		sys.exit(1)

	binary = f"{PROJECT_DIR}/vendor/SpaceCadetPinball/bin/SpaceCadetPinball"
	if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
		print(f"error: build did not produce {binary}", flush=True)
		sys.exit(1)
	print(f"Built: {binary}", flush=True)

	print("\n=== 7. clearing any stray processes ===", flush=True)
	run("pkill -9 -f watchdog.sh; pkill -9 -f train_pinball_circuit_cem; pkill -9 -f SpaceCadetPinball; true")

	print("\n=== 8. writing flock-protected watchdog ===", flush=True)
	# flock guard (colab_cli_notes.md gotcha #4): a session-suspend/resume can "resurrect" an
	# already-killed watchdog with its original stale flags, running a duplicate trainer against
	# the same checkpoint file. The lock makes an accidental duplicate launch a harmless no-op
	# instead of silent data corruption.
	watchdog_script = f"""#!/bin/bash
exec 200>/tmp/watchdog.lock
if ! flock -n 200; then
  echo "WATCHDOG: $(date) another instance already holds the lock, exiting" >> {WATCHDOG_LOG}
  exit 1
fi
cd {PROJECT_DIR}
CKPT={CKPT}
TARGET_GEN={TARGET_GEN}
while true; do
  if [ -f "$CKPT" ]; then
    GEN=$(python3 -c "import torch; print(torch.load('$CKPT', weights_only=False)['generation'])" 2>/dev/null)
    RESUME_ARG="--resume $CKPT"
  else
    GEN=0
    RESUME_ARG=""
  fi
  echo "WATCHDOG: $(date) current saved generation: $GEN" >> {WATCHDOG_LOG}
  if [ -n "$GEN" ] && [ "$GEN" -ge "$TARGET_GEN" ]; then
    echo "WATCHDOG: reached generation $GEN, stopping" >> {WATCHDOG_LOG}
    break
  fi
  python agents/train_pinball_circuit_cem.py --binary {binary} \\
    --connectome agents/checkpoints/big_v2/connectome.json --readout-dim 8 \\
    --optimizer cma --cma-sigma0 0.2 --workers $(( $(nproc) * 2 )) --population 16 --fresh-fraction 0 \\
    --ipop-patience 80 --ipop-growth 2.0 --ipop-max-population 48 \\
    --generations $TARGET_GEN --courses-per-candidate 8 --max-steps 3000 \\
    --validation-episodes 32 --promotion-t 2.0 \\
    $RESUME_ARG --out "$CKPT" --save-every 5 < /dev/null >> {WATCHDOG_LOG} 2>&1
  echo "WATCHDOG: $(date) training process exited with code $?, restarting in 5s" >> {WATCHDOG_LOG}
  sleep 5
done
"""
	with open(WATCHDOG_PATH, "w") as f:
		f.write(watchdog_script)
	print(f"wrote {WATCHDOG_PATH}", flush=True)

	print("\n=== 9. launching training (fully detached) ===", flush=True)
	with open(WATCHDOG_STDOUT, "w") as log_f:
		subprocess.Popen(
			["setsid", "bash", WATCHDOG_PATH],
			stdin=subprocess.DEVNULL,
			stdout=log_f,
			stderr=subprocess.STDOUT,
			start_new_session=True,
		)
	print("launched. Check progress with a small colab exec script doing:", flush=True)
	print(f"  grep -a gen {WATCHDOG_LOG} | tail -10", flush=True)
	print(f"  ps aux | grep -c SpaceCadetPinball", flush=True)


if __name__ == "__main__":
	main()
