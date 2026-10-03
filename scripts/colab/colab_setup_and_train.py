#!/usr/bin/env python3
"""Run this on the Colab VM: sync the RNG-reproducibility fix, rebuild the native binary, clear
any stray processes, write a crash-resilient watchdog script, and launch the next training round
(fresh start - the old pinball_circuit_cem_595n.pt's fitness numbers are not trustworthy, see
agents/train_pinball_circuit_cem.py's history: identical champion weights logged two different
fitness values because the engine's rand()/RandFloat() was never seeded per-episode).

Usage in Colab: upload this file into /content/3DPinball-DrosophilaCadet/, then run:
    !python3 colab_setup_and_train.py
"""
import os
import subprocess
import sys

PROJECT_DIR = "/content/3DPinball-DrosophilaCadet"
DRIVE_ZIP = "/content/drive/MyDrive/pinball_checkpoints/rng_fix.zip"
CONTENT_ZIP = "/content/rng_fix.zip"
CKPT = "/content/drive/MyDrive/pinball_checkpoints/pinball_circuit_cem_595n_v2.pt"
WATCHDOG_PATH = "/content/watchdog_v2.sh"
WATCHDOG_STDOUT = "/content/watchdog_v2_stdout.log"
TARGET_GEN = 400


def run(cmd, **kwargs):
    print(f"$ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, **kwargs)
    if result.returncode != 0:
        print(f"  -> exited with code {result.returncode}", flush=True)
    return result


def main():
    print("=== 1. syncing RNG-reproducibility fix from Drive ===", flush=True)
    run(f"cp {DRIVE_ZIP} {CONTENT_ZIP}")
    run(f"cd {PROJECT_DIR} && unzip -o {CONTENT_ZIP}")

    print("\n=== 2. rebuilding native binary (incremental) ===", flush=True)
    build = run(f"cd {PROJECT_DIR} && cmake --build build-native --parallel")
    if build.returncode != 0:
        print("build failed - stopping here, do not launch training on a stale binary", flush=True)
        sys.exit(1)

    print("\n=== 3. clearing any stray processes ===", flush=True)
    run("pkill -9 -f watchdog.sh; pkill -9 -f train_pinball_circuit_cem; pkill -9 -f SpaceCadetPinball; true")

    print("\n=== 4. writing watchdog script ===", flush=True)
    watchdog_script = f"""#!/bin/bash
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
  echo "WATCHDOG_V2: $(date) current saved generation: $GEN" >> /content/train_v2.log
  if [ -n "$GEN" ] && [ "$GEN" -ge "$TARGET_GEN" ]; then
    echo "WATCHDOG_V2: reached generation $GEN, stopping" >> /content/train_v2.log
    break
  fi
  python agents/train_pinball_circuit_cem.py --binary vendor/SpaceCadetPinball/bin/SpaceCadetPinball \\
    --connectome web/connectome.json --readout-dim 32 --workers $(nproc) --population 16 --elites 4 \\
    --generations $TARGET_GEN --courses-per-candidate 2 --validation-episodes 4 --frame-skip 4 \\
    --max-steps 800 $RESUME_ARG --out "$CKPT" --save-every 5 < /dev/null >> /content/train_v2.log 2>&1
  echo "WATCHDOG_V2: $(date) training process exited with code $?, restarting in 5s" >> /content/train_v2.log
  sleep 5
done
"""
    with open(WATCHDOG_PATH, "w") as f:
        f.write(watchdog_script)
    print(f"wrote {WATCHDOG_PATH}", flush=True)

    print("\n=== 5. launching training (fully detached) ===", flush=True)
    with open(WATCHDOG_STDOUT, "w") as log_f:
        subprocess.Popen(
            ["setsid", "bash", WATCHDOG_PATH],
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print("launched. Check progress with:", flush=True)
    print(f"  !grep -a champion /content/train_v2.log | tail -5", flush=True)
    print(f"  !ps aux | grep -c SpaceCadetPinball", flush=True)


if __name__ == "__main__":
    main()
