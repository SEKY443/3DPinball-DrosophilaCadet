# Using `google-colab-cli` for this project — gotchas and runbook

Google's official Colab CLI (`pip install google-colab-cli`, repo:
[googlecolab/google-colab-cli](https://github.com/googlecolab/google-colab-cli)) lets you drive a
Colab runtime from a local terminal instead of Chrome/notebook-cell automation. Real first-party
tool, not a ban-risk workaround like ngrok/cloudflared SSH tunnels into a plain notebook session.
Several rough edges as of 2026-09-17 (PyPI version 0.6.0):

## 1. Broken dependency out of the box

`pipx install google-colab-cli` pulls `jupyter_kernel_client` from PyPI, but the CLI actually
needs Google's own fork (different API - `KernelClient` class vs. PyPI's `JupyterKernelClient`).
Every `colab` command that touches a kernel fails with:

```
AttributeError: module 'jupyter_kernel_client' has no attribute 'KernelClient'
```

Fix (run once after every fresh `pipx install`/`pipx reinstall`/`pipx install --force`):

```bash
pipx runpip google-colab-cli install --force-reinstall \
  "jupyter-kernel-client @ git+https://github.com/googlecolab/jupyter-kernel-client.git"
```

Verify: `~/.local/pipx/venvs/google-colab-cli/bin/python3 -c "import jupyter_kernel_client as k; print('KernelClient' in dir(k))"` should print `True`.

## 2. `--high-mem` isn't in the released PyPI version yet

`colab new --help` on 0.6.0 has no `--high-mem` flag, even though the GitHub README documents
it. Without it you get a 2-vCPU/12GB default shape - not enough for 6-8 parallel CEM workers.
Install the dev build straight from GitHub main instead:

```bash
pipx install --force "git+https://github.com/googlecolab/google-colab-cli.git" \
  --python /opt/homebrew/bin/python3.13
# then redo the jupyter_kernel_client fix from step 1 - a fresh install re-breaks it
```

This got us `colab new -s pinball --high-mem` → 8 vCPU / 50GB, matching what the browser-based
Colab Pro session had. Attaching `--gpu T4` alone, without `--high-mem`, did NOT increase vCPU
count (still 2) - GPU tier and CPU/RAM tier are independent knobs.

## 3. `colab drivemount` is unreliable for interactive OAuth

`colab drivemount -s NAME` runs `drive.mount()` remotely, which needs a human to open a URL and
authorize. In practice this repeatedly hit `ValueError: mount failed` even after visiting the
link and approving - almost certainly the default 120s timeout inside `drive.mount()` being too
short for a real back-and-forth (show link → user switches to browser → clicks through → comes
back → confirms). `colab exec -f script.py` calling `drive.mount(path, timeout_ms=600000)`
directly didn't help either - `colab exec`'s own output appears to be fully buffered until the
whole cell finishes, so you can't see the auth prompt to act on it at all.

**Workaround that actually worked: skip Drive entirely for a CLI-managed session.** Use
`colab upload -s NAME LOCAL REMOTE` to push files straight into `/content` (no OAuth needed), and
`colab download` to pull results back out. Only reach for `drivemount` if you specifically need
the *training script itself* to read/write Drive at runtime (we didn't - checkpoints just live in
plain `/content/` and get `colab download`ed by hand).

## 4. Session containers can "resurrect" killed processes

The most expensive lesson: after `pkill -9 -f watchdog.sh` *and* a `ps aux` check confirming zero
survivors, watchdog + training processes from earlier in the session reappeared minutes later,
each still running its **original, stale** launch flags (e.g. an old `--generations 400` run
resurrected alongside a newer `--generations 800` one). Ended up with up to 4 concurrent training
processes all writing the same checkpoint file, corrupting the log (interleaved/non-monotonic
generation numbers) and the CEM state itself. Root cause not confirmed with certainty, but
consistent with the underlying container being suspended/resumed to save compute between CLI
calls, reviving processes whose kill signal didn't fully "stick" before a suspend.

**Fix: make the watchdog self-protecting with a file lock**, so accidental duplicate launches are
harmless no-ops instead of silently running in parallel:

```bash
#!/bin/bash
exec 200>/tmp/watchdog.lock
if ! flock -n 200; then
  echo "another instance already holds the lock, exiting" >> /content/train.log
  exit 1
fi
# ... rest of the while-true retrain loop ...
```

Before *every* relaunch, also do a real kill-wait-verify loop (kill, sleep ~5s, `ps aux` check,
repeat 3-4x) rather than a single kill+check - a single check can report "clean" and then be
wrong seconds later.

## 5a. Root cause of "sessions disappear" found and patched: proxy token TTL, never refreshed

The recurring "session lost (404/401)" below turned out to be a real upstream bug, not an
unavoidable platform quirk: [googlecolab/google-colab-cli#106](https://github.com/googlecolab/google-colab-cli/issues/106).

The runtime proxy token (`RuntimeProxyInfo.token`) has a TTL reported in
`tokenExpiresInSeconds`, but the CLI never reads that field and never refreshes the token. Once
it expires, the next `exec`/`run` gets a genuine 401/404 from the *proxy* (not from the
assignment itself), `is_terminal_error()` classifies that as "session gone", and
`prune_session()` deletes the local binding unconditionally - while the VM is still alive and
still listed by `list_assignments()`, which always hands back a fresh token. Every "session
lost" we hit and "recovered" from by recreating the VM from scratch was very likely this: the
old VM was still running the whole time, we just orphaned it and paid for a full rebuild instead
of a token swap. Confirmed independently by two other users: issue #115 ("sessions closing
prematurely" on Colab Pro+, ~1.5hr) and a session-loss report at ~60min - the exact interval
varies but the mechanism is the same TTL-expiry misclassification.

**Fix applied locally** (patches the installed pipx venv directly - `pip install`/`pipx upgrade`
will overwrite it, so redo this after any CLI reinstall):

`~/.local/pipx/venvs/google-colab-cli/lib/python3.13/site-packages/colab_cli/common.py` -
`State.prune_session()` now confirms against the server first via a new `_assignment_gone()`
helper (mirrors the fix suggested in the issue): it calls `client.list_assignments()`, and if the
session's `endpoint` still appears in the list, it swaps in the fresh `token`/`url` from that
assignment and returns without deleting anything. Only a genuinely absent endpoint (or update:
this check happens to trigger for a real error, not a listing failure - failing to even query
`list_assignments()` also counts as "keep the binding", since deleting on an inconclusive check
is the strictly worse failure mode) results in an actual prune. All four `prune_session()` call
sites (3 in `execution.py`, 1 in `run.py`) get this protection from one place.

Verified working: after applying the patch, a `colab exec` that printed the familiar `[colab]
Session 'pinball' appears to be lost (404/401). Cleaning up.` (that message is echoed
unconditionally by the call site *before* `prune_session` runs its new check) was immediately
followed by `colab status -s pinball` still showing the session bound, and a retried `colab exec`
succeeding against the *same* long-running process (verified via its unchanged process start
time) - no VM recreation needed.

**Side effect discovered while investigating**: `colab sessions` also revealed 2 orphaned "[?]"
(unnamed) VM assignments still running on the server - leftovers from earlier "lost" sessions
whose local binding was deleted by the bug above, so `colab stop -s NAME` could never target them
(no local name to resolve). These were burning Colab Pro compute quota for no reason. Cleaned up
with a one-off script calling the CLI's own `state.client.unassign(endpoint)` directly (bypasses
the name-based `stop` command, which requires a local binding):

```python
from colab_cli.common import state
sessions, assignments = state.sync_sessions()
name_by_endpoint = {s.endpoint: s.name for s in sessions.values()}
for a in assignments:
    if name_by_endpoint.get(a.endpoint, "?") == "?":
        state.client.unassign(a.endpoint)
```

Worth an occasional `colab sessions` sanity check on any long overnight run predating this patch,
to catch and clean up similar leftover orphans.

## 5. Sessions disappear entirely, roughly every 30-60 minutes

Separately from issue #4 (duplicate resurrection), a CLI-managed session can just vanish outright -
any `colab exec`/`colab status` call starts failing with `Session 'NAME' appears to be lost
(404/401). Cleaning up.`, and the local session registry gets pruned. Observed 3 times in about
1.5 hours of a single overnight run, each time with no warning beforehand (previous check a
Q~30min earlier looked completely healthy). Root cause not confirmed - possibly the keep-alive
daemon (see the GitHub repo's session-management design doc) dying silently, since normal browser
Colab tabs don't disappear anywhere near this often. Whatever the cause, **build recovery into the
workflow rather than trying to prevent it**:

- Periodically `colab download` the training checkpoint to local disk (every monitoring check-in,
  not just occasionally) - this is the only real mitigation, since a lost session means the entire
  `/content` filesystem is gone, including anything not saved to Drive/downloaded.
- Keep a rolling window of a few local checkpoint backups (renamed with the generation number) in
  case the very latest download happens to be from mid-write or otherwise bad.
- Recovery procedure when a session is lost: `colab new -s NAME --high-mem` → re-upload the base
  project zip + all patch zips + connectome + **the latest local checkpoint backup** (upload it to
  the same `/content/<ckpt>.pt` path the training script's `--resume`/`--out` expects) → re-run the
  full setup script → relaunch the watchdog. The watchdog's own `if [ -f "$CKPT" ]` resume-detection
  logic (see the watchdog template below) then picks it up automatically with no code changes -
  confirmed this recovers cleanly with only a few generations of progress lost each time (whatever
  wasn't saved since the last `--save-every` checkpoint before the loss).

## 6. Bumping TARGET_GEN proactively during a recovery near the boundary

If a session loss happens to occur when the saved checkpoint's generation is already close to
the watchdog's `TARGET_GEN` (e.g. gen 721 with `TARGET_GEN=800`), don't just recover at the old
target - the watchdog would stop again within minutes, requiring a second relaunch almost
immediately. Instead, bump `TARGET_GEN` to the next milestone (e.g. 1200) directly in the
`colab_launch_locked.py` driver script's embedded watchdog heredoc *before* relaunching as part
of the same recovery. This value persists in the local driver script for all future recoveries
too, so it only needs to be bumped once per milestone, not re-applied every time a session is
lost.

## Practical workflow used for this project

1. `colab new -s pinball --high-mem` (plain CPU, no `--gpu`, unless you actually need one - it
   doesn't buy extra vCPU on its own).
2. `colab upload -s pinball <local zip> /content/<name>.zip` for the base project zip, any patch
   zips, and the current `web/connectome.json` (patch zips: `zip -q out.zip path/to/file` from the
   repo root, preserving relative paths, so `unzip -o` on the other end drops files in place).
3. `colab exec -s pinball -f script.py --timeout N` to run a plain (non-interactive) Python driver
   script that does `subprocess.run(...)` for each shell step - far more reliable than trying to
   paste multi-line shell into a `!`-prefixed one-liner.
4. Write the watchdog with the `flock` guard above, launch via
   `subprocess.Popen(["setsid", "bash", path], stdin=DEVNULL, stdout=logfile, stderr=STDOUT, start_new_session=True)`
   from inside that same driver script.
5. Poll progress with a small `colab exec -f` script that just does `grep`/`ps aux` - keep it
   separate from the launch script so re-running it can't accidentally relaunch training.
