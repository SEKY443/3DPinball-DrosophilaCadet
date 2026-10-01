#!/usr/bin/env bash
# Applies src_cpp/patches/*.patch onto vendor/SpaceCadetPinball. Idempotent: a patch already
# applied (checked via `git apply --check --reverse`) is skipped rather than re-applied or errored on.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$ROOT_DIR/vendor/SpaceCadetPinball"
PATCH_DIR="$ROOT_DIR/src_cpp/patches"

if [ ! -d "$VENDOR_DIR/.git" ] && [ ! -f "$VENDOR_DIR/.git" ]; then
	echo "error: $VENDOR_DIR is not a git checkout (submodule not initialized?)" >&2
	echo "hint: run 'git submodule update --init --recursive' first" >&2
	exit 1
fi

cd "$VENDOR_DIR"

for patch in "$PATCH_DIR"/*.patch; do
	name="$(basename "$patch")"
	if git apply --check --reverse "$patch" >/dev/null 2>&1; then
		echo "skip:  $name (already applied)"
		continue
	fi
	if ! git apply --check "$patch" >/dev/null 2>&1; then
		echo "error: $name does not apply cleanly onto the current submodule checkout" >&2
		exit 1
	fi
	git apply "$patch"
	echo "apply: $name"
done
