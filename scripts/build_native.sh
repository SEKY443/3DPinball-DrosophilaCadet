#!/usr/bin/env bash
# Native macOS build (Phase 1/2 debug loop): applies the src_cpp patches onto the submodule,
# configures via Homebrew's SDL2/SDL2_mixer, and builds the SpaceCadetPinball binary. Requires
# `brew install sdl2 sdl2_mixer cmake` and Xcode command-line tools license accepted
# (`sudo xcodebuild -license accept`) beforehand.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$ROOT_DIR/vendor/SpaceCadetPinball"
BUILD_DIR="$ROOT_DIR/build-native"

"$ROOT_DIR/scripts/apply_patches.sh"

# Never track build output or the bin/ dir CMakeLists.txt places inside the submodule tree -
# this is a local-only exclude (.git/info/exclude), not a committed change to the submodule.
EXCLUDE_FILE="$VENDOR_DIR/.git/info/exclude"
if [ -f "$VENDOR_DIR/.git" ]; then
	# Submodule checkout: .git is a file pointing at the real gitdir.
	GIT_DIR="$(sed -n 's/^gitdir: //p' "$VENDOR_DIR/.git")"
	EXCLUDE_FILE="$VENDOR_DIR/$GIT_DIR/info/exclude"
fi
mkdir -p "$(dirname "$EXCLUDE_FILE")"
touch "$EXCLUDE_FILE"
grep -qxF 'bin/' "$EXCLUDE_FILE" || echo 'bin/' >> "$EXCLUDE_FILE"

HOMEBREW_PREFIX="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"

cmake -S "$VENDOR_DIR" -B "$BUILD_DIR" \
	-DCMAKE_BUILD_TYPE=Debug \
	-DCMAKE_OSX_ARCHITECTURES="$(uname -m)" \
	-DSDL2_PATH="$HOMEBREW_PREFIX" \
	-DSDL2_MIXER_PATH="$HOMEBREW_PREFIX"

cmake --build "$BUILD_DIR" --parallel

echo "Built: $VENDOR_DIR/bin/SpaceCadetPinball"
