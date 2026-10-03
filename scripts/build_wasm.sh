#!/usr/bin/env bash
# Emscripten build (Phase 6): builds the same src_cpp-patched engine as build_native.sh, but
# links state_export_wasm.cpp's exports instead of ipc_server.cpp's sockets for state/action I/O
# (see src_cpp/patches/0002 - the pb::timed_frame hook branches on __EMSCRIPTEN__).
#
# Verified working end-to-end in a real Chrome renderer (headless Chrome screenshot test,
# 2026-09-16). Two non-obvious flags below were required beyond the standard USE_SDL pattern:
#   -sASYNCIFY=1            upstream's winmain::MainLoop() is a plain blocking `while(true)`
#                            (frame pacing via std::this_thread::sleep_for, not SDL_Delay) -
#                            without Asyncify it never yields to the browser and the canvas
#                            stays black/frozen forever.
#   HEAPU8 in EXPORTED_RUNTIME_METHODS   web/bridge.js reads StateFrame bytes directly out of
#                            Wasm linear memory via Module.HEAPU8.buffer; ccall/cwrap alone
#                            don't expose the heap views to the outer Module object.
#   --preload-file DEMO.DAT  pb::SelectDatFile (pb.cpp) fopen()s CADET.DAT/PINBALL.DAT/DEMO.DAT
#                            from cwd first - under Emscripten that's MEMFS "/", which starts
#                            empty. Without this, pb::init() silently never finds game data,
#                            MainTable stays null, and the canvas stays black forever with no
#                            error (state_export_wasm.cpp's magic never gets written either, so
#                            web/bridge.js also sees only "no frame yet"). Requires a real
#                            DEMO.DAT already present at vendor/SpaceCadetPinball/bin/DEMO.DAT
#                            (project README) - skipped with a warning if absent.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$ROOT_DIR/vendor/SpaceCadetPinball"
BUILD_DIR="$ROOT_DIR/build-wasm"
OUT_DIR="$ROOT_DIR/web/dist"
DAT_FILE="$VENDOR_DIR/bin/DEMO.DAT"

command -v emcmake >/dev/null 2>&1 || {
	echo "error: emcmake not found on PATH - install Emscripten (brew install emscripten, or emsdk)" >&2
	exit 1
}

"$ROOT_DIR/scripts/apply_patches.sh"

PRELOAD_FLAG=""
if [ -f "$DAT_FILE" ]; then
	PRELOAD_FLAG="--preload-file $DAT_FILE@DEMO.DAT"
else
	echo "warning: $DAT_FILE not found - build will have no game data (black canvas at runtime)" >&2
fi

# Sound effects: the original SOUND*.WAV files are supplied locally in assets/original/sound/ (copyrighted, never
# committed). DEMO.DAT runs in Full Tilt mode, whose loader opens "sound/<name>.wav" in lower case first (then the
# whole path upper-cased), and the Emscripten file system is case-sensitive, so stage lower-case copies.
SOUND_SRC="$ROOT_DIR/assets/original/sound"
if [ -d "$SOUND_SRC" ] && [ -n "$PRELOAD_FLAG" ]; then
	SOUND_STAGE="$BUILD_DIR/sound_stage"
	rm -rf "$SOUND_STAGE"
	mkdir -p "$SOUND_STAGE"
	for f in "$SOUND_SRC"/*; do
		[ -f "$f" ] || continue
		base="$(basename "$f")"
		cp "$f" "$SOUND_STAGE/$(printf '%s' "$base" | tr '[:upper:]' '[:lower:]')"
	done
	PRELOAD_FLAG="$PRELOAD_FLAG --preload-file $SOUND_STAGE@sound"
else
	echo "note: no $SOUND_SRC - build will have no sound effects" >&2
fi

EMSCRIPTEN_LINK_FLAGS="-sUSE_SDL=2 -sUSE_SDL_MIXER=2 -sASYNCIFY=1 -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createPinballModule -sEXPORTED_RUNTIME_METHODS=[\"ccall\",\"cwrap\",\"HEAPU8\"] -sEXPORTED_FUNCTIONS=[\"_pb_get_state_ptr\",\"_pb_get_state_size\",\"_pb_set_action\",\"_pb_menu_command\",\"_pb_plunger\",\"_pb_diag\",\"_pb_menu_state\",\"_pb_set_player_name\",\"_pb_seed_high_score\",\"_pb_hs_commit_count\",\"_pb_game_end_count\",\"_pb_last_game_score\",\"_pb_current_score\",\"_pb_hs_name\",\"_pb_hs_score\",\"_main\"] -sALLOW_MEMORY_GROWTH=1 $PRELOAD_FLAG"

emcmake cmake -S "$VENDOR_DIR" -B "$BUILD_DIR" \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_EXECUTABLE_SUFFIX=".js" \
	-DCMAKE_CXX_FLAGS="-sUSE_SDL=2 -sUSE_SDL_MIXER=2" \
	-DCMAKE_EXE_LINKER_FLAGS="$EMSCRIPTEN_LINK_FLAGS"

emmake cmake --build "$BUILD_DIR" --parallel

mkdir -p "$OUT_DIR"
cp "$VENDOR_DIR/bin/SpaceCadetPinball.js" "$OUT_DIR/"
cp "$VENDOR_DIR/bin/SpaceCadetPinball.wasm" "$OUT_DIR/"
if [ -f "$VENDOR_DIR/bin/SpaceCadetPinball.data" ]; then
	cp "$VENDOR_DIR/bin/SpaceCadetPinball.data" "$OUT_DIR/"
fi

echo "Built: $OUT_DIR/SpaceCadetPinball.js (+ .wasm)"
