// Connects the Emscripten-built game (web/dist/SpaceCadetPinball.js, built by
// scripts/build_wasm.sh from the same src_cpp-patched engine as the native build) to a trained
// FixedCircuitAgent (agents/fixed_circuit_agent.py) running as a pure-JS forward pass
// (circuit_policy.js + observation.js). Reads a StateFrame out of Wasm linear memory each
// animation frame and, unless manual play is toggled on, feeds the circuit's decision back in
// via the pb_set_action() export.
//
// StateFrame layout below must stay in lockstep with src_cpp/ipc_protocol.h.

import { PinballCircuitPolicyJS } from "./circuit_policy.js";
import { stateToObs } from "./observation.js";
import createPinballModule from "./dist/SpaceCadetPinball.js";

const STATE_OFFSETS = {
	magic: 0, tick: 4,
	ball_x: 8, ball_y: 12, ball_vx: 16, ball_vy: 20,
	flipper_left: 24, flipper_right: 25, ball_in_play: 26, tilted: 27,
	score_delta: 28, done: 32, flipper_hit: 33, relaunch_pending: 34,
	ball2_x: 36, ball2_y: 40, ball2_vx: 44, ball2_vy: 48, ball2_active: 52,
};
const STATE_MAGIC = 0x424e4950; // "PINB", little-endian - matches ipc_protocol.h::kStateMagic

// Training's env_python/pinball_env.py::PinballEnv uses frame_skip=4: one circuit decision holds
// for 4 native engine ticks. Matching that cadence here (instead of deciding every animation
// frame) is what makes the browser policy behave like the one that was actually trained/scored.
const FRAME_SKIP = 4;

// env_python/pinball_env.py's PLUNGER_REST_X/Y/TOL - where a fresh ball parks after a drain.
// ball_in_play reads true there (see pinball_env.py's comment), so pb_set_action's launch is a
// guarded no-op; only a real plunger key pull moves it, same as a human player would do.
const PLUNGER_REST_X = -7.02;
const PLUNGER_REST_Y = 10.09;
const PLUNGER_REST_TOL = 0.3;
// env_python/pinball_env.py's DRAIN_FROM_Y: a drain is the ball jumping from below this y straight
// to the plunger rest spot.
const DRAIN_FROM_Y = 12.5;
const PLUNGER_REST_HOLD_MS = 500; // how long the ball must sit at rest before we pull the plunger
const PLUNGER_KEY_HOLD_MS = 400; // how long the synthetic keydown is held before keyup

// Only the best fly brain is served: the 444-neuron connectome with the popcode sensory encoding and
// delay1 flipper feedback, imitation-trained from the one-step-lead reflex
// (agents/experiments/timing/seeds/dagger_popcode_delay1_r8.pt). Older readouts stay on disk
// (circuit_readout_444.json, circuit_readout_108.json, circuit_readout_444_fullgame.json) but are
// no longer selectable.
const BRAINS = {
	popcode: {
		label: "444-neuron fly circuit (popcode, lead-trained)",
		connectome: "./connectome_444.json",
		readout: "./circuit_readout_444_popcode.json",
	},
};

function readState(module) {
	const ptr = module.ccall("pb_get_state_ptr", "number", [], []);
	const size = module.ccall("pb_get_state_size", "number", [], []);
	const view = new DataView(module.HEAPU8.buffer, ptr, size);

	const magic = view.getUint32(STATE_OFFSETS.magic, true);
	if (magic !== STATE_MAGIC) {
		return null; // engine hasn't produced a frame yet (still loading/initializing)
	}
	return {
		tick: view.getUint32(STATE_OFFSETS.tick, true),
		ball_x: view.getFloat32(STATE_OFFSETS.ball_x, true),
		ball_y: view.getFloat32(STATE_OFFSETS.ball_y, true),
		ball_vx: view.getFloat32(STATE_OFFSETS.ball_vx, true),
		ball_vy: view.getFloat32(STATE_OFFSETS.ball_vy, true),
		flipper_left: view.getUint8(STATE_OFFSETS.flipper_left) !== 0,
		flipper_right: view.getUint8(STATE_OFFSETS.flipper_right) !== 0,
		ball_in_play: view.getUint8(STATE_OFFSETS.ball_in_play) !== 0,
		tilted: view.getUint8(STATE_OFFSETS.tilted) !== 0,
		score_delta: view.getInt32(STATE_OFFSETS.score_delta, true),
		done: view.getUint8(STATE_OFFSETS.done) !== 0,
		flipper_hit: view.getUint8(STATE_OFFSETS.flipper_hit) !== 0,
		relaunch_pending: view.getUint8(STATE_OFFSETS.relaunch_pending) !== 0,
		ball2_x: view.getFloat32(STATE_OFFSETS.ball2_x, true),
		ball2_y: view.getFloat32(STATE_OFFSETS.ball2_y, true),
		ball2_vx: view.getFloat32(STATE_OFFSETS.ball2_vx, true),
		ball2_vy: view.getFloat32(STATE_OFFSETS.ball2_vy, true),
		ball2_active: view.getUint8(STATE_OFFSETS.ball2_active) !== 0,
	};
}

// Synthesizes a real keyboard event so Emscripten's SDL2 port (which listens on actual DOM
// key events, not a JS-callable input API) sees a plunger key press - see options.cpp's default
// binding (GameBindings::Plunger -> SDLK_SPACE).
function synthesizeKeyPress(key, code, keyCode) {
	const down = new KeyboardEvent("keydown", { key, code, keyCode, which: keyCode, bubbles: true });
	const up = new KeyboardEvent("keyup", { key, code, keyCode, which: keyCode, bubbles: true });
	window.dispatchEvent(down);
	document.dispatchEvent(down);
	setTimeout(() => {
		window.dispatchEvent(up);
		document.dispatchEvent(up);
	}, PLUNGER_KEY_HOLD_MS);
}

function pullPlunger() {
	synthesizeKeyPress(" ", "Space", 32);
}

function setText(el, text) {
	el.textContent = text; // never innerHTML - no dynamic HTML injection
}

async function main() {
	const statusEl = document.getElementById("status");
	const scoreEl = document.getElementById("score");
	const ballEl = document.getElementById("ball-info");
	const flippersEl = document.getElementById("flippers");
	const manualCheckbox = document.getElementById("manual-play");

	const brainParam = new URLSearchParams(window.location.search).get("brain");
	const brain = BRAINS[brainParam] || BRAINS.popcode;

	setText(statusEl, `Loading ${brain.label}...`);
	const policy = await PinballCircuitPolicyJS.load(brain.connectome, brain.readout);
	// Read-only hook for the "Brain Activity" window (brain_activity.js): the live circuit state
	// (policy.activity, mutated in place each decision) and the latest decision's logits/actions.
	window.flyBrain = { policy, last: null, decisions: 0 };

	setText(statusEl, "Starting engine...");
	const module = await createPinballModule({
		canvas: document.getElementById("canvas"),
		// The emcc glue in ./dist/ fetches its .wasm/.data siblings relative to the *page* URL
		// by default, not relative to its own location - since index.html lives one directory
		// above dist/, redirect those lookups back into dist/.
		locateFile: (path) => `./dist/${path}`,
	});
	setText(statusEl, "Running");

	let manualLeft = false;
	let manualRight = false;
	window.addEventListener("keydown", (e) => {
		if (e.key.toLowerCase() === "z") manualLeft = true;
		if (e.key === "/") manualRight = true;
	});
	window.addEventListener("keyup", (e) => {
		if (e.key.toLowerCase() === "z") manualLeft = false;
		if (e.key === "/") manualRight = false;
	});

	let cumulativeScore = 0;
	let heldAction = { flipper_left: false, flipper_right: false };
	let lastDecisionTick = null;
	let wasBallInPlay = false;
	let lastBallY = null;
	let ballNumber = 0;
	let ballLifeStartTick = 0;
	let plungerRestSince = null;
	let plungerPulled = false;
	let gameOver = false;

	function onNewBallLaunched() {
		policy.reset(); // one training episode = one ball life; circuit state resets with it
		ballNumber += 1;
		lastDecisionTick = null;
		heldAction = { flipper_left: false, flipper_right: false };
	}

	function frame() {
		if (gameOver) return;
		const state = readState(module);
		if (state) {
			cumulativeScore += state.score_delta;

			if (state.done) {
				gameOver = true;
				setText(statusEl, "Game over");
				return;
			}

			// A training episode is one ball life and ends at a drain (pinball_env.py's drain
			// detector). After a drain the next ball parks on the plunger with ball_in_play still
			// true, so the in-play edge below never fires for it - detect the drain the same way
			// training does, so the circuit (and the delay1 flipper state) reset per life.
			const drained = lastBallY !== null && lastBallY > DRAIN_FROM_Y
				&& Math.abs(state.ball_x - PLUNGER_REST_X) < PLUNGER_REST_TOL
				&& Math.abs(state.ball_y - PLUNGER_REST_Y) < PLUNGER_REST_TOL;
			lastBallY = state.ball_y;
			if ((state.ball_in_play && !wasBallInPlay) || drained) {
				ballLifeStartTick = state.tick;
				onNewBallLaunched();
			}
			wasBallInPlay = state.ball_in_play;

			if (manualCheckbox.checked) {
				module.ccall("pb_set_action", null, ["number", "number", "number"],
					[manualLeft ? 1 : 0, manualRight ? 1 : 0, 0]);
			} else {
				// Decision cadence matches training's frame_skip: only step the circuit once
				// every FRAME_SKIP engine ticks, holding the last action in between. One
				// animation frame can span multiple (or zero, under Asyncify frame pacing)
				// engine ticks, so this is checked against the engine's own tick counter, not
				// once per requestAnimationFrame call.
				if (lastDecisionTick === null || state.tick - lastDecisionTick >= FRAME_SKIP) {
					const obs = stateToObs(state);
					const result = policy.forward(obs);
					const { actions } = result;
					window.flyBrain.last = result;
					window.flyBrain.decisions++;
					heldAction = actions;
					lastDecisionTick = state.tick;
				}
				module.ccall("pb_set_action", null, ["number", "number", "number"], [
					heldAction.flipper_left ? 1 : 0,
					heldAction.flipper_right ? 1 : 0,
					state.ball_in_play ? 0 : 1, // auto-launch pulse; engine no-ops it once in play
				]);
			}

			// After a drain the engine parks the next ball on the plunger with ball_in_play
			// still true - pb_set_action's launch is ignored there, so pull the plunger key
			// like a human would once the ball has settled at rest for a bit.
			const atRest = state.ball_in_play
				&& Math.abs(state.ball_x - PLUNGER_REST_X) < PLUNGER_REST_TOL
				&& Math.abs(state.ball_y - PLUNGER_REST_Y) < PLUNGER_REST_TOL;
			if (atRest) {
				if (plungerRestSince === null) plungerRestSince = performance.now();
				else if (!plungerPulled && performance.now() - plungerRestSince > PLUNGER_REST_HOLD_MS) {
					pullPlunger();
					plungerPulled = true;
				}
			} else {
				plungerRestSince = null;
				plungerPulled = false;
			}

			setText(scoreEl, `score: ${cumulativeScore}  |  tick: ${state.tick}`);
			setText(ballEl, `brain: ${brain.label}  |  ball #${ballNumber}  |  ball life: ${state.tick - ballLifeStartTick} ticks`);
			setText(flippersEl, `flippers: ${heldAction.flipper_left ? "L" : "-"} ${heldAction.flipper_right ? "R" : "-"}`);
		}
		requestAnimationFrame(frame);
	}
	requestAnimationFrame(frame);
}

main().catch((err) => {
	console.error(err);
	document.getElementById("status").textContent = `Error: ${err.message}`;
});

// Surface errors on-page too, not just in devtools console - useful when checking from a
// device without easy devtools access (e.g. a phone).
window.addEventListener("error", (e) => {
	const el = document.getElementById("status");
	if (el) el.textContent = `window.onerror: ${e.message} (${e.filename}:${e.lineno})`;
});
window.addEventListener("unhandledrejection", (e) => {
	const el = document.getElementById("status");
	if (el) el.textContent = `unhandledrejection: ${e.reason}`;
});
