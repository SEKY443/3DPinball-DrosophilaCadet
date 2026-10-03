// Connects the Emscripten-built game (web/dist/SpaceCadetPinball.js, built by
// scripts/build_wasm.sh from the same src_cpp-patched engine as the native build) to a trained
// FixedCircuitAgent (agents/fixed_circuit_agent.py) running as a pure-JS forward pass
// (circuit_policy.js + observation.js). Reads a StateFrame out of Wasm linear memory each
// animation frame and, unless manual play is toggled on, feeds the circuit's decision back in
// via the pb_set_action() export.
//
// StateFrame layout below must stay in lockstep with src_cpp/ipc_protocol.h.

import { PinballCircuitPolicyJS } from "./circuit_policy.js";
import { RetinaGiantFiberPolicyJS } from "./gf_body.js?v=3";
import { stateToObs } from "./observation.js";
import { createMenuBar } from "./xp_menu.js?v=6";
import { randomFlyName } from "./fly_names.js";
import { createScoreHistory, HOUSE_RECORD } from "./score_history.js?v=7";
import { setText } from "./dom.js?v=1";
import createPinballModule from "./dist/SpaceCadetPinball.js?v=7"; // bump with every WASM rebuild

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

// Plunger lane geometry (table units). A parked ball (fresh feed or after a drain) sits in the lane at
// x ~ -7.02; in the browser build it first appears at y ~ 10.3 and settles at y ~ 11.67 once the plunger
// spring is at rest. (env_python/pinball_env.py's PLUNGER_REST_Y = 10.09 only matches the fresh-feed
// spot, so matching a single point +-0.3 never saw the settled ball - that is why the old key-press
// auto launch never fired.) The lane window below covers both.
const PLUNGER_LANE_X = -7.02;
const PLUNGER_LANE_X_TOL = 0.4;
const PLUNGER_LANE_Y_MIN = 9.5;
const PLUNGER_LANE_Y_MAX = 12.4;
const PLUNGER_MAX_REST_SPEED = 1.0; // table units / s - "parked" means (nearly) stationary
// env_python/pinball_env.py's DRAIN_FROM_Y: a drain is the ball jumping from below this y straight
// back into the plunger lane.
const DRAIN_FROM_Y = 12.5;
const MIN_BALL_GAP_TICKS = 60; // engine ticks (120/s) the ball must be absent before the next one counts as new
const AUTO_LAUNCH_HOLD_MS = 500; // ball must sit parked this long before the plunger is pulled
// Plunger pull (percent of the maximum spring pull-back, read from the engine via pb_diag(2); the pull grows with
// game time while the plunger is held, so a percentage is more repeatable than a wall-clock hold).
// PLUNGER_SKILL_PULL_PCT: the first plunge of every ball is a deliberate PARTIAL pull (Space Cadet skill shot). The
// ball then climbs the deployment chute only part of the way, lights exactly three skill-shot gates and falls back
// into the escape chute, which scores the Skill Shot (75000 for three lit gates, the table's best value) and -
// because it drains soon after passing the first gate - usually gets the engine's "Re-Deploy" ball save (a free
// ball). 38 % (the pull stops at ~39 %) was the best value in a headless-Chrome sweep: 21 of 24 plunges lit exactly
// three gates and fell back; the other three did not climb at all. More pull lights 4-6 gates (30000 and less), less
// pull often leaves the ball in the lane. A ball that is parked again afterwards (re-deployed, or never left the
// plunger) is plunged at PLUNGER_FULL_PULL_PCT instead, so the ball saver cannot chain skill shots forever and the
// game still ends.
const PLUNGER_SKILL_PULL_PCT = 38;
const PLUNGER_FULL_PULL_PCT = 100;
const PLUNGER_MAX_HOLD_MS = 4000; // safety: release after this long even if the pull percentage never arrives
const PLUNGER_RETRY_MS = 1500; // ball still parked this long after a partial pull -> pull again, fully
const AUTO_LAUNCH_FALLBACK_MS = 4000; // ball still parked this long after a full pull -> engine "Launch Ball" command
const SPLASH_MIN_MS = 2000; // the splash screen stays at least this long
const SPLASH_FADE_MS = 500; // must match the #splash opacity transition in index.html
const AUTO_NEW_GAME_DELAY_MS = 3000; // pause on "Game over" before the next game starts by itself

function inPlungerLane(x, y) {
	return Math.abs(x - PLUNGER_LANE_X) < PLUNGER_LANE_X_TOL && y > PLUNGER_LANE_Y_MIN && y < PLUNGER_LANE_Y_MAX;
}

// Two controllers, switchable at runtime (menu bar of the game window; ?brain=popcode|gf_retina picks the
// initial one, default gf_retina):
//  - gf_retina: the UNTRAINED retinotopic giant-fiber body (gf_body.js). LC4/LPLC2 looming cells of the fixed
//    MaleCNS connectome each watch their own spot along the flipper and are driven by the ball's angular
//    velocity/size seen from that spot; the two giant fibers (DNp01) press the flippers. No learned weights.
//  - popcode: the 444-neuron connectome with the popcode sensory encoding and delay1 flipper feedback,
//    imitation-trained from the one-step-lead reflex (agents/experiments/timing/seeds/dagger_popcode_delay1_r8.pt).
// Older readouts stay on disk but are no longer selectable.
const BRAINS = {
	gf_retina: {
		label: "Giant-fiber body (untrained)",
		kind: "gf",
		body: "./gf_body_retina.json?v=2",
	},
	popcode: {
		label: "444-neuron fly circuit (popcode, lead-trained)",
		kind: "trained",
		connectome: "./connectome_444.json",
		readout: "./circuit_readout_444_popcode.json",
	},
};
const DEFAULT_BRAIN = "gf_retina";

function loadPolicy(brain) {
	return brain.kind === "gf"
		? RetinaGiantFiberPolicyJS.load(brain.body)
		: PinballCircuitPolicyJS.load(brain.connectome, brain.readout);
}

// Reads the StateFrame out of Wasm linear memory. The state pointer/size are looked up once; the DataView is only
// recreated when the Wasm heap buffer was replaced (memory growth detaches the old one).
function createStateReader(module) {
	const ptr = module.cwrap("pb_get_state_ptr", "number", [])();
	const size = module.cwrap("pb_get_state_size", "number", [])();
	let heap = null;
	let view = null;
	return function readState() {
		if (module.HEAPU8.buffer !== heap) {
			heap = module.HEAPU8.buffer;
			view = new DataView(heap, ptr, size);
		}
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
	};
}

// Splash screen (index.html #splash): shown from page load until the engine produces its first frame, but
// at least SPLASH_MIN_MS; then it fades out and is removed from layout.
const splashEl = document.getElementById("splash");
const splashStatusEl = document.getElementById("splash-status");
const splashShownAt = performance.now();
let splashState = splashEl ? "shown" : "gone";
// Discreet load/error text on the splash (the old status strip of the monitor window is gone). Once the splash
// is gone the text is dropped; errors still reach the devtools console.
function setLoadStatus(text) {
	if (splashStatusEl && splashState === "shown") setText(splashStatusEl, text);
}
function hideSplash(force = false) {
	if (splashState !== "shown") return;
	if (!force && performance.now() - splashShownAt < SPLASH_MIN_MS) return;
	splashState = "fading";
	splashEl.classList.add("fade");
	setTimeout(() => { splashEl.classList.add("gone"); splashState = "gone"; }, SPLASH_FADE_MS + 50);
}

async function main() {
	// Hidden input toggled from the menu bar (Options > Manual Play); bridge and menu share this one state.
	const manualCheckbox = document.getElementById("manual-play");

	const brainParam = new URLSearchParams(window.location.search).get("brain");
	let brainKey = Object.hasOwn(BRAINS, brainParam ?? "") ? brainParam : DEFAULT_BRAIN;
	let brain = BRAINS[brainKey];

	// The engine's SDL window title ("3D Pinball for Windows - Space Cadet") is written to document.title by the
	// Emscripten SDL port; install a fixed title (writes ignored) before the engine starts.
	const PAGE_TITLE = document.title;
	Object.defineProperty(document, "title", { get: () => PAGE_TITLE, set: () => {}, configurable: true });

	// Master volume cap: the engine's audio (Emscripten SDL -> Web Audio) is routed through one GainNode at 30%.
	// Any connection made straight to an AudioContext's destination is redirected into that context's gain node.
	const MASTER_VOLUME = 0.3;
	const masterGains = new WeakMap();
	const origConnect = AudioNode.prototype.connect;
	AudioNode.prototype.connect = function (target, ...rest) {
		const ctx = this.context;
		if (ctx && target === ctx.destination) {
			let gain = masterGains.get(ctx);
			if (!gain) {
				gain = ctx.createGain();
				gain.gain.value = MASTER_VOLUME;
				origConnect.call(gain, ctx.destination);
				masterGains.set(ctx, gain);
			}
			if (this !== gain) return origConnect.call(this, gain, ...rest);
		}
		return origConnect.call(this, target, ...rest);
	};

	// Started now so the engine downloads/compiles while the policies load; awaited further down.
	const modulePromise = createPinballModule({
		canvas: document.getElementById("canvas"),
		// The emcc glue in ./dist/ fetches its .wasm/.data siblings relative to the *page* URL
		// by default, not relative to its own location - since index.html lives one directory
		// above dist/, redirect those lookups back into dist/.
		locateFile: (path) => `./dist/${path}?v=7`, // bump with every WASM rebuild (cache-buster)
	});
	modulePromise.catch(() => {}); // an engine failure surfaces at the await below; avoid an early unhandled rejection

	setLoadStatus(`Loading ${brain.label}...`);
	const policies = {};
	await Promise.all(Object.entries(BRAINS).map(async ([key, b]) => { policies[key] = await loadPolicy(b); }));
	let policy = policies[brainKey];
	// Read-only hook for the "Fly Brain Monitor" window (brain_dashboard.js): the live circuit state
	// (policy.activity, mutated in place each decision) and the latest decision's logits/actions.
	// `kind` is "gf" or "trained"; `last.gf` holds the giant-fiber internals when kind === "gf".
	window.flyBrain = { policy, kind: brain.kind, key: brainKey, last: null, decisions: 0 };

	let module = null; // set once the engine is up; the menu is inert until then
	let pb = null; // cwrap'd engine exports used every frame, created together with `module`
	let lastDecisionTick = null;
	let heldAction = { flipper_left: false, flipper_right: false };

	// Fresh controller state: circuit state, decision cadence and the held flipper action.
	function resetController() {
		policy.reset();
		lastDecisionTick = null;
		heldAction = { flipper_left: false, flipper_right: false };
	}

	function selectBrain(key) {
		if (!Object.hasOwn(BRAINS, key)) return;
		brainKey = key;
		brain = BRAINS[brainKey];
		policy = policies[brainKey];
		resetController(); // fresh circuit state for the new controller
		Object.assign(window.flyBrain, { policy, kind: brain.kind, key: brainKey, last: null, decisions: 0 });
	}

	// Set by New Game / Select Players (menu or F2); the frame loop then zeroes the HUD score and ball count.
	let newGameRequested = false;
	window.addEventListener("keydown", (e) => { if (e.key === "F2") newGameRequested = true; });
	const menuBar = document.getElementById("game-menubar");
	if (menuBar) {
		createMenuBar(menuBar, {
			command: (cmd, arg) => { if (cmd === 1 || cmd === 6) newGameRequested = true; if (pb) pb.menuCommand(cmd, arg ?? 0); },
			flags: () => (pb ? pb.menuState() : 0),
			brains: Object.entries(BRAINS).map(([key, b]) => ({ key, label: b.label })),
			brainKey: () => brainKey,
			brainLabel: () => brain.label,
			manual: () => manualCheckbox.checked,
			toggleManual: () => { manualCheckbox.checked = !manualCheckbox.checked; },
			selectBrain,
			dialogHost: document.getElementById("win-game"),
		});
	}

	setLoadStatus("Starting engine...");
	module = await modulePromise;
	pb = {
		menuCommand: module.cwrap("pb_menu_command", null, ["number", "number"]),
		menuState: module.cwrap("pb_menu_state", "number", []),
		setAction: module.cwrap("pb_set_action", null, ["number", "number", "number"]),
		plunger: module.cwrap("pb_plunger", null, ["number"]),
		diag: module.cwrap("pb_diag", "number", ["number"]),
		gameEndCount: module.cwrap("pb_game_end_count", "number", []),
		lastGameScore: module.cwrap("pb_last_game_score", "number", []),
	};
	const readState = createStateReader(module);
	window.flyBrain.module = module; // diagnostics hook (headless tests, console)
	// House record in the in-game High Scores table (same record as the Score History window).
	module.ccall("pb_seed_high_score", "number", ["string", "number"], [HOUSE_RECORD.name, HOUSE_RECORD.score]);
	setLoadStatus("Running");

	// Sound effects are played by the engine itself (SDL_mixer; original WAVs preloaded by build_wasm.sh). Browsers
	// keep audio suspended until the first click/key press, which the Emscripten SDL port resumes by itself.

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

	// Touch controls for Manual Play on phones: left / right flipper buttons (same state as the Z and slash keys)
	// and a Launch button that pulls the plunger while held (pb_plunger). Shown only in the mobile layout while
	// Manual Play is on; Manual Play can be toggled from the menu without a change event, so visibility is polled.
	const pad = document.createElement("div");
	pad.id = "touch-pad";
	const padButton = (label, cls, down, up) => {
		const b = document.createElement("button");
		b.type = "button";
		b.textContent = label;
		if (cls) b.className = cls;
		const press = (e) => { e.preventDefault(); b.setPointerCapture?.(e.pointerId); b.classList.add("down"); down(); };
		const release = () => { if (b.classList.contains("down")) { b.classList.remove("down"); up(); } };
		b.addEventListener("pointerdown", press);
		b.addEventListener("pointerup", release);
		b.addEventListener("pointercancel", release);
		b.addEventListener("lostpointercapture", release);
		b.addEventListener("contextmenu", (e) => e.preventDefault());
		return b;
	};
	pad.append(
		padButton("Left flipper", "", () => { manualLeft = true; }, () => { manualLeft = false; }),
		padButton("Launch", "launch",
			() => pb.plunger(1),
			() => pb.plunger(0)),
		padButton("Right flipper", "", () => { manualRight = true; }, () => { manualRight = false; }),
	);
	document.body.append(pad);
	let padOn = false;
	// Checked once per frame; classes are only touched when the state actually changed.
	function updateTouchPad() {
		const on = document.documentElement.classList.contains("mobile") && manualCheckbox.checked;
		if (on === padOn) return;
		if (padOn && !on) { manualLeft = false; manualRight = false; } // pad hidden mid-press: release the flippers
		padOn = on;
		pad.classList.toggle("show", on);
		document.documentElement.classList.toggle("touch-pad-on", on);
	}

	let wasBallInPlay = false;
	let notInPlaySinceTick = 0;
	let lastBallY = null;
	let plungerRestSince = null;
	let plungerPhase = "idle"; // idle -> held (spring pulled back) -> released (waiting for the ball to leave the lane)
	let plungerPhaseAt = 0;
	let plungerTargetPct = PLUNGER_SKILL_PULL_PCT;
	let lastPlungeBalls = null; // balls left (engine BallCount) at the last plunge; equal again = re-deployed ball
	let gameOver = false;
	let gameOverAt = 0;

	const statsBody = document.getElementById("stats-body");
	const history = statsBody ? createScoreHistory(statsBody) : null;

	// Player name for the high-score table: a fresh screen name per game, handed to the engine, which
	// inserts a qualifying score under it directly (patch 0010) instead of opening the name dialog.
	let playerName = "";
	let endsAtGameStart = 0;
	let pendingEnd = null; // { brain } of a game the page saw end (state.done) but the engine has not finalized yet
	function startGameBookkeeping() {
		playerName = randomFlyName(playerName);
		module.ccall("pb_set_player_name", null, ["string"], [playerName]);
		endsAtGameStart = pb.gameEndCount();
	}
	startGameBookkeeping();

	// The engine ends a game (pb::end_game: high-score insert) ~3 s AFTER state.done turns true (end-of-game
	// timer) or at once when New Game is pressed in between, so the Score History row is written when
	// the engine's game-end counter ticks - that is also when the "made the high-score table" marker is known.
	function recordFinishedGame() {
		if (!pendingEnd || pb.gameEndCount() <= endsAtGameStart) return;
		history?.add({
			name: playerName,
			brain: pendingEnd.brain,
			score: pb.lastGameScore(),
		});
		pendingEnd = null;
	}

	// The plunger (pb_plunger / pb_menu_command 2) is only usable while running, not paused/demo.
	function canAutoLaunch() {
		const flags = pb.menuState();
		return (flags & (1 << 0)) === 0 && (flags & (1 << 1)) === 0 && (flags & (1 << 2)) !== 0;
	}

	function frame() {
		updateTouchPad();
		const state = readState();
		if (state) {
			hideSplash();
			if (state.done) {
				// Keep polling so a "New Game" from the menu bar can revive the controller.
				const now = performance.now();
				if (!gameOver) {
					gameOver = true;
					gameOverAt = now;
					pendingEnd = { brain: window.flyBrain.kind === "gf" ? "Giant fiber" : "Trained" };
				}
				recordFinishedGame();
				if (!manualCheckbox.checked && now - gameOverAt >= AUTO_NEW_GAME_DELAY_MS
					&& (pb.menuState() & 1) === 0) {
					// Unattended demo: start the next game by itself (retried every delay if it did not take).
					gameOverAt = now;
					newGameRequested = true;
					pb.menuCommand(1, 0);
				}
				requestAnimationFrame(frame);
				return;
			}
			recordFinishedGame(); // New Game can finalize the previous game inside the engine call
			if (gameOver || newGameRequested) {
				pendingEnd = null;
				gameOver = false;
				newGameRequested = false;
				lastPlungeBalls = null; // first ball of a new game gets the skill-shot plunge again
				startGameBookkeeping();
			}
			window.flyBrain.state = state; // read-only hook for the dashboard / diagnostics

			// A training episode is one ball life and ends at a drain (pinball_env.py's drain
			// detector). After a drain the next ball parks on the plunger with ball_in_play still
			// true, so the in-play edge below never fires for it - detect the drain the same way
			// training does, so the circuit (and the delay1 flipper state) reset per life.
			const drained = lastBallY !== null && lastBallY > DRAIN_FROM_Y && inPlungerLane(state.ball_x, state.ball_y);
			lastBallY = state.ball_y;
			// ball_in_play is a position heuristic (primary ball further than 1 unit from the table origin), so
			// a ball rolling through the middle of the table flickers it to false for a few ticks. Only an
			// absence of at least MIN_BALL_GAP_TICKS counts as "between balls".
			if (!state.ball_in_play && wasBallInPlay) notInPlaySinceTick = state.tick;
			const newBall = state.ball_in_play && !wasBallInPlay && state.tick - notInPlaySinceTick >= MIN_BALL_GAP_TICKS;
			if (newBall || drained) {
				resetController(); // one training episode = one ball life; circuit state resets with it
			}
			wasBallInPlay = state.ball_in_play;

			if (manualCheckbox.checked) {
				pb.setAction(manualLeft ? 1 : 0, manualRight ? 1 : 0, 0);
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
				// Launch flag stays 0: the engine feeds each ball into the plunger lane by itself (~1.4 s after
				// New Game in this build, patch 0011; ~2.7 s after the drain bonus) and the plunger pull below fires it.
				pb.setAction(heldAction.flipper_left ? 1 : 0, heldAction.flipper_right ? 1 : 0, 0);
			}

			// A ball parked in the plunger lane is launched like a player does it: the plunger is held back for
			// (visible spring pull, engine export pb_plunger) until the pull reaches the target percentage, then released. The engine's own
			// "Launch Ball" command (no pull animation) is only the fallback if the ball is still parked
			// AUTO_LAUNCH_FALLBACK_MS after the release. No synthetic key events, so it works regardless of keyboard
			// focus or an open canvas dialog. Both exported balls are checked: after a drain the primary slot
			// (ball_*) keeps a stale position, and the engine-fed ball is often in the second slot (ball2_*).
			// Never while the Manual checkbox is on (a human plays the plunger then).
			const primaryParked = state.ball_in_play && inPlungerLane(state.ball_x, state.ball_y)
				&& Math.hypot(state.ball_vx, state.ball_vy) < PLUNGER_MAX_REST_SPEED;
			const secondParked = state.ball2_active && inPlungerLane(state.ball2_x, state.ball2_y)
				&& Math.hypot(state.ball2_vx, state.ball2_vy) < PLUNGER_MAX_REST_SPEED;
			const now = performance.now();
			if ((primaryParked || secondParked) && !manualCheckbox.checked) {
				if (plungerRestSince === null) plungerRestSince = now;
				const canLaunch = canAutoLaunch();
				if (plungerPhase === "idle") {
					if (now - plungerRestSince > AUTO_LAUNCH_HOLD_MS && canLaunch) {
						const balls = pb.diag(3);
						plungerTargetPct = balls !== lastPlungeBalls ? PLUNGER_SKILL_PULL_PCT : PLUNGER_FULL_PULL_PCT;
						lastPlungeBalls = balls;
						pb.plunger(1);
						plungerPhase = "held";
						plungerPhaseAt = now;
					}
				} else if (plungerPhase === "held") {
					if (!canLaunch) { // paused / demo while pulling: let go, start over later
						pb.plunger(0);
						plungerPhase = "idle";
						plungerRestSince = now;
					} else if (pb.diag(2) >= plungerTargetPct
						|| now - plungerPhaseAt >= PLUNGER_MAX_HOLD_MS) {
						pb.plunger(0);
						plungerPhase = "released";
						plungerPhaseAt = now;
						if (!primaryParked) { // the primary-slot edge detectors above cannot see this ball
							resetController(); // one training episode = one ball life; circuit state resets with it
						}
					}
				} else if (canLaunch) {
					const sinceRelease = now - plungerPhaseAt;
					if (plungerTargetPct < PLUNGER_FULL_PULL_PCT) {
						// The partial pull left the ball in the lane: pull again, fully this time (visible pull again).
						if (sinceRelease > PLUNGER_RETRY_MS) { plungerPhase = "idle"; plungerRestSince = now - AUTO_LAUNCH_HOLD_MS; }
					} else if (sinceRelease > AUTO_LAUNCH_FALLBACK_MS) {
						// Even a full pull did not move it: engine "Launch Ball" command as the last resort.
						pb.menuCommand(2, 0);
						plungerPhase = "idle";
						plungerRestSince = now;
					}
				}
			} else {
				if (plungerPhase === "held") pb.plunger(0); // Manual toggled on mid-pull
				plungerPhase = "idle";
				plungerRestSince = null;
			}

			// The flipper state the engine was just given (controller decision or Manual keys) - the Paint window
			// (paint.js) shows the matching drawing.
			window.flyBrain.pressed = manualCheckbox.checked
				? { left: manualLeft, right: manualRight }
				: { left: !!heldAction.flipper_left, right: !!heldAction.flipper_right };
		}
		requestAnimationFrame(frame);
	}
	requestAnimationFrame(frame);
}

main().catch((err) => {
	console.error(err);
	setText(splashStatusEl, `Error: ${err.message}`); // the splash stays up so the error is visible
});

// Surface errors on-page too, not just in devtools console - useful when checking from a
// device without easy devtools access (e.g. a phone).
window.addEventListener("error", (e) => {
	if (e.message.startsWith("ResizeObserver loop")) return; // harmless browser notice, not an application error
	if (splashState === "shown") setText(splashStatusEl, `window.onerror: ${e.message} (${e.filename}:${e.lineno})`);
});
window.addEventListener("unhandledrejection", (e) => {
	if (splashState === "shown") setText(splashStatusEl, `unhandledrejection: ${e.reason}`);
});
