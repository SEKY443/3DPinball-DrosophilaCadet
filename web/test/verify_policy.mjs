// Replays real-engine test vectors (agents/export_pinball_circuit.py's --test-vectors output)
// through observation.js and circuit_policy.js and checks both JS ports match the live PyTorch
// agent, using real gameplay data (no synthetic/random inputs).
//
// Two separate checks per step:
//  1. Teacher-forced single-step check (the pass/fail gate): JS's circuit activity is reset to
//     the Python agent's own activity_before this step, so only ONE forward pass is compared -
//     no compounding. Gated on RELATIVE, not absolute, logit error (see below for why).
//  2. Free-running trajectory check (diagnostic only, printed but not asserted): JS carries its
//     OWN activity across all 200 steps, like a real deployment would.
//
// Why relative error: these CMA-trained decoders are unregularized linear heads on top of a
// gain=1.4 recurrent circuit, and empirically land on very large weights - the 108-neuron
// checkpoint's logits reach +-1300. A ~1e-7-relative float32 rounding difference in the shared
// upstream computation (edge-sum order differs between this sequential JS loop and PyTorch's
// index_add_, which may reduce in a different, possibly multi-threaded order) is faithfully
// amplified by that large linear gain into an absolute logit difference up to ~8e-3 - not a JS
// porting bug (single-step relative error stays ~1e-4, and greedy actions never disagree; see
// agents/export_pinball_circuit.py's _write_test_vectors docstring for the free-running/chaotic-
// looking trajectory numbers this same amplification produces over many compounded steps).
// Checking relative error, and separately that the greedy action never disagrees, is what
// actually matters for correctness; a fixed absolute bound would be meaningless across
// checkpoints whose logit scale is itself an emergent, unconstrained property of training.
//
// Usage: node web/test/verify_policy.mjs <connectome.json> <readout.json> <vectors.json>

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { PinballCircuitPolicyJS } from "../circuit_policy.js";
import { stateToObs } from "../observation.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const OBS_TOLERANCE = 1e-5;
const RELATIVE_LOGIT_TOLERANCE = 2e-4;
// per-input-cell drive of a sensory encoding (float32 values in [-1, 1]): at most a few float32 ulps
const DRIVE_TOLERANCE = 1e-6;

function loadJson(path) {
	return JSON.parse(readFileSync(path, "utf8"));
}

function maxAbsDiff(a, b) {
	let m = 0;
	for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
	return m;
}

function maxRelDiff(actual, expected) {
	let m = 0;
	for (let i = 0; i < actual.length; i++) {
		m = Math.max(m, Math.abs(actual[i] - expected[i]) / Math.max(1, Math.abs(expected[i])));
	}
	return m;
}

function run(connectomePath, readoutPath, vectorsPath, label) {
	const graph = loadJson(connectomePath);
	const readout = loadJson(readoutPath);
	const vectors = loadJson(vectorsPath);

	// Bypasses PinballCircuitPolicyJS.load (fetch-based, no DOM/fetch needed in plain Node) -
	// same constructor either way.
	const teacherForced = new PinballCircuitPolicyJS(graph, readout);
	const freeRunning = new PinballCircuitPolicyJS(graph, readout);

	let obsMaxDiff = 0;
	let singleStepMaxAbsDiff = 0;
	let singleStepMaxRelDiff = 0;
	let trajectoryMaxAbsDiff = 0;
	let actionMismatches = 0;

	// Vectors from agents/experiments/encoding/export_enc.py also carry the flipper_obs (delay1)
	// state before each step (prev_flippers, null right after a circuit reset), the filtered obs
	// and the per-input-cell drive; older vector files have none of these and are checked as before.
	const hasFilterState = vectors.steps.length > 0 && "prev_flippers" in vectors.steps[0];
	let filterMaxDiff = 0;
	let driveMaxDiff = 0;

	vectors.steps.forEach((step, i) => {
		const jsObs = stateToObs(step.raw);
		obsMaxDiff = Math.max(obsMaxDiff, maxAbsDiff(jsObs, step.obs));

		if (hasFilterState) {
			teacherForced.prevFlippers = step.prev_flippers;
			const filtered = teacherForced.filterObs(step.obs);
			filterMaxDiff = Math.max(filterMaxDiff, maxAbsDiff(filtered, step.filtered_obs));
			const drive = teacherForced.encodeInputs(step.filtered_obs);
			if (drive) driveMaxDiff = Math.max(driveMaxDiff, maxAbsDiff(drive, step.input_drive));
			teacherForced.prevFlippers = step.prev_flippers; // forward() applies the filter itself
			if (i > 0 && step.prev_flippers === null) freeRunning.reset(); // new life in the recording
		}
		teacherForced.activity.set(step.activity_before);
		const single = teacherForced.forward(step.obs);
		singleStepMaxAbsDiff = Math.max(singleStepMaxAbsDiff, maxAbsDiff(single.logits, step.logits));
		singleStepMaxRelDiff = Math.max(singleStepMaxRelDiff, maxRelDiff(single.logits, step.logits));

		const free = freeRunning.forward(step.obs);
		trajectoryMaxAbsDiff = Math.max(trajectoryMaxAbsDiff, maxAbsDiff(free.logits, step.logits));

		const expectedLeft = step.logits[0] > 0;
		const expectedRight = step.logits[1] > 0;
		if (single.actions.flipper_left !== expectedLeft || single.actions.flipper_right !== expectedRight) {
			actionMismatches++;
		}
	});

	console.log(`[${label}] steps=${vectors.steps.length} obs_max_diff=${obsMaxDiff.toExponential(3)} ` +
		`single_step_logit_max_abs_diff=${singleStepMaxAbsDiff.toExponential(3)} ` +
		`single_step_logit_max_rel_diff=${singleStepMaxRelDiff.toExponential(3)} ` +
		`free_running_trajectory_max_abs_diff=${trajectoryMaxAbsDiff.toExponential(3)} (diagnostic only) ` +
		`action_mismatches=${actionMismatches}` +
		(hasFilterState ? ` flipper_filter_max_diff=${filterMaxDiff.toExponential(3)} ` +
			`input_drive_max_diff=${driveMaxDiff.toExponential(3)}` : ""));

	const ok = obsMaxDiff < OBS_TOLERANCE && singleStepMaxRelDiff < RELATIVE_LOGIT_TOLERANCE && actionMismatches === 0
		&& filterMaxDiff === 0 && driveMaxDiff < DRIVE_TOLERANCE;
	if (!ok) {
		console.error(`[${label}] FAILED (obs tolerance ${OBS_TOLERANCE}, relative logit tolerance ${RELATIVE_LOGIT_TOLERANCE})`);
	}
	return ok;
}

function main() {
	const args = process.argv.slice(2);
	let ok;
	if (args.length >= 3) {
		ok = run(args[0], args[1], args[2], args[3] || "custom");
	} else {
		// Default: verify all shipped models.
		const web = join(HERE, "..");
		const results = [
			run(join(web, "connectome_444.json"), join(web, "circuit_readout_444.json"),
				join(web, "testdata", "circuit_444_vectors.json"), "444-neuron"),
			run(join(web, "connectome.json"), join(web, "circuit_readout_108.json"),
				join(web, "testdata", "circuit_108_vectors.json"), "108-neuron"),
			run(join(web, "connectome_444.json"), join(web, "circuit_readout_444_popcode.json"),
				join(web, "testdata", "circuit_444_popcode_vectors.json"), "444-neuron popcode+delay1"),
		];
		ok = results.every(Boolean);
	}
	if (!ok) process.exit(1);
	console.log("all checks passed");
}

main();
