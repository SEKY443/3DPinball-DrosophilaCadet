// Shared parity check for the giant-fiber body tests (verify_gf_body.mjs, verify_gf_retina.mjs): replays real-engine
// obs sequences (scripts/export_gf_web.py) through a JS body and checks it reproduces the Python body: actions
// EXACTLY, GF activities and (on sampled steps) the full activity vector within 1e-5. State is carried across each
// life, like deployment.

import { readFileSync } from "node:fs";

const TOL = 1e-5;

// PolicyClass: the JS body class (constructed from the graph JSON); graphPath / vectorsPath: JSON files.
// Prints the same report as the old per-body scripts and exits with status 1 on failure.
export function runParity(PolicyClass, graphPath, vectorsPath) {
	const graph = JSON.parse(readFileSync(graphPath, "utf8"));
	const vectors = JSON.parse(readFileSync(vectorsPath, "utf8"));

	const policy = new PolicyClass(graph);
	let steps = 0, mismatches = 0, maxA = 0, maxH = 0, hChecked = 0;
	const presses = [0, 0];
	for (const life of vectors.lives) {
		policy.reset();
		for (const [i, s] of life.steps.entries()) {
			const r = policy.forward(s.obs);
			steps++;
			const act = [r.actions.flipper_left ? 1 : 0, r.actions.flipper_right ? 1 : 0];
			if (act[0] !== s.action[0] || act[1] !== s.action[1]) {
				mismatches++;
				if (mismatches <= 5) console.log(`action mismatch seed ${life.seed} step ${i}: js ${act} py ${s.action}`);
			}
			presses[0] += act[0]; presses[1] += act[1];
			for (let k = 0; k < 2; k++) maxA = Math.max(maxA, Math.abs(policy.internals.a[k] - s.a[k]));
			if (s.h) {
				hChecked++;
				for (let c = 0; c < s.h.length; c++) maxH = Math.max(maxH, Math.abs(policy.activity[c] - s.h[c]));
			}
		}
	}
	console.log(`steps ${steps}, pressed-steps L/R ${presses}, action mismatches ${mismatches}`);
	console.log(`max |GF activity err| ${maxA.toExponential(2)}, max |full-vector err| ${maxH.toExponential(2)} over ${hChecked} sampled steps`);
	if (mismatches > 0 || maxA > TOL || maxH > TOL || presses[0] === 0 || presses[1] === 0) {
		console.log("FAIL");
		process.exit(1);
	}
	console.log("PASS");
}
