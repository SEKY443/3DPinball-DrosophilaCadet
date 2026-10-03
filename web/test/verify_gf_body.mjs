// Replays real-engine obs sequences (scripts/export_gf_web.py) through gf_body.js and checks the JS port of
// the split giant-fiber body reproduces the Python body: actions EXACTLY, GF activities and (on sampled
// steps) the full 373-cell activity vector within 1e-5. State is carried across each life, like deployment.
//
// Usage: node web/test/verify_gf_body.mjs [gf_body_split.json] [vectors.json]

import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { SplitGiantFiberPolicyJS } from "../gf_body.js";
import { runParity } from "./verify_gf_common.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
runParity(
	SplitGiantFiberPolicyJS,
	process.argv[2] || join(HERE, "..", "gf_body_split.json"),
	process.argv[3] || join(HERE, "..", "testdata", "gf_body_split_vectors.json"),
);
