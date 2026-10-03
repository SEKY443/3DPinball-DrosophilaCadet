// Replays real-engine obs sequences (scripts/export_gf_web.py) through gf_body.js and checks the JS port of
// the retinotopic giant-fiber body reproduces the Python body: actions EXACTLY, GF activities and (on sampled
// steps) the full full activity vector within 1e-5. State is carried across each life, like deployment.
//
// Usage: node web/test/verify_gf_retina.mjs [gf_body_retina.json] [vectors.json]

import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { RetinaGiantFiberPolicyJS } from "../gf_body.js";
import { runParity } from "./verify_gf_common.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
runParity(
	RetinaGiantFiberPolicyJS,
	process.argv[2] || join(HERE, "..", "gf_body_retina.json"),
	process.argv[3] || join(HERE, "..", "testdata", "gf_body_retina_vectors.json"),
);
