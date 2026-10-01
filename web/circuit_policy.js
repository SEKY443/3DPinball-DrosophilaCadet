// Pure-JS re-implementation of agents/fixed_circuit_agent.py's FixedCircuitAgent forward pass,
// driven by a connectome JSON (the fixed, measured MaleCNS subgraph - see
// agents/build_pinball_connectome.py, e.g. web/connectome.json or web/connectome_444.json) and a
// circuit_readout_*.json (the small trained readout - see agents/export_pinball_circuit.py for
// the schema and web/test/verify_policy.mjs for the verification against the live PyTorch agent
// this must match).
//
// Unlike web/inference.js's DrosophilaControllerJS (a hand-built SNN, fully trained), only the
// readout here is trained; the circuit's own connectivity (PinballConnectomeJS) is measured
// MaleCNS data and never changes - this is the cobanov/flyjump method
// (https://github.com/cobanov/flyjump), not a from-scratch network.

export class PinballConnectomeJS {
	constructor(graph) {
		this.n = graph.nodes.length;
		this.nChannels = graph.channels.length;

		const signs = graph.nodes.map((node) => node.sign);
		// float32 throughout this class (not float64): the trained PyTorch agent runs in
		// float32, and this recurrent tanh loop is sensitive enough over many ticks that
		// float64's extra precision actually DIVERGES from - rather than better approximates -
		// the float32 trajectory the agent was calibrated against (confirmed: float64 gave
		// ~3e-2 logit drift after 200 real decisions vs <1e-6 with matching float32 rounding).
		const totals = new Float32Array(this.n);
		for (const [pre, post, contacts] of graph.edges) {
			totals[post] = Math.fround(totals[post] + Math.fround(contacts * Math.abs(signs[pre])));
		}
		this.edgePre = new Int32Array(graph.edges.map(([pre]) => pre));
		this.edgePost = new Int32Array(graph.edges.map(([, post]) => post));
		this.edgeWeight = new Float32Array(graph.edges.map(([pre, post, contacts]) =>
			totals[post] > 0 ? Math.fround(contacts * signs[pre]) / totals[post] : 0
		));

		this.inputCells = graph.inputs.map(([cell]) => cell);
		this.inputChannels = graph.inputs.map(([, channel]) => channel);
		this.outputs = graph.outputs;
	}

	// activity: Float32Array(n), mutated in place and returned - persists across ticks (the
	// circuit is stateful/recurrent within an episode, reset only at episode start).
	// channelDrive: plain array of length nChannels, already scaled to roughly [-1, 1].
	// inputDrive (optional): Float32Array(graph.inputs.length), one drive value per input cell in
	// graph.inputs order - used by per-cell sensory encodings (see encodeInputs below); when given,
	// channelDrive is ignored.
	step(activity, channelDrive, dynamics, inputDrive = null) {
		const drive = new Float32Array(this.n);
		for (let k = 0; k < this.inputCells.length; k++) {
			drive[this.inputCells[k]] = inputDrive ? inputDrive[k] : channelDrive[this.inputChannels[k]];
		}
		const leak = Math.fround(dynamics.leak);
		const oneMinusLeak = Math.fround(1 - leak);
		const gain = Math.fround(dynamics.gain);
		for (let t = 0; t < dynamics.iterations; t++) {
			const msg = new Float32Array(this.n);
			for (let e = 0; e < this.edgePre.length; e++) {
				msg[this.edgePost[e]] += activity[this.edgePre[e]] * this.edgeWeight[e];
			}
			for (let i = 0; i < this.n; i++) {
				activity[i] = oneMinusLeak * activity[i] + leak * Math.tanh(drive[i] + gain * msg[i]);
			}
		}
		return activity;
	}

	readOutputs(activity) {
		return this.outputs.map((i) => activity[i]);
	}
}

// ---- Optional per-cell sensory encoding (agents/experiments/encoding/encoding.py) ----
// A readout JSON without an "encoding" field keeps the original behaviour: every input cell of a
// channel gets the same clamp(obs / obs_scale, -1, 1). With "encoding": {"name": "popcode", ...}
// the listed channels' cells get Gaussian position bumps ("bumps") or direction-selective sigmoid
// speed units ("ds_speed", one half per direction); every other channel stays broadcast. Math is
// done in float64 on float32-rounded observation values and rounded to float32 once at the end,
// the same as the NumPy encoder (float64 math, cast to float32).

function sigmoid(z, clip) {
	// same clip as encoding.py's _sig: no Math.exp overflow for extreme speeds
	const zc = Math.max(-clip, Math.min(clip, z));
	return 1 / (1 + Math.exp(-zc));
}

function gaussianBump(v, center, sigma) {
	// exp of a non-positive number: underflows to 0 far from the centre, never overflows
	const d = v - center;
	return Math.exp(-(d * d) / (2 * sigma * sigma));
}

// Returns one float64 value per cell of channel `ch`, or null when the channel is broadcast.
function popcodeChannel(enc, ch, obs32) {
	const spec = enc.channels[String(ch)];
	if (!spec) return null;
	const v = obs32[ch];
	if (spec.kind === "bumps") {
		return spec.centers.map((c, k) => gaussianBump(v, c, spec.sigmas[k]));
	}
	if (spec.kind === "ds_speed") {
		const pos = spec.thresholds.map((t, k) => sigmoid((v - t) / spec.scales[k], enc.sigmoid_clip));
		const neg = spec.thresholds.map((t, k) => sigmoid((-v - t) / spec.scales[k], enc.sigmoid_clip));
		return pos.concat(neg);
	}
	throw new Error(`unknown popcode channel kind ${spec.kind}`);
}

export class PinballCircuitPolicyJS {
	constructor(graph, readout) {
		this.brain = new PinballConnectomeJS(graph);
		this.encoding = readout.encoding || null;
		if (this.encoding && this.encoding.name !== "popcode") {
			throw new Error(`unsupported sensory encoding ${this.encoding.name}`);
		}
		// FlipperObsFilter mode (agents/train_pinball_circuit_cem.py): absent/"raw" = unchanged,
		// "delay1" = the policy sees the flipper-state channels from its previous decision step
		// (0 on the first step of a ball life). Other modes are not ported.
		this.flipperObs = readout.flipper_obs || "raw";
		if (!["raw", "delay1"].includes(this.flipperObs)) {
			throw new Error(`unsupported flipper_obs mode ${this.flipperObs}`);
		}
		this.prevFlippers = null; // delay1 state, reset with the circuit
		// input-cell slot -> [channel, index within that channel], in graph.inputs order
		const seen = new Array(graph.channels.length).fill(0);
		this.inputSlots = graph.inputs.map(([, ch]) => [ch, seen[ch]++]);
		this.dynamics = readout.dynamics;
		this.obsScale = readout.obs_scale;
		this.norm = readout.readout_norm;
		this.projWeight = readout.proj_weight; // (readoutDim, nReadout)
		this.projBias = readout.proj_bias || null; // (readoutDim,), absent for every checkpoint so far
		this.head = readout.head; // { weight: (3, readoutDim), bias: (3,) }
		this.actionLabels = readout.action_labels;
		this.activity = new Float32Array(this.brain.n); // persistent circuit state across ticks
	}

	static async load(connectomeUrl, readoutUrl) {
		const [graph, readout] = await Promise.all([
			fetch(connectomeUrl).then((r) => {
				if (!r.ok) throw new Error(`failed to fetch ${connectomeUrl}: ${r.status}`);
				return r.json();
			}),
			fetch(readoutUrl).then((r) => {
				if (!r.ok) throw new Error(`failed to fetch ${readoutUrl}: ${r.status}`);
				return r.json();
			}),
		]);
		return new PinballCircuitPolicyJS(graph, readout);
	}

	reset() {
		this.activity.fill(0);
		this.prevFlippers = null;
	}

	// Applies the flipper_obs filter. Stateful for delay1: call exactly once per decision step.
	filterObs(obs) {
		if (this.flipperObs === "raw") return obs;
		const out = obs.slice();
		const cur = [obs[12], obs[13]];
		out[12] = this.prevFlippers ? this.prevFlippers[0] : 0.0;
		out[13] = this.prevFlippers ? this.prevFlippers[1] : 0.0;
		this.prevFlippers = cur;
		return out;
	}

	// Per-input-cell drive (Float32Array, graph.inputs order) for an already-filtered obs, or null
	// when the readout uses the original per-channel broadcast.
	encodeInputs(obs) {
		if (!this.encoding) return null;
		const obs32 = obs.map((v) => Math.fround(v));
		const perChannel = obs32.map((_, ch) => popcodeChannel(this.encoding, ch, obs32));
		const drive = new Float32Array(this.inputSlots.length);
		for (let s = 0; s < this.inputSlots.length; s++) {
			const [ch, k] = this.inputSlots[s];
			drive[s] = perChannel[ch] ? perChannel[ch][k]
				: Math.max(-1, Math.min(1, obs32[ch] / Math.fround(this.obsScale[ch])));
		}
		return drive;
	}

	// obs: plain array of 15 values, see web/observation.js::stateToObs for the exact order (the
	// UNfiltered observation - the flipper_obs filter is applied here).
	// Returns { logits: Float64Array(3), rates: Float64Array(3), actions: {label: bool} }.
	forward(rawObs) {
		const obs = this.filterObs(rawObs);
		const channelDrive = new Float32Array(obs.length);
		for (let k = 0; k < obs.length; k++) {
			channelDrive[k] = Math.max(-1, Math.min(1, Math.fround(obs[k]) / this.obsScale[k]));
		}
		this.brain.step(this.activity, channelDrive, this.dynamics, this.encodeInputs(obs));
		const raw = this.brain.readOutputs(this.activity);

		const { mean, log_scale: logScale, clip } = this.norm;
		const normed = new Float32Array(raw.length);
		for (let k = 0; k < raw.length; k++) {
			const x = Math.fround(raw[k] - mean[k]) * Math.fround(Math.exp(-logScale[k]));
			normed[k] = Math.max(-clip, Math.min(clip, x));
		}

		const readoutDim = this.projWeight.length;
		const feats = new Float32Array(readoutDim);
		for (let o = 0; o < readoutDim; o++) {
			let sum = this.projBias ? this.projBias[o] : 0;
			const row = this.projWeight[o];
			for (let k = 0; k < normed.length; k++) sum = Math.fround(sum + Math.fround(row[k] * normed[k]));
			feats[o] = sum;
		}

		const logits = new Float32Array(this.head.bias.length);
		for (let a = 0; a < logits.length; a++) {
			let sum = this.head.bias[a];
			const row = this.head.weight[a];
			for (let k = 0; k < readoutDim; k++) sum = Math.fround(sum + Math.fround(row[k] * feats[k]));
			logits[a] = sum;
		}

		const rates = logits.map((z) => 1 / (1 + Math.exp(-z)));
		const actions = {};
		for (let a = 0; a < this.actionLabels.length; a++) {
			// Deterministic threshold (logit > 0, equivalently rate > 0.5) - see
			// web/inference.js's identical convention and its rationale for why this is only
			// correct once training has taught the policy confident probabilities.
			actions[this.actionLabels[a]] = logits[a] > 0;
		}
		return { logits, rates, actions };
	}
}
