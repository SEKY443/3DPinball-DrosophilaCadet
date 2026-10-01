// Pure-JS re-implementation of agents/snn_model.py's DrosophilaController forward pass,
// driven entirely by weights.json (see agents/export_weights.py for the schema and the
// bit-for-bit verification against the trained PyTorch model this must match).
//
// LIF update equations mirror norse.torch.functional.lif.lif_feed_forward_step exactly:
//   i_new = i + input
//   v += dt * tau_mem_inv * ((v_leak - v) + i_new)
//   i = i_new - dt * tau_syn_inv * i_new
//   spike = (v - v_th) > 0 ? 1 : 0   // strict >, matches norse.functional.heaviside
//   v = spike ? v_reset : v

export class DrosophilaControllerJS {
	constructor(weights) {
		this.inputDim = weights.input_dim;
		this.hiddenDim = weights.hidden_dim;
		this.outputDim = weights.output_dim;
		this.simSteps = weights.sim_steps;
		this.inputLinear = weights.input_linear; // { weight: (hidden, input), bias: (hidden,) }
		this.outputLinear = weights.output_linear; // { weight: (output, hidden), bias: (output,) }
		this.hiddenLif = weights.hidden_lif;
		this.outputLif = weights.output_lif;
		this.outputLabels = weights.output_labels;
	}

	static async load(url) {
		const response = await fetch(url);
		if (!response.ok) {
			throw new Error(`failed to fetch ${url}: ${response.status}`);
		}
		const weights = await response.json();
		return new DrosophilaControllerJS(weights);
	}

	// obs: plain array of length input_dim. Returns { rates: Float64Array(output_dim), actions: {label: bool} }
	forward(obs) {
		if (obs.length !== this.inputDim) {
			throw new Error(`expected ${this.inputDim} observation values, got ${obs.length}`);
		}

		let hiddenV = new Float64Array(this.hiddenDim);
		let hiddenI = new Float64Array(this.hiddenDim);
		let outputV = new Float64Array(this.outputDim);
		let outputI = new Float64Array(this.outputDim);
		const spikeSum = new Float64Array(this.outputDim);

		for (let t = 0; t < this.simSteps; t++) {
			const hiddenCurrent = linear(obs, this.inputLinear.weight, this.inputLinear.bias);
			const hiddenSpikes = new Float64Array(this.hiddenDim);
			lifStep(hiddenCurrent, hiddenV, hiddenI, this.hiddenLif, hiddenSpikes);

			const outputCurrent = linear(hiddenSpikes, this.outputLinear.weight, this.outputLinear.bias);
			const outputSpikes = new Float64Array(this.outputDim);
			lifStep(outputCurrent, outputV, outputI, this.outputLif, outputSpikes);

			for (let k = 0; k < this.outputDim; k++) {
				spikeSum[k] += outputSpikes[k];
			}
		}

		const rates = spikeSum.map((s) => s / this.simSteps);
		const actions = {};
		for (let k = 0; k < this.outputLabels.length; k++) {
			// Deterministic: play the policy's mode, not a random sample. This is only the
			// right call once training has actually taught the policy a confident probability
			// for each action (LAUNCH_SUCCESS_BONUS + entropy regularization in
			// agents/train.py exist specifically so "launch" converges near 1 instead of
			// getting stuck near 0 from early bad luck) - a demo shouldn't stand in for
			// exploration a properly-trained policy shouldn't still need.
			actions[this.outputLabels[k]] = rates[k] > 0.5;
		}
		return { rates, actions };
	}
}

function linear(input, weight, bias) {
	// weight: (outDim, inDim) row-major, matching PyTorch nn.Linear.weight and JSON export.
	const outDim = weight.length;
	const out = new Float64Array(outDim);
	for (let o = 0; o < outDim; o++) {
		let sum = bias[o];
		const row = weight[o];
		for (let k = 0; k < input.length; k++) {
			sum += row[k] * input[k];
		}
		out[o] = sum;
	}
	return out;
}

function lifStep(inputCurrent, v, i, params, outSpikes) {
	const { dt, tau_mem_inv, tau_syn_inv, v_leak, v_th, v_reset } = params;
	for (let n = 0; n < v.length; n++) {
		const iNew = i[n] + inputCurrent[n];
		const dv = dt * tau_mem_inv * (v_leak - v[n] + iNew);
		const vDecayed = v[n] + dv;
		const iDecayed = iNew - dt * tau_syn_inv * iNew;

		const spike = (vDecayed - v_th) > 0.0 ? 1.0 : 0.0;
		outSpikes[n] = spike;
		v[n] = spike ? v_reset : vDecayed;
		i[n] = iDecayed;
	}
}
