// JS port of agents/experiments/pathway_body/gf_split_body.py::SplitGiantFiberBody (untrained
// giant-fiber "split looming" body). LC4 cells get the angular VELOCITY of the ball as seen from each
// flipper, LPLC2 cells its angular SIZE; a fixed (never trained) MaleCNS circuit carries the drive to the
// two giant fibers (DNp01), whose activity gates the flippers with hysteresis, a hold cap and a
// refractory period. Graph, normalized weights, dynamics gain and parameters come from
// gf_body_split.json (written by scripts/export_gf_web.py); verified against Python by
// web/test/verify_gf_body.mjs.
//
// Exposes the same surface the dashboard/bridge use for the trained policy: brain, activity,
// actionLabels, reset(), forward(obs) -> { logits, rates, actions }, plus `internals` for the GF view.

const f32 = Math.fround;

// obs: any array whose first four entries are ball x, y, vx, vy (pinball_env.py OBS_BALL_*).
// Returns [theta, thetaDot] of a ball of radius R seen from target; [0, 0] unless it is closing in.
export function eyeSignals(obs, target, R) {
	const dx = target[0] - obs[0];
	const dy = target[1] - obs[1];
	const d = Math.hypot(dx, dy);
	if (d < 1e-6) return [Math.PI, 0.0];
	const closing = (obs[2] * dx + obs[3] * dy) / d;
	if (closing <= 0) return [0.0, 0.0];
	return [2.0 * Math.atan(R / d), (2.0 * R * closing) / (d * d + R * R)];
}

export class SplitGiantFiberPolicyJS {
	constructor(g) {
		this.graph = g;
		this.n = g.nodes.id.length;
		this.params = g.params;
		this.actionLabels = ["flipper_left", "flipper_right", "launch"];
		this.edgePre = Int32Array.from(g.edges.pre);
		this.edgePost = Int32Array.from(g.edges.post);
		this.edgeWeight = Float32Array.from(g.edges.w);
		this.inputCells = g.inputs.map(([c]) => c);
		this.inputChannels = g.inputs.map(([, ch]) => ch);
		this.outputs = g.outputs; // [GF_L, GF_R]
		this.dynGain = f32(g.dynamics.gain);
		this.leak = f32(g.dynamics.leak);
		this.iterations = g.dynamics.iterations;
		this.thetaOn = g.params.theta_on;
		this.thetaOff = g.params.theta_off;
		// the dashboard reads policy.brain.{n,inputCells,inputChannels,outputs,edgePre,edgePost,edgeWeight}
		this.brain = this;
		this.activity = new Float32Array(this.n);
		// Per-step scratch buffers, reused every decision (zeroed with fill(0) before use).
		this._drive = new Float32Array(this.n);
		this._msg = new Float32Array(this.n);
		this.reset();
	}

	static async load(url) {
		const res = await fetch(url);
		if (!res.ok) throw new Error(`failed to load ${url}: ${res.status}`);
		return new SplitGiantFiberPolicyJS(await res.json());
	}

	reset() {
		this.activity.fill(0);
		this.pressed = [false, false];
		this.run = [0, 0];
		this.rest = [this.params.refractory, this.params.refractory];
		this.internals = {
			drives: [0, 0, 0, 0], // LC4_L, LC4_R, LPLC2_L, LPLC2_R
			theta: [0, 0], thetaDot: [0, 0], // per eye (left, right)
			a: [0, 0], // GF_L, GF_R activity
			pressed: [false, false],
		};
	}

	// One recurrent step: same math as FixedConnectome.step with the precomputed dynamics gain.
	_step(channelDrive) {
		const drive = this._drive;
		drive.fill(0);
		for (let k = 0; k < this.inputCells.length; k++) drive[this.inputCells[k]] = channelDrive[this.inputChannels[k]];
		this._stepDrive(drive);
	}

	// `drive` is the per-cell input (Float32Array of length n).
	_stepDrive(drive) {
		const n = this.n, act = this.activity, msg = this._msg;
		const oneMinusLeak = f32(1 - this.leak);
		for (let t = 0; t < this.iterations; t++) {
			msg.fill(0);
			for (let e = 0; e < this.edgePre.length; e++) msg[this.edgePost[e]] += act[this.edgePre[e]] * this.edgeWeight[e];
			for (let i = 0; i < n; i++) {
				act[i] = f32(oneMinusLeak * act[i]) + f32(this.leak * f32(Math.tanh(f32(drive[i] + f32(this.dynGain * msg[i])))));
			}
		}
	}

	forward(obs) {
		const p = this.params, tg = this.graph.loom_targets;
		const [lTh, lTd] = eyeSignals(obs, tg.left, p.R);
		const [rTh, rTd] = eyeSignals(obs, tg.right, p.R);
		const drives = [
			p.g4 * Math.tanh(lTd / p.v0), p.g4 * Math.tanh(rTd / p.v0),
			p.g2 * Math.tanh(lTh / p.s0), p.g2 * Math.tanh(rTh / p.s0),
		].map(f32);
		this._step(drives);
		return this._finish(drives, [lTh, rTh], [lTd, rTd]);
	}

	// Hysteresis gate + hold cap + refractory on the two GF activities; builds the result and `internals`.
	_finish(drives, theta, thetaDot) {
		const p = this.params;
		const a = [this.activity[this.outputs[0]], this.activity[this.outputs[1]]];
		for (let s = 0; s < 2; s++) {
			if (this.pressed[s]) {
				if (a[s] < this.thetaOff || this.run[s] >= p.hold_max) {
					this.pressed[s] = false; this.run[s] = 0; this.rest[s] = 0;
				} else {
					this.run[s] += 1;
				}
			} else {
				this.rest[s] += 1;
				if (a[s] > this.thetaOn && this.rest[s] > p.refractory) { this.pressed[s] = true; this.run[s] = 1; }
			}
		}
		this.internals = { drives, theta, thetaDot, a, pressed: [this.pressed[0], this.pressed[1]] };
		return {
			logits: [a[0] - this.thetaOn, a[1] - this.thetaOn, 0],
			rates: [a[0], a[1], 0],
			actions: { flipper_left: this.pressed[0], flipper_right: this.pressed[1], launch: false },
			gf: this.internals,
		};
	}
}

// Python cell_signals (gf_retina_body.py): like eyeSignals but d is clamped to 1e-6 instead of special-cased.
// Writes [theta, thetaDot] into `out` (a preallocated 2-element buffer) instead of returning a new array per cell.
function cellSignals(obs, x, y, R, out) {
	const dx = x - obs[0], dy = y - obs[1];
	const d = Math.max(Math.hypot(dx, dy), 1e-6);
	const closing = (obs[2] * dx + obs[3] * dy) / d;
	if (!(closing > 0)) { out[0] = 0.0; out[1] = 0.0; return; }
	out[0] = 2.0 * Math.atan(R / d);
	out[1] = (2.0 * R * closing) / (d * d + R * R);
}

// JS port of gf_retina_body.py::RetinaGiantFiberBody (untrained). Same circuit, dynamics and gating as the split
// body, but every LC4 / LPLC2 cell has its own receptive-field point (x, y) along the flipper sweep line and gets
// its own drive: LC4 g4*tanh(thetaDot/v0), LPLC2 g2*tanh(theta/s0). Points come from gf_body_retina.json
// (input_targets, input_is_lc4, inputs[k][1] = side 0/1; display groups 0..3 = LC4_L, LC4_R, LPLC2_L, LPLC2_R).
// Dashboard internals: drives[g] = MEAN drive over the cells of group g; theta[eye] = MAX theta over that eye's
// LPLC2 cells; thetaDot[eye] = MAX thetaDot over that eye's LC4 cells.
export class RetinaGiantFiberPolicyJS extends SplitGiantFiberPolicyJS {
	constructor(g) {
		super(g);
		this.targets = g.input_targets;
		this.isLc4 = g.input_is_lc4;
		// gf_body_retina.json stores each input cell's SIDE (0 = left eye, 1 = right eye) in inputs[k][1]; the display
		// group is type + side: 0 LC4_L, 1 LC4_R, 2 LPLC2_L, 3 LPLC2_R. The dashboard bands read inputChannels, so it
		// gets the groups too (the retina forward pass doesn't use inputChannels, so the dynamics are unaffected).
		this.groups = this.inputChannels.map((side, k) => (this.isLc4[k] ? 0 : 2) + side);
		this.inputChannels = this.groups;
		this.groupSize = [0, 0, 0, 0];
		for (const gr of this.groups) this.groupSize[gr]++;
		this._sum = [0, 0, 0, 0];
		this._sig = new Float64Array(2); // [theta, thetaDot] of the cell being processed
	}

	static async load(url) {
		const res = await fetch(url);
		if (!res.ok) throw new Error(`failed to load ${url}: ${res.status}`);
		return new RetinaGiantFiberPolicyJS(await res.json());
	}

	forward(obs) {
		const p = this.params;
		const drive = this._drive, sum = this._sum, sig = this._sig;
		drive.fill(0);
		sum.fill(0);
		const theta = [0, 0], thetaDot = [0, 0];
		for (let k = 0; k < this.inputCells.length; k++) {
			cellSignals(obs, this.targets[k][0], this.targets[k][1], p.R, sig);
			const th = sig[0], td = sig[1];
			const gr = this.groups[k], eye = gr & 1;
			const d = f32(this.isLc4[k] ? p.g4 * Math.tanh(td / p.v0) : p.g2 * Math.tanh(th / p.s0));
			drive[this.inputCells[k]] = d;
			sum[gr] += d;
			if (this.isLc4[k]) { if (td > thetaDot[eye]) thetaDot[eye] = td; } else if (th > theta[eye]) theta[eye] = th;
		}
		this._stepDrive(drive);
		return this._finish(sum.map((v, gr) => (this.groupSize[gr] ? v / this.groupSize[gr] : 0)), theta, thetaDot);
	}
}
