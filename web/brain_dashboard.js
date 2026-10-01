// "Fly Brain Monitor" dashboard: live wiring graph, neuron activity grid, flipper drive gauges,
// stat tiles and a timeline, all fed from window.flyBrain (set by bridge.js, read-only here).
// window.flyBrain.policy.activity: 444-cell circuit state in [-1, 1], updated once per decision.
// window.flyBrain.last: the latest decision's { logits, rates, actions }.
// DOM is built with createElement/textContent only; drawing is plain canvas.

const MAX_EDGES = 500;
const HISTORY = 240;
const C = {
	bg: "#0b0e14", panel: "#121722", border: "#232b3b", text: "#9aa4b2", dim: "#5d6778",
	pos: [255, 176, 0], neg: [42, 127, 255], left: "#ffb000", right: "#3fd0ff", line: "#59d18b",
};

function el(tag, cls, text) {
	const e = document.createElement(tag);
	if (cls) e.className = cls;
	if (text !== undefined) e.textContent = text;
	return e;
}

function shade(v, lift = true) {
	const a = Math.max(-1, Math.min(1, v));
	const c = a >= 0 ? C.pos : C.neg;
	const t = Math.abs(a);
	const l = lift ? Math.max(0, t - 0.7) / 0.3 * 80 : 0;
	return `rgb(${Math.round(c[0] * t + l)},${Math.round(c[1] * t + l)},${Math.round(c[2] * t + l)})`;
}

function nodeColor(v) {
	const a = Math.max(-1, Math.min(1, v));
	const t = Math.abs(a);
	return a >= 0 ? `rgb(${Math.round(60 + 195 * t)},${Math.round(60 + 116 * t)},60)`
		: `rgb(60,${Math.round(60 + 67 * t)},${Math.round(60 + 195 * t)})`;
}

// A canvas that tracks its element's size (device-pixel-ratio aware).
function sizedCanvas(parent, onResize) {
	const canvas = el("canvas");
	canvas.style.display = "block";
	canvas.style.width = "100%";
	canvas.style.height = "100%";
	parent.appendChild(canvas);
	const ctx = canvas.getContext("2d");
	const state = { canvas, ctx, w: 0, h: 0 };
	new ResizeObserver(() => {
		const dpr = window.devicePixelRatio || 1;
		const r = parent.getBoundingClientRect();
		canvas.width = Math.max(1, Math.round(r.width * dpr));
		canvas.height = Math.max(1, Math.round(r.height * dpr));
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		state.w = r.width;
		state.h = r.height;
		if (onResize) onResize(state);
	}).observe(parent);
	return state;
}

function buildDom(body) {
	const style = el("style");
	style.textContent = `
		.dash { position: absolute; inset: 0; display: grid; gap: 6px; padding: 6px;
			grid-template-columns: 1.35fr 1fr; grid-template-rows: auto 1fr 70px;
			background: ${C.bg}; font: 10px Tahoma, sans-serif; color: ${C.text}; }
		.dash .tiles { grid-column: 1 / 3; display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
		.dash .tile, .dash .panel { background: ${C.panel}; border: 1px solid ${C.border}; border-radius: 4px; }
		.dash .tile { padding: 4px 8px; }
		.dash .tile .k { text-transform: uppercase; letter-spacing: 0.5px; color: ${C.dim}; font-size: 9px; }
		.dash .tile .v { font: bold 17px "Trebuchet MS", Tahoma, sans-serif; color: #e6edf6; margin-top: 1px; }
		.dash .tile .v small { font-size: 10px; color: ${C.text}; font-weight: normal; }
		.dash .panel { display: flex; flex-direction: column; min-height: 0; min-width: 0; }
		.dash .panel h4 { margin: 0; padding: 4px 8px; font-size: 9px; letter-spacing: 0.6px; text-transform: uppercase;
			color: ${C.dim}; border-bottom: 1px solid ${C.border}; font-weight: normal; }
		.dash .panel .area { position: relative; flex: 1 1 auto; min-height: 0; }
		.dash .right { display: grid; grid-template-rows: 1fr 118px; gap: 6px; min-height: 0; }
		.dash .timeline { grid-column: 1 / 3; }
	`;
	body.appendChild(style);
	body.style.position = "relative";
	body.style.background = C.bg;
	body.style.overflow = "hidden";
	const dash = el("div", "dash");
	const tiles = el("div", "tiles");
	const tileVals = {};
	for (const [key, label] of [["rate", "Decisions / s"], ["active", "Active neurons"], ["mean", "Mean |activity|"], ["press", "Presses L / R"]]) {
		const t = el("div", "tile");
		t.append(el("div", "k", label));
		const v = el("div", "v", "–");
		t.append(v);
		tiles.append(t);
		tileVals[key] = v;
	}
	const panel = (title, cls) => {
		const p = el("div", `panel${cls ? " " + cls : ""}`);
		p.append(el("h4", null, title));
		const area = el("div", "area");
		p.append(area);
		return { p, area };
	};
	const wiring = panel("Neural wiring · strongest live connections");
	const right = el("div", "right");
	const grid = panel("Neuron activity");
	const gauges = panel("Flipper drive");
	right.append(grid.p, gauges.p);
	const timeline = panel("Timeline · interneuron activity and presses", "timeline");
	dash.append(tiles, wiring.p, right, timeline.p);
	body.appendChild(dash);
	return { tileVals, wiringArea: wiring.area, gridArea: grid.area, gaugeArea: gauges.area, timelineArea: timeline.area };
}

function setup() {
	const win = document.getElementById("win-dash");
	if (!win) return;
	const dom = buildDom(win.querySelector(".window-body"));

	let brainRef = null;
	let wiringPos = null, role = null, edges = [];
	let gridLayout = null;
	const history = [];      // { mean, left, right } per decision
	let lastDecisions = -1;
	let presses = [0, 0], prevAct = [false, false];
	const decisionTimes = [];

	const wiring = sizedCanvas(dom.wiringArea, () => { wiringPos = null; });
	const grid = sizedCanvas(dom.gridArea, () => { gridLayout = null; });
	const gauge = sizedCanvas(dom.gaugeArea);
	const timeline = sizedCanvas(dom.timelineArea);

	function layoutWiring(brain, w, h) {
		const n = brain.n;
		wiringPos = new Float32Array(2 * n);
		role = new Uint8Array(n).fill(1);
		const sensory = brain.inputCells.map((c, k) => [c, brain.inputChannels[k]]).sort((a, b) => a[1] - b[1]).map(([c]) => c);
		sensory.forEach((c) => { role[c] = 0; });
		brain.outputs.forEach((c) => { role[c] = 2; });
		const cy = h / 2 + 4, ry = h * 0.42;
		const arc = (cells, cx, bulge) => cells.forEach((c, k) => {
			const ang = ((cells.length > 1 ? k / (cells.length - 1) : 0.5) - 0.5) * Math.PI * 0.85;
			wiringPos[2 * c] = cx + bulge * Math.cos(ang);
			wiringPos[2 * c + 1] = cy + ry * Math.sin(ang);
		});
		arc(sensory, w * 0.05, w * 0.06);
		arc(brain.outputs, w * 0.95, -w * 0.06);
		const inter = [];
		for (let i = 0; i < n; i++) if (role[i] === 1) inter.push(i);
		const golden = Math.PI * (3 - Math.sqrt(5));
		const R = Math.min(w * 0.28, h * 0.42);
		inter.forEach((c, k) => {
			const r = R * Math.sqrt((k + 0.5) / inter.length);
			wiringPos[2 * c] = w / 2 + r * Math.cos(k * golden);
			wiringPos[2 * c + 1] = cy + r * Math.sin(k * golden);
		});
	}

	function strongestEdges(brain, act) {
		const E = brain.edgePre.length;
		const sig = new Float32Array(E);
		for (let e = 0; e < E; e++) sig[e] = brain.edgeWeight[e] * act[brain.edgePre[e]];
		const idx = Array.from({ length: E }, (_, i) => i).sort((a, b) => Math.abs(sig[b]) - Math.abs(sig[a])).slice(0, MAX_EDGES);
		const maxAbs = Math.max(1e-6, Math.abs(sig[idx[0]] || 0));
		return idx.map((e) => ({ pre: brain.edgePre[e], post: brain.edgePost[e], s: sig[e] / maxAbs }));
	}

	function layoutGrid(brain, w) {
		const inputSet = new Set(brain.inputCells), outSet = new Set(brain.outputs);
		const sensory = brain.inputCells.map((c, k) => [c, brain.inputChannels[k]]).sort((a, b) => a[1] - b[1]).map(([c]) => c);
		const inter = [];
		for (let i = 0; i < brain.n; i++) if (!inputSet.has(i) && !outSet.has(i)) inter.push(i);
		const groups = [["Sensory", sensory], ["Interneurons", inter], ["Readout", brain.outputs]];
		const pad = 6, labelH = 11, gap = 4;
		const cell = Math.max(4, Math.floor((w - 2 * pad) / 30));
		const cols = Math.max(1, Math.floor((w - 2 * pad) / cell));
		let y = pad;
		gridLayout = groups.map(([name, cells]) => {
			const block = { name: `${name} (${cells.length})`, cells, x: pad, y: y + labelH, cols, cell };
			y += labelH + Math.ceil(cells.length / cols) * cell + gap;
			return block;
		});
	}

	function drawWiring(brain, act) {
		const { ctx, w, h } = wiring;
		if (w < 10 || h < 10) return;
		if (!wiringPos) layoutWiring(brain, w, h);
		ctx.fillStyle = C.panel;
		ctx.fillRect(0, 0, w, h);
		ctx.globalCompositeOperation = "lighter";
		for (let i = edges.length - 1; i >= 0; i--) {
			const { pre, post, s } = edges[i];
			const m = Math.abs(s);
			ctx.strokeStyle = s >= 0 ? `rgba(255,176,0,${0.08 + 0.6 * m})` : `rgba(42,127,255,${0.08 + 0.6 * m})`;
			ctx.lineWidth = 0.4 + 1.6 * m;
			ctx.beginPath();
			ctx.moveTo(wiringPos[2 * pre], wiringPos[2 * pre + 1]);
			ctx.lineTo(wiringPos[2 * post], wiringPos[2 * post + 1]);
			ctx.stroke();
		}
		ctx.globalCompositeOperation = "source-over";
		for (let i = 0; i < brain.n; i++) {
			ctx.fillStyle = nodeColor(act[i]);
			ctx.beginPath();
			ctx.arc(wiringPos[2 * i], wiringPos[2 * i + 1], role[i] === 1 ? 2.2 : 1.8, 0, Math.PI * 2);
			ctx.fill();
		}
		ctx.font = "9px Tahoma, sans-serif";
		ctx.fillStyle = C.dim;
		ctx.fillText("SENSORY", 4, 10);
		ctx.fillText("INTERNEURONS", w / 2 - 30, 10);
		ctx.fillText("READOUT", w - 46, 10);
	}

	function drawGrid(brain, act) {
		const { ctx, w, h } = grid;
		if (w < 10 || h < 10) return;
		if (!gridLayout) layoutGrid(brain, w);
		ctx.fillStyle = C.panel;
		ctx.fillRect(0, 0, w, h);
		ctx.font = "9px Tahoma, sans-serif";
		for (const g of gridLayout) {
			ctx.fillStyle = C.dim;
			ctx.fillText(g.name, g.x, g.y - 3);
			const s = g.cell - 1;
			g.cells.forEach((c, k) => {
				ctx.fillStyle = shade(act[c]);
				ctx.fillRect(g.x + (k % g.cols) * g.cell, g.y + Math.floor(k / g.cols) * g.cell, s, s);
			});
		}
	}

	function dial(ctx, cx, cy, r, value, label, color, on) {
		// semicircle 0..1, red zone above 0.5 (the press threshold: logit > 0)
		const a0 = Math.PI, a1 = 2 * Math.PI;
		ctx.lineWidth = 7;
		ctx.strokeStyle = "#1c2230";
		ctx.beginPath(); ctx.arc(cx, cy, r, a0, a1); ctx.stroke();
		ctx.strokeStyle = "rgba(255,75,75,0.35)";
		ctx.beginPath(); ctx.arc(cx, cy, r, a0 + Math.PI * 0.5, a1); ctx.stroke();
		ctx.strokeStyle = color;
		ctx.beginPath(); ctx.arc(cx, cy, r, a0, a0 + Math.PI * value); ctx.stroke();
		const ang = a0 + Math.PI * value;
		ctx.strokeStyle = "#e6edf6";
		ctx.lineWidth = 2;
		ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + (r - 2) * Math.cos(ang), cy + (r - 2) * Math.sin(ang)); ctx.stroke();
		ctx.fillStyle = on ? "#ff4b4b" : "#3a1414";
		ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
		ctx.fillStyle = C.text;
		ctx.font = "9px Tahoma, sans-serif";
		ctx.textAlign = "center";
		ctx.fillText(`${label} ${Math.round(value * 100)}%`, cx, cy + 14);
		ctx.textAlign = "start";
	}

	function drawGauges(fb) {
		const { ctx, w, h } = gauge;
		if (w < 10 || h < 10) return;
		ctx.fillStyle = C.panel;
		ctx.fillRect(0, 0, w, h);
		const r = Math.min(w / 4 - 8, h - 30);
		const last = fb.last;
		const labels = fb.policy.actionLabels;
		const rl = last ? last.rates[0] : 0, rr = last ? last.rates[1] : 0;
		dial(ctx, w * 0.27, h - 22, r, rl, "LEFT", C.left, !!(last && last.actions[labels[0]]));
		dial(ctx, w * 0.73, h - 22, r, rr, "RIGHT", C.right, !!(last && last.actions[labels[1]]));
	}

	function drawTimeline() {
		const { ctx, w, h } = timeline;
		if (w < 10 || h < 10) return;
		ctx.fillStyle = C.panel;
		ctx.fillRect(0, 0, w, h);
		if (history.length < 2) return;
		const pad = 6, top = 4, gh = h - 16;
		const maxV = Math.max(0.05, ...history.map((x) => x.mean));
		ctx.strokeStyle = C.line;
		ctx.lineWidth = 1.2;
		ctx.beginPath();
		history.forEach((x, i) => {
			const px = pad + (i / (HISTORY - 1)) * (w - 2 * pad);
			const py = top + gh - (x.mean / maxV) * gh;
			if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
		});
		ctx.stroke();
		history.forEach((x, i) => {
			const px = pad + (i / (HISTORY - 1)) * (w - 2 * pad);
			if (x.left) { ctx.fillStyle = C.left; ctx.fillRect(px, h - 10, 2, 4); }
			if (x.right) { ctx.fillStyle = C.right; ctx.fillRect(px, h - 5, 2, 4); }
		});
	}

	function updateTiles(act) {
		const now = performance.now();
		while (decisionTimes.length && now - decisionTimes[0] > 1000) decisionTimes.shift();
		let active = 0;
		for (let i = 0; i < act.length; i++) active += Math.abs(act[i]) > 0.1 ? 1 : 0;
		const mean = history.length ? history[history.length - 1].mean : 0;
		dom.tileVals.rate.textContent = String(decisionTimes.length);
		dom.tileVals.active.textContent = `${Math.round((100 * active) / act.length)}%`;
		dom.tileVals.mean.textContent = mean.toFixed(3);
		dom.tileVals.press.textContent = `${presses[0]} / ${presses[1]}`;
	}

	function frame() {
		requestAnimationFrame(frame);
		const fb = window.flyBrain;
		if (!fb || !fb.policy) {
			const { ctx, w, h } = wiring;
			if (w > 10 && h > 10) {
				ctx.fillStyle = C.panel;
				ctx.fillRect(0, 0, w, h);
				ctx.fillStyle = C.text;
				ctx.font = "10px Tahoma, sans-serif";
				ctx.fillText("Waiting for the fly brain...", 10, 20);
			}
			return;
		}
		const brain = fb.policy.brain;
		if (brain !== brainRef) { brainRef = brain; wiringPos = null; gridLayout = null; }
		const act = fb.policy.activity;
		if (fb.decisions !== lastDecisions) {
			lastDecisions = fb.decisions;
			decisionTimes.push(performance.now());
			edges = strongestEdges(brain, act);
			const labels = fb.policy.actionLabels;
			const cur = fb.last ? [!!fb.last.actions[labels[0]], !!fb.last.actions[labels[1]]] : [false, false];
			for (let s = 0; s < 2; s++) { if (cur[s] && !prevAct[s]) presses[s]++; }
			let sum = 0, cnt = 0;
			const inSet = new Set(brain.inputCells), outSet = new Set(brain.outputs);
			for (let i = 0; i < brain.n; i++) if (!inSet.has(i) && !outSet.has(i)) { sum += Math.abs(act[i]); cnt++; }
			history.push({ mean: sum / Math.max(1, cnt), left: cur[0] && !prevAct[0], right: cur[1] && !prevAct[1] });
			if (history.length > HISTORY) history.shift();
			prevAct = cur;
		}
		drawWiring(brain, act);
		drawGrid(brain, act);
		drawGauges(fb);
		drawTimeline();
		updateTiles(act);
	}
	requestAnimationFrame(frame);
}

setup();
