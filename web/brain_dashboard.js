// "Fly Brain Monitor" dashboard, styled after the Windows XP Task Manager "Performance" tab: beige window
// body, XP group boxes, sunken stat fields, black graph panels with a dark-green grid and bright-green traces,
// segmented green progress bars and LED-like press indicators. Live wiring graph, neuron activity grid,
// flipper drive / looming-drive bars, stat fields and a timeline, all fed from window.flyBrain (set by
// bridge.js, read-only here).
// window.flyBrain.policy.activity: 444-cell circuit state in [-1, 1], updated once per decision.
// window.flyBrain.last: the latest decision's { logits, rates, actions }.
// When window.flyBrain.kind === "gf" (untrained giant-fiber body, gf_body.js) the wiring graph is laid out
// by looming channel with the two giant fibers highlighted and the bar panel becomes a looming-drive /
// GF-activity readout (policy.internals); otherwise the trained popcode view is shown.
// DOM is built with createElement/textContent only; drawing is plain canvas.

import { el } from "./dom.js?v=1";

const MAX_EDGES = 500;
const HISTORY = 240;
const GRID_STEP = 14; // graph grid spacing in px (Task Manager style)
const FONT = '11px Tahoma, Verdana, "Segoe UI", sans-serif';
const C = {
	face: "#ECE9D8", black: "#000000", grid: "#1b6b1b", green: "#00ff00", label: "#59d659",
	exc: [0, 255, 0], inh: [255, 64, 64], press: { left: "#ffff00", right: "#ff4040" },
};

// Activity colours on the black graph panels: excitatory = green, inhibitory = red, brightness = magnitude.
// Zero stays a dim green so the cell structure remains visible.
function shade(v) {
	const a = Math.max(-1, Math.min(1, v));
	const t = Math.abs(a);
	return a >= 0 ? `rgb(${Math.round(20 * t)},${Math.round(48 + 207 * t)},${Math.round(20 * t)})`
		: `rgb(${Math.round(48 + 207 * t)},${Math.round(30 * (1 - t))},${Math.round(30 * (1 - t))})`;
}

function nodeColor(v) {
	const a = Math.max(-1, Math.min(1, v));
	const t = Math.abs(a);
	return a >= 0 ? `rgb(${Math.round(30 * t)},${Math.round(90 + 165 * t)},${Math.round(30 * t)})`
		: `rgb(${Math.round(110 + 145 * t)},50,50)`;
}

// Set when a canvas was resized (which clears it) so the next frame redraws everything.
let dirty = true;

// Cells that are neither inputs nor outputs, ascending.
function interneuronIndices(brain) {
	const skip = new Set([...brain.inputCells, ...brain.outputs]);
	const inter = [];
	for (let i = 0; i < brain.n; i++) if (!skip.has(i)) inter.push(i);
	return inter;
}

// A canvas that tracks its element's content size (device-pixel-ratio aware).
function sizedCanvas(parent, onResize) {
	const canvas = el("canvas");
	canvas.style.display = "block";
	canvas.style.width = "100%";
	canvas.style.height = "100%";
	parent.appendChild(canvas);
	const ctx = canvas.getContext("2d");
	const state = { canvas, ctx, w: 0, h: 0 };
	new ResizeObserver(() => {
		dirty = true; // resizing clears the canvas: redraw on the next frame
		const dpr = window.devicePixelRatio || 1;
		const w = parent.clientWidth, h = parent.clientHeight;
		canvas.width = Math.max(1, Math.round(w * dpr));
		canvas.height = Math.max(1, Math.round(h * dpr));
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		state.w = w;
		state.h = h;
		if (onResize) onResize(state);
	}).observe(parent);
	return state;
}

// Black panel with the dark-green Task Manager grid; `offset` scrolls it horizontally.
function graphBackground(ctx, w, h, offset = 0) {
	ctx.fillStyle = C.black;
	ctx.fillRect(0, 0, w, h);
	ctx.strokeStyle = C.grid;
	ctx.lineWidth = 1;
	ctx.beginPath();
	for (let x = w - (offset % GRID_STEP) - 0.5; x > 0; x -= GRID_STEP) { ctx.moveTo(x, 0); ctx.lineTo(x, h); }
	for (let y = h - 0.5; y > 0; y -= GRID_STEP) { ctx.moveTo(0, y); ctx.lineTo(w, y); }
	ctx.stroke();
}

// Canvas label with a black backing box so it stays legible on top of nodes and edges.
function graphLabel(ctx, text, x, y, color = C.label) {
	ctx.font = FONT;
	ctx.textBaseline = "top";
	const tw = ctx.measureText(text).width;
	ctx.fillStyle = "rgba(0,0,0,0.85)";
	ctx.fillRect(x - 2, y - 1, tw + 4, 14);
	ctx.fillStyle = color;
	ctx.fillText(text, x, y);
	ctx.textBaseline = "alphabetic";
}

function buildDom(body) {
	const style = el("style");
	style.textContent = `
		.dash { position: absolute; inset: 0; display: grid; gap: 10px 8px; padding: 12px 8px 8px; overflow: hidden;
			grid-template-columns: 1.35fr 1fr; grid-template-rows: auto minmax(0, 1fr) auto;
			background: ${C.face}; font: 11px Tahoma, Verdana, "Segoe UI", sans-serif; color: #000; }
		.dash .gb { position: relative; display: flex; flex-direction: column; min-width: 0; min-height: 0;
			padding: 12px 6px 6px; border: 1px solid #d0d0bf; border-radius: 4px; box-shadow: 1px 1px 0 #fff, inset 1px 1px 0 #fff; }
		.dash .gb > .cap { position: absolute; top: -8px; left: 8px; padding: 0 3px; background: ${C.face};
			color: #0046d5; line-height: 14px; white-space: nowrap; }
		.dash .graph { position: relative; flex: 1 1 0; min-height: 0; min-width: 0; background: #000;
			border: 1px solid; border-color: #808080 #fff #fff #808080; box-shadow: inset 1px 1px 0 #404040; }
		.dash .stats { grid-column: 1 / 3; flex-direction: row; gap: 8px; }
		.dash .stat { flex: 1 1 0; min-width: 0; display: flex; flex-direction: column; gap: 2px; }
		.dash .stat .k { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
		.dash .field { height: 19px; padding: 0 5px; line-height: 17px; background: #fff; white-space: nowrap; overflow: hidden;
			border: 1px solid; border-color: #7f9db9 #fff #fff #7f9db9; box-shadow: inset 1px 1px 0 #404040; }
		.dash .bottom { grid-column: 1 / 3; display: grid; grid-template-columns: 1.35fr 1fr; gap: 8px; min-height: 0; }
		.dash .bottom .gb { height: 118px; }
		.dash .bars { display: grid; gap: 4px 14px; align-content: center; flex: 1 1 auto; min-height: 0; }
		.dash .bars.two { grid-template-columns: 1fr 1fr; }
		.dash .col { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
		.dash .row { display: flex; align-items: center; gap: 5px; height: 16px; min-width: 0; }
		.dash .row .lbl { flex: 0 0 52px; white-space: nowrap; }
		.dash .row .val { flex: 0 0 34px; text-align: right; white-space: nowrap; }
		.dash .pbar { position: relative; flex: 1 1 auto; min-width: 20px; height: 15px; padding: 2px; background: #fff;
			border: 1px solid #686868; border-radius: 3px; }
		.dash .pbar .track { position: relative; height: 100%; }
		.dash .pbar .fill { position: absolute; left: 0; top: 0; bottom: 0; width: 0;
			background: linear-gradient(180deg, rgba(255,255,255,0.5), rgba(255,255,255,0) 55%, rgba(0,0,0,0.12)),
				repeating-linear-gradient(90deg, #2ec92e 0 6px, transparent 6px 8px); }
		.dash .pbar .mark { position: absolute; top: -3px; bottom: -3px; width: 1px; background: #ff0000; }
		.dash .led { flex: 0 0 9px; width: 9px; height: 9px; background: #3d0000; border: 1px solid; border-color: #404040 #c0c0c0 #c0c0c0 #404040; }
		.dash .led.on { background: #ff2a2a; box-shadow: 0 0 3px #ff0000; }
		.dash .note { color: #404040; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
		.dash .legend { display: flex; gap: 12px; align-items: center; height: 15px; margin-top: 3px; white-space: nowrap; overflow: hidden; }
		.dash .legend .sw { display: inline-block; width: 8px; height: 8px; margin-right: 4px; vertical-align: -1px; border: 1px solid #404040; }
	`;
	body.appendChild(style);
	body.style.position = "relative";
	body.style.background = C.face;
	body.style.overflow = "hidden";
	// The window body is a flex column; the dashboard fills it.
	const wrap = el("div");
	wrap.style.cssText = "position: relative; flex: 1 1 auto; min-height: 0;";
	body.appendChild(wrap);
	const dash = el("div", "dash");

	const group = (title, cls) => {
		const g = el("div", `gb${cls ? " " + cls : ""}`);
		const cap = el("div", "cap", title);
		g.append(cap);
		return { g, cap };
	};
	const graph = (parent) => { const d = el("div", "graph"); parent.append(d); return d; };

	const stats = group("Statistics", "stats");
	const tileVals = {};
	for (const [key, label] of [["rate", "Decisions / s"], ["active", "Active neurons"], ["mean", "Mean |activity|"], ["press", "Presses L / R"]]) {
		const s = el("div", "stat");
		s.append(el("div", "k", label));
		const v = el("div", "field", "-");
		s.append(v);
		stats.g.append(s);
		tileVals[key] = v;
	}
	const wiring = group("Neural wiring - strongest live connections");
	const wiringArea = graph(wiring.g);
	const grid = group("Neuron activity");
	const gridArea = graph(grid.g);

	const bottom = el("div", "bottom");
	const gauges = group("Flipper drive");
	const gaugeBody = el("div", "bars");
	gauges.g.append(gaugeBody);
	const timeline = group("Timeline - interneuron activity and presses");
	const timelineArea = graph(timeline.g);
	const legend = el("div", "legend");
	for (const [color, text] of [[C.green, "Mean activity"], [C.press.left, "L press"], [C.press.right, "R press"]]) {
		const item = el("span");
		const sw = el("span", "sw");
		sw.style.background = color;
		item.append(sw, document.createTextNode(text));
		legend.append(item);
	}
	timeline.g.append(legend);
	bottom.append(gauges.g, timeline.g);

	dash.append(stats.g, wiring.g, grid.g, bottom);
	wrap.appendChild(dash);
	return { tileVals, wiringArea, gridArea, timelineArea, gaugeBody, gaugeCap: gauges.cap };
}

// One progress-bar row: label, segmented green bar (+ optional threshold marker), value text, optional LED.
function barRow(label, opts = {}) {
	const row = el("div", "row");
	const lbl = el("span", "lbl", label);
	const pbar = el("div", "pbar");
	const track = el("div", "track");
	const fill = el("div", "fill");
	track.append(fill);
	let mark = null;
	if (opts.mark !== undefined) {
		mark = el("div", "mark");
		mark.style.left = `${Math.max(0, Math.min(1, opts.mark)) * 100}%`;
		track.append(mark);
	}
	pbar.append(track);
	const val = el("span", "val", "");
	row.append(lbl, pbar, val);
	let led = null;
	if (opts.led) { led = el("span", "led"); row.append(led); }
	return {
		row, mark,
		set(v, text, on) {
			fill.style.width = `${Math.max(0, Math.min(1, v)) * 100}%`;
			val.textContent = text;
			if (led) led.classList.toggle("on", !!on);
		},
		setMark(m) { if (mark) mark.style.left = `${Math.max(0, Math.min(1, m)) * 100}%`; },
		setLabel(t) { lbl.textContent = t; },
	};
}

function setup() {
	const win = document.getElementById("win-dash");
	if (!win) return;
	const dom = buildDom(win.querySelector(".window-body"));

	let brainRef = null;
	let wiringPos = null, role = null, edges = [];
	let gridLayout = null;
	let interIdx = []; // interneuron cell indices (neither input nor output) of the current brain
	const history = [];      // { mean, left, right } per decision
	let totalSamples = 0;
	let lastDecisions = -1;
	let shownRate = -1; // Decisions/s value currently shown in its tile
	let presses = [0, 0], prevAct = [false, false];
	const decisionTimes = [];
	let gaugeKind = null, rows = null;

	const wiring = sizedCanvas(dom.wiringArea, () => { wiringPos = null; });
	const grid = sizedCanvas(dom.gridArea, () => { gridLayout = null; });
	const timeline = sizedCanvas(dom.timelineArea);

	// Bars panel: rebuilt whenever the controller kind changes.
	function buildGauges(kind) {
		gaugeKind = kind;
		dom.gaugeBody.replaceChildren();
		if (kind === "gf") {
			dom.gaugeCap.textContent = "Looming drive and giant-fiber activity";
			dom.gaugeBody.className = "bars two";
			const colA = el("div", "col"), colB = el("div", "col");
			const drives = ["LC4 L", "LC4 R", "LPLC2 L", "LPLC2 R"].map((t) => barRow(t));
			drives.forEach((r) => colA.append(r.row));
			const gfs = ["GF L", "GF R"].map((t) => barRow(t, { mark: 0.5, led: true }));
			gfs.forEach((r) => colB.append(r.row));
			const note1 = el("div", "note", "");
			const note2 = el("div", "note", "");
			colB.append(note1, note2);
			dom.gaugeBody.append(colA, colB);
			rows = { drives, gfs, note1, note2 };
		} else {
			dom.gaugeCap.textContent = "Flipper drive";
			dom.gaugeBody.className = "bars";
			const bars = ["LEFT", "RIGHT"].map((t) => barRow(t, { mark: 0.5, led: true }));
			bars.forEach((r) => dom.gaugeBody.append(r.row));
			const note = el("div", "note", "Red line = press threshold (logit 0)");
			dom.gaugeBody.append(note);
			rows = { bars };
		}
	}

	function updateGauges(fb) {
		if (gaugeKind !== fb.kind) buildGauges(fb.kind);
		if (fb.kind === "gf") {
			const policy = fb.policy, it = policy.internals;
			rows.drives.forEach((r, k) => {
				const g = k < 2 ? policy.params.g4 : policy.params.g2;
				r.set(it.drives[k] / g, it.drives[k].toFixed(2));
			});
			rows.gfs.forEach((r, k) => {
				r.setMark(policy.thetaOn);
				r.set(it.a[k], it.a[k].toFixed(2), it.pressed[k]);
			});
			rows.note1.textContent = `Press threshold: ${policy.thetaOn.toFixed(2)}`;
			rows.note2.textContent = `theta L/R: ${it.theta[0].toFixed(2)} / ${it.theta[1].toFixed(2)} rad`;
		} else {
			const last = fb.last;
			const labels = fb.policy.actionLabels;
			[0, 1].forEach((k) => {
				const rate = last ? last.rates[k] : 0;
				rows.bars[k].set(rate, `${Math.round(rate * 100)}%`, !!(last && last.actions[labels[k]]));
			});
		}
	}

	// Giant-fiber layout: sensory cells in four vertical bands (LC4_L, LC4_R, LPLC2_L, LPLC2_R), interneurons
	// in the middle disc, the two GFs (role 2) at the right.
	function layoutWiringGf(brain, w, h) {
		const n = brain.n;
		wiringPos = new Float32Array(2 * n);
		role = new Uint8Array(n).fill(1);
		const bands = [[], [], [], []];
		brain.inputCells.forEach((c, k) => { role[c] = 0; bands[brain.inputChannels[k]].push(c); });
		const top = 20, bandH = (h - top - 4) / 4, step = 4.5;
		bands.forEach((cells, b) => {
			const rows = Math.max(1, Math.floor((bandH - 18) / step));
			cells.forEach((c, k) => {
				wiringPos[2 * c] = 8 + Math.floor(k / rows) * step;
				wiringPos[2 * c + 1] = top + b * bandH + 16 + (k % rows) * step;
			});
		});
		brain.outputs.forEach((c, k) => {
			role[c] = 2;
			wiringPos[2 * c] = w - 26;
			wiringPos[2 * c + 1] = h * (k === 0 ? 0.35 : 0.65);
		});
		const inter = interIdx;
		const golden = Math.PI * (3 - Math.sqrt(5));
		const R = Math.min(w * 0.22, h * 0.4);
		inter.forEach((c, k) => {
			const r = R * Math.sqrt((k + 0.5) / inter.length);
			wiringPos[2 * c] = w * 0.58 + r * Math.cos(k * golden);
			wiringPos[2 * c + 1] = h / 2 + 8 + r * Math.sin(k * golden);
		});
	}

	function layoutWiring(brain, w, h) {
		if (window.flyBrain.kind === "gf") return layoutWiringGf(brain, w, h);
		const n = brain.n;
		wiringPos = new Float32Array(2 * n);
		role = new Uint8Array(n).fill(1);
		const sensory = brain.inputCells.map((c, k) => [c, brain.inputChannels[k]]).sort((a, b) => a[1] - b[1]).map(([c]) => c);
		sensory.forEach((c) => { role[c] = 0; });
		brain.outputs.forEach((c) => { role[c] = 2; });
		const cy = h / 2 + 8, ry = h * 0.4;
		const arc = (cells, cx, bulge) => cells.forEach((c, k) => {
			const ang = ((cells.length > 1 ? k / (cells.length - 1) : 0.5) - 0.5) * Math.PI * 0.85;
			wiringPos[2 * c] = cx + bulge * Math.cos(ang);
			wiringPos[2 * c + 1] = cy + ry * Math.sin(ang);
		});
		arc(sensory, w * 0.05, w * 0.06);
		arc(brain.outputs, w * 0.95, -w * 0.06);
		const inter = interIdx;
		const golden = Math.PI * (3 - Math.sqrt(5));
		const R = Math.min(w * 0.28, h * 0.4);
		inter.forEach((c, k) => {
			const r = R * Math.sqrt((k + 0.5) / inter.length);
			wiringPos[2 * c] = w / 2 + r * Math.cos(k * golden);
			wiringPos[2 * c + 1] = cy + r * Math.sin(k * golden);
		});
	}

	// The MAX_EDGES strongest live connections (|weight * presynaptic activity|, ties by edge index), strongest
	// first, scaled by the strongest. Buffers are kept per brain; the K-th largest magnitude is found with one
	// typed-array sort and only the edges at or above it are ordered.
	let edgeBuf = null;
	function strongestEdges(brain, act) {
		const E = brain.edgePre.length;
		if (!edgeBuf || edgeBuf.E !== E) {
			edgeBuf = { E, sig: new Float32Array(E), sorted: new Float32Array(E), pool: [] };
		}
		const { sig, sorted, pool } = edgeBuf;
		for (let e = 0; e < E; e++) sig[e] = brain.edgeWeight[e] * act[brain.edgePre[e]];
		for (let e = 0; e < E; e++) sorted[e] = Math.abs(sig[e]);
		sorted.sort();
		const K = Math.min(MAX_EDGES, E);
		const thr = K > 0 ? sorted[E - K] : 0;
		const idx = [];
		for (let e = 0; e < E; e++) if (Math.abs(sig[e]) > thr) idx.push(e);
		for (let e = 0; e < E && idx.length < K; e++) if (Math.abs(sig[e]) === thr) idx.push(e);
		idx.sort((a, b) => (Math.abs(sig[b]) - Math.abs(sig[a])) || (a - b));
		const maxAbs = Math.max(1e-6, Math.abs(sig[idx[0]] || 0));
		return idx.map((e, i) => {
			const edge = pool[i] || (pool[i] = { pre: 0, post: 0, s: 0 });
			edge.pre = brain.edgePre[e]; edge.post = brain.edgePost[e]; edge.s = sig[e] / maxAbs;
			return edge;
		});
	}

	// Neuron activity grid: the largest cell size (<= 20 px) at which all three groups fit the panel height.
	function layoutGrid(brain, w, h) {
		const sensory = brain.inputCells.map((c, k) => [c, brain.inputChannels[k]]).sort((a, b) => a[1] - b[1]).map(([c]) => c);
		const groups = [["Sensory", sensory], ["Interneurons", interIdx], ["Readout", brain.outputs]];
		const pad = 6, labelH = 15, gap = 4;
		let cell = 3, cols = 1;
		for (let c = 20; c >= 3; c--) {
			cols = Math.max(1, Math.floor((w - 2 * pad) / c));
			const total = 2 * pad + groups.reduce((acc, [, cells]) => acc + labelH + Math.ceil(cells.length / cols) * c + gap, 0);
			cell = c;
			if (total <= h) break;
		}
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
		graphBackground(ctx, w, h);
		ctx.globalCompositeOperation = "lighter";
		for (let i = edges.length - 1; i >= 0; i--) {
			const { pre, post, s } = edges[i];
			const m = Math.abs(s);
			const c = s >= 0 ? C.exc : C.inh;
			ctx.strokeStyle = `rgba(${c[0]},${c[1]},${c[2]},${0.08 + 0.6 * m})`;
			ctx.lineWidth = 0.4 + 1.4 * m;
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
		if (window.flyBrain.kind === "gf") {
			const top = 20, bandH = (h - top - 4) / 4;
			["LC4 L", "LC4 R", "LPLC2 L", "LPLC2 R"].forEach((t, b) => graphLabel(ctx, t, 4, top + b * bandH + 1));
			brain.outputs.forEach((c, k) => {
				const x = wiringPos[2 * c], y = wiringPos[2 * c + 1];
				ctx.strokeStyle = "#ffffff";
				ctx.lineWidth = 2;
				ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2); ctx.stroke();
				ctx.fillStyle = nodeColor(act[c]);
				ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill();
				graphLabel(ctx, k === 0 ? "GF L" : "GF R", x - 14, y - 24);
			});
			graphLabel(ctx, "Looming inputs", 4, 2);
			graphLabel(ctx, "Intermediate cells", Math.max(110, w * 0.58 - 45), 2);
			return;
		}
		graphLabel(ctx, "Sensory", 4, 2);
		graphLabel(ctx, "Interneurons", w / 2 - 28, 2);
		graphLabel(ctx, "Readout", w - 52, 2);
	}

	function drawGrid(brain, act) {
		const { ctx, w, h } = grid;
		if (w < 10 || h < 10) return;
		if (!gridLayout) layoutGrid(brain, w, h);
		ctx.fillStyle = C.black;
		ctx.fillRect(0, 0, w, h);
		ctx.font = FONT;
		for (const g of gridLayout) {
			ctx.fillStyle = C.label;
			ctx.fillText(g.name, g.x, g.y - 4);
			const s = g.cell - 1;
			g.cells.forEach((c, k) => {
				ctx.fillStyle = shade(act[c]);
				ctx.fillRect(g.x + (k % g.cols) * g.cell, g.y + Math.floor(k / g.cols) * g.cell, s, s);
			});
		}
	}

	function drawTimeline() {
		const { ctx, w, h } = timeline;
		if (w < 10 || h < 10) return;
		const pad = 4, top = 6, gh = h - 22;
		const step = (w - 2 * pad) / (HISTORY - 1);
		graphBackground(ctx, w, h, totalSamples * step);
		if (history.length < 2) return;
		const maxV = Math.max(0.05, ...history.map((x) => x.mean));
		// Newest sample sits at the right edge and older ones scroll left, like the Task Manager graphs.
		const px = (i) => w - pad - (history.length - 1 - i) * step;
		ctx.strokeStyle = C.green;
		ctx.lineWidth = 1.2;
		ctx.beginPath();
		history.forEach((x, i) => {
			const py = top + gh - (x.mean / maxV) * gh;
			if (i === 0) ctx.moveTo(px(i), py); else ctx.lineTo(px(i), py);
		});
		ctx.stroke();
		history.forEach((x, i) => {
			if (x.left) { ctx.fillStyle = C.press.left; ctx.fillRect(px(i), h - 13, 2, 5); }
			if (x.right) { ctx.fillStyle = C.press.right; ctx.fillRect(px(i), h - 7, 2, 5); }
		});
	}

	function updateTiles(act) {
		let active = 0;
		for (let i = 0; i < act.length; i++) active += Math.abs(act[i]) > 0.1 ? 1 : 0;
		const mean = history.length ? history[history.length - 1].mean : 0;
		dom.tileVals.rate.textContent = String(shownRate);
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
				graphBackground(ctx, w, h);
				graphLabel(ctx, "Waiting for the fly brain...", 10, 10);
			}
			return;
		}
		const brain = fb.policy.brain;
		if (brain !== brainRef) {
			brainRef = brain; wiringPos = null; gridLayout = null; // controller switched: restart the stats
			history.length = 0; totalSamples = 0; presses = [0, 0]; prevAct = [false, false]; lastDecisions = -1;
			interIdx = interneuronIndices(brain);
			dirty = true;
		}
		const act = fb.policy.activity;
		if (fb.decisions !== lastDecisions) {
			lastDecisions = fb.decisions;
			decisionTimes.push(performance.now());
			edges = strongestEdges(brain, act);
			const labels = fb.policy.actionLabels;
			const cur = fb.last ? [!!fb.last.actions[labels[0]], !!fb.last.actions[labels[1]]] : [false, false];
			for (let s = 0; s < 2; s++) { if (cur[s] && !prevAct[s]) presses[s]++; }
			let sum = 0;
			for (const i of interIdx) sum += Math.abs(act[i]);
			history.push({ mean: sum / Math.max(1, interIdx.length), left: cur[0] && !prevAct[0], right: cur[1] && !prevAct[1] });
			totalSamples++;
			if (history.length > HISTORY) history.shift();
			prevAct = cur;
			dirty = true;
		}
		// Everything below only changes with a new decision (or a resize / brain switch), except the Decisions/s tile,
		// which also decays as time passes.
		const now = performance.now();
		while (decisionTimes.length && now - decisionTimes[0] > 1000) decisionTimes.shift();
		const rate = decisionTimes.length;
		if (!dirty && rate === shownRate) return;
		shownRate = rate;
		if (dirty) {
			dirty = false;
			drawWiring(brain, act);
			drawGrid(brain, act);
			updateGauges(fb);
			drawTimeline();
		}
		updateTiles(act);
	}
	requestAnimationFrame(frame);
}

setup();
