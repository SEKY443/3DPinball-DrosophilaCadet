// "Player Self-Portrait.bmp - Paint": an MS Paint (Windows XP) lookalike window whose canvas shows a drawing of the
// fly at a keyboard. Which drawing is shown follows the flipper state the bridge hands to the engine
// (window.flyBrain.pressed = { left, right }, set by bridge.js every frame - controller decision or Manual keys):
//   nothing pressed -> fly_idle.png, left (Z) -> fly_left.png, right (slash / colon key) -> fly_right.png,
//   both -> composite of the two (left half of the "right" drawing + right half of the "left" drawing, i.e. the
//   arm that is down in each picture).
// Menus, tools and palette are decoration only. DOM is built with createElement(NS)/textContent only.
import { xpScrollbar } from "./xp_scrollbar.js?v=3";
import { el, svgEl } from "./dom.js?v=1";

const IMG_VERSION = 3;
const SOURCES = {
	idle: `./paint/fly_idle.png?v=${IMG_VERSION}`,
	left: `./paint/fly_left.png?v=${IMG_VERSION}`,
	right: `./paint/fly_right.png?v=${IMG_VERSION}`,
};

// XP default palette, row by row (14 per row).
const PALETTE = [
	"#000000", "#808080", "#800000", "#808000", "#008000", "#008080", "#000080", "#800080", "#808040", "#004040", "#0080ff", "#004080", "#4000ff", "#804000",
	"#ffffff", "#c0c0c0", "#ff0000", "#ffff00", "#00ff00", "#00ffff", "#0000ff", "#ff00ff", "#ffff80", "#00ff80", "#80ffff", "#8080ff", "#ff0080", "#ff8040",
];

// 16 tool glyphs on a 16x16 grid: [tag, attributes]; "text" entries carry their content as `text`.
const TOOLS = [
	[["polygon", { points: "3,4 7,2 12,4 13,8 9,13 4,11", fill: "none", stroke: "#000", "stroke-dasharray": "2 1.5" }]], // free-form select
	[["rect", { x: 2.5, y: 3.5, width: 11, height: 9, fill: "none", stroke: "#000", "stroke-dasharray": "2 1.5" }]], // select
	[["polygon", { points: "3,10 8,4 13,7 8,13", fill: "#f4a0b4", stroke: "#000" }]], // eraser
	[["polygon", { points: "3,8 8,3 12,7 7,12", fill: "#fff", stroke: "#000" }], ["circle", { cx: 13, cy: 12, r: 1.8, fill: "#0000ff" }]], // fill
	[["line", { x1: 3, y1: 13, x2: 10, y2: 6, stroke: "#000", "stroke-width": 2 }], ["circle", { cx: 12, cy: 4, r: 2.2, fill: "#808080", stroke: "#000" }]], // pick colour
	[["circle", { cx: 6.5, cy: 6.5, r: 3.8, fill: "#e8f4ff", stroke: "#000" }], ["line", { x1: 9.5, y1: 9.5, x2: 14, y2: 14, stroke: "#000", "stroke-width": 2.2 }]], // magnifier
	[["polygon", { points: "3,13 4,10 11,3 13,5 6,12", fill: "#f0d000", stroke: "#000" }]], // pencil
	[["line", { x1: 4, y1: 13, x2: 11, y2: 4, stroke: "#8b5a2b", "stroke-width": 2.4 }], ["circle", { cx: 3.8, cy: 13, r: 1.6, fill: "#000" }]], // brush
	[["rect", { x: 5, y: 7, width: 6, height: 7, fill: "#a0a0a0", stroke: "#000" }], ["circle", { cx: 12, cy: 3, r: 0.8, fill: "#000" }], ["circle", { cx: 9, cy: 2.5, r: 0.8, fill: "#000" }], ["circle", { cx: 7, cy: 4.5, r: 0.8, fill: "#000" }]], // airbrush
	[["text", { x: 8, y: 13, "text-anchor": "middle", "font-size": 14, "font-weight": "bold", "font-family": "serif", fill: "#000", text: "A" }]], // text
	[["line", { x1: 3, y1: 13, x2: 13, y2: 3, stroke: "#000", "stroke-width": 1.4 }]], // line
	[["path", { d: "M2 12 C5 2 10 14 14 4", fill: "none", stroke: "#000", "stroke-width": 1.4 }]], // curve
	[["rect", { x: 2.5, y: 4.5, width: 11, height: 8, fill: "none", stroke: "#000" }]], // rectangle
	[["polygon", { points: "3,12 4,4 10,3 13,9 8,13", fill: "none", stroke: "#000" }]], // polygon
	[["ellipse", { cx: 8, cy: 8, rx: 5.5, ry: 4, fill: "none", stroke: "#000" }]], // ellipse
	[["rect", { x: 2.5, y: 4.5, width: 11, height: 8, rx: 3, fill: "none", stroke: "#000" }]], // rounded rectangle
];
const DEFAULT_TOOL = 6; // pencil, as in Paint

// Key press effect: while a flipper is pressed, its key in the drawing is filled grey. Corner points of the two keys
// in the 549x620 drawings (traced on fly_idle.png; the pressed drawings are aligned to it), scaled to the loaded size.
// The Z key is the left flipper's, the / key the right flipper's. "multiply" keeps the black pencil strokes on top.
const KEY_REF_WIDTH = 549;
const KEY_POLYGONS = {
	left: [[289, 491], [430, 549], [386, 613], [243, 559]], // Z key
	right: [[80, 406], [196, 450], [150, 519], [36, 466]], // / key
};
const KEY_PRESSED_FILL = "#a9a9a9";

function toolIcon(shapes) {
	const svg = svgEl("svg", { viewBox: "0 0 16 16", width: 16, height: 16 });
	for (const [tag, { text, ...attrs }] of shapes) {
		const node = svgEl(tag, attrs);
		if (text !== undefined) node.textContent = text;
		svg.append(node);
	}
	return svg;
}

function buildWindow(body) {
	const style = el("style");
	style.textContent = `
		#win-paint .window-body { display: flex; flex-direction: column; background: var(--face); overflow: hidden; }
		.pt-main { flex: 1 1 auto; display: flex; min-height: 0; }
		.pt-tools { flex: 0 0 auto; width: 58px; padding: 4px 3px; }
		.pt-tool-grid { display: grid; grid-template-columns: 25px 25px; }
		.pt-tool { width: 25px; height: 25px; display: flex; align-items: center; justify-content: center;
			border: 1px solid; border-color: #fff #808080 #808080 #fff; box-shadow: inset -1px -1px 0 #404040; background: var(--face); }
		.pt-tool.sel { border-color: #808080 #fff #fff #808080; box-shadow: inset 1px 1px 0 #404040;
			background: repeating-conic-gradient(#fff 0 25%, #d4d0c8 0 50%) 0 0 / 2px 2px; }
		.pt-opt { width: 52px; height: 64px; margin-top: 6px; background: var(--face); border: 1px solid; border-color: #808080 #fff #fff #808080; }
		.pt-work-frame { position: relative; flex: 1 1 auto; min-width: 0; display: flex; }
		.pt-work { position: absolute; inset: 0 17px 17px 0; padding: 4px; overflow: auto; background: #808080;
			border: 1px solid; border-color: #808080 #fff #fff #808080; box-shadow: inset 1px 1px 0 #404040; }
		.pt-sheet { position: relative; display: inline-block; margin: 0 5px 5px 0; background: #fff; }
		.pt-sheet canvas { display: block; }
		.pt-handle { position: absolute; width: 3px; height: 3px; background: #000080; }
		.pt-palette { flex: 0 0 auto; display: flex; align-items: center; gap: 6px; padding: 3px 4px; }
		.pt-sel { position: relative; width: 32px; height: 32px; border: 1px solid; border-color: #808080 #fff #fff #808080; flex: 0 0 auto; }
		.pt-sel i { position: absolute; width: 14px; height: 14px; border: 1px solid; border-color: #808080 #fff #fff #808080; }
		.pt-swatches { display: grid; grid-template-columns: repeat(14, 16px); grid-auto-rows: 16px; }
		.pt-swatch { border: 1px solid; border-color: #808080 #fff #fff #808080; box-shadow: inset 1px 1px 0 #404040; }
		.pt-status { flex: 0 0 auto; display: flex; padding: 2px; border-top: 1px solid #fff; }
		.pt-status span { flex: 1 1 auto; padding: 1px 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
			border: 1px solid; border-color: #808080 #fff #fff #808080; }
	`;
	body.append(style);

	const main = el("div", "pt-main");
	const tools = el("div", "pt-tools");
	const grid = el("div", "pt-tool-grid");
	TOOLS.forEach((shapes, i) => {
		const t = el("div", i === DEFAULT_TOOL ? "pt-tool sel" : "pt-tool");
		t.append(toolIcon(shapes));
		grid.append(t);
	});
	tools.append(grid, el("div", "pt-opt"));
	const work = el("div", "pt-work");
	const sheet = el("div", "pt-sheet");
	const canvas = el("canvas");
	sheet.append(canvas);
	for (const pos of [{ right: "-4px", top: "calc(50% - 1px)" }, { left: "calc(50% - 1px)", bottom: "-4px" }, { right: "-4px", bottom: "-4px" }]) {
		const h = el("span", "pt-handle");
		Object.assign(h.style, pos);
		sheet.append(h);
	}
	work.append(sheet);
	// Greyed-out XP scrollbars around the drawing area: the picture fits, as in XP Paint (decoration only).
	const workFrame = el("div", "pt-work-frame");
	const vbar = xpScrollbar("v", { disabled: true });
	const hbar = xpScrollbar("h", { disabled: true });
	vbar.classList.add("with-h");
	hbar.classList.add("with-v");
	workFrame.append(work, vbar, hbar, el("div", "xp-sb-corner"));
	main.append(tools, workFrame);

	const palette = el("div", "pt-palette");
	const sel = el("div", "pt-sel");
	const bg = el("i"); bg.style.cssText = "right: 3px; bottom: 3px; background: #fff;";
	const fg = el("i"); fg.style.cssText = "left: 3px; top: 3px; background: #000;";
	sel.append(bg, fg);
	const swatches = el("div", "pt-swatches");
	for (const color of PALETTE) {
		const sw = el("div", "pt-swatch");
		sw.style.background = color;
		swatches.append(sw);
	}
	palette.append(sel, swatches);

	const status = el("div", "pt-status");
	status.append(el("span", null, "For Help, click Help Topics on the Help Menu."));
	body.append(main, palette, status);
	return { canvas, work };
}

function loadImage(src) {
	return new Promise((resolve, reject) => {
		const img = new Image();
		img.onload = () => resolve(img);
		img.onerror = () => reject(new Error(`could not load ${src}`));
		img.src = src;
	});
}

async function setup() {
	const win = document.getElementById("win-paint");
	if (!win) return;
	const { canvas, work } = buildWindow(win.querySelector(".window-body"));
	const ctx = canvas.getContext("2d");

	// Preload all three drawings and pre-render the "both pressed" composite, so switching never flickers.
	const [idle, left, right] = await Promise.all([loadImage(SOURCES.idle), loadImage(SOURCES.left), loadImage(SOURCES.right)]);
	const w = idle.naturalWidth, h = idle.naturalHeight;
	canvas.width = w;
	canvas.height = h;
	const half = Math.floor(w / 2);
	const both = document.createElement("canvas");
	both.width = w;
	both.height = h;
	const bctx = both.getContext("2d");
	bctx.fillStyle = "#fff";
	bctx.fillRect(0, 0, w, h);
	bctx.drawImage(right, 0, 0, half, h, 0, 0, half, h);           // left half: the "right key" drawing (left arm down)
	bctx.drawImage(left, half, 0, w - half, h, half, 0, w - half, h); // right half: the "left key" drawing (right arm down)

	// The sheet is scaled to fit the work area (never above the drawing's natural size), so it re-lays out whenever
	// the window is resized.
	new ResizeObserver(() => {
		const availW = work.clientWidth - 8 - 10, availH = work.clientHeight - 8 - 10;
		if (availW < 20 || availH < 20) return;
		const scale = Math.min(1, availW / w, availH / h);
		canvas.style.width = `${Math.round(w * scale)}px`;
		canvas.style.height = `${Math.round(h * scale)}px`;
	}).observe(work);

	const images = { idle, left, right, both };
	let shown = null;
	function show(key) {
		if (key === shown) return;
		shown = key;
		ctx.fillStyle = "#fff";
		ctx.fillRect(0, 0, w, h);
		ctx.drawImage(images[key], 0, 0, w, h);
		ctx.save();
		ctx.globalCompositeOperation = "multiply";
		ctx.fillStyle = KEY_PRESSED_FILL;
		for (const side of ["left", "right"]) {
			if (key !== side && key !== "both") continue;
			const k = w / KEY_REF_WIDTH;
			ctx.beginPath();
			KEY_POLYGONS[side].forEach(([x, y], i) => (i ? ctx.lineTo(x * k, y * k) : ctx.moveTo(x * k, y * k)));
			ctx.closePath();
			ctx.fill();
		}
		ctx.restore();
		canvas.dataset.pose = key; // diagnostics / tests
	}
	show("idle");

	// The drawing follows window.flyBrain.pressed (the flipper actions the bridge sends to the engine) directly,
	// with no hold time, so it changes exactly when the flippers do.
	function frame() {
		requestAnimationFrame(frame);
		const p = window.flyBrain?.pressed;
		const left = !!p?.left, right = !!p?.right;
		show(left ? (right ? "both" : "left") : right ? "right" : "idle");
	}
	requestAnimationFrame(frame);
}

setup().catch((err) => console.error(err));
