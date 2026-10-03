// Windows XP (Luna) scrollbars drawn with plain DOM elements. Browsers such as Orion/Safari ignore styled native
// scrollbars (and macOS hides overlay ones), so the page draws its own. Styles live in index.html (.xp-sb).
//
// xpScrollbar("v" | "h")                 -> decorative bar (fixed thumb, does nothing).
// xpScrollbar("v", { disabled: true })   -> greyed-out bar, as XP shows when the content fits (Notepad, Paint).
// xpScrollbar("v", { target: element })  -> live bar for a scrolling element whose native bar is hidden: the thumb
//                                           follows target.scrollTop, arrows/track scroll it, the thumb can be dragged.
import { el, svgEl } from "./dom.js?v=1";

const ARROW = {
	up: "M4 9.5 L7.5 6 L11 9.5",
	down: "M4 5.5 L7.5 9 L11 5.5",
	left: "M9.5 4 L6 7.5 L9.5 11",
	right: "M5.5 4 L9 7.5 L5.5 11",
};

function arrowButton(dir) {
	const btn = el("div", "xp-sb-btn");
	const svg = svgEl("svg", { width: 15, height: 15, viewBox: "0 0 15 15" });
	svg.append(svgEl("path", { d: ARROW[dir], fill: "none", stroke: "#4d6185", "stroke-width": 2 }));
	btn.append(svg);
	return btn;
}

export function xpScrollbar(orientation = "v", { target = null, thumbFraction = 0.6, disabled = false } = {}) {
	const vertical = orientation === "v";
	const bar = el("div", `xp-sb ${vertical ? "v" : "h"}`);
	bar.setAttribute("aria-hidden", "true");
	const dec = arrowButton(vertical ? "up" : "left");
	const inc = arrowButton(vertical ? "down" : "right");
	const track = el("div", "xp-sb-track");
	const thumb = el("div", "xp-sb-thumb");
	for (let i = 0; i < 3; i++) thumb.append(el("span"));
	track.append(thumb);
	bar.append(dec, track, inc);

	const setThumb = (fraction, offset) => {
		thumb.style[vertical ? "height" : "width"] = `${Math.max(8, fraction * 100)}%`;
		thumb.style[vertical ? "top" : "left"] = `${offset * 100}%`;
	};
	if (!target) {
		bar.classList.toggle("disabled", disabled);
		setThumb(thumbFraction, 0);
		return bar;
	}

	const sizeKey = vertical ? "clientHeight" : "clientWidth";
	const totalKey = vertical ? "scrollHeight" : "scrollWidth";
	const posKey = vertical ? "scrollTop" : "scrollLeft";
	const sync = () => {
		const total = target[totalKey], view = target[sizeKey];
		const fraction = total > 0 ? Math.min(1, view / total) : 1;
		const maxPos = Math.max(1, total - view);
		bar.classList.toggle("disabled", fraction >= 1);
		setThumb(fraction, (target[posKey] / maxPos) * (1 - fraction));
	};
	const step = (delta) => { target[posKey] += delta; };
	dec.addEventListener("click", () => step(-20));
	inc.addEventListener("click", () => step(20));
	track.addEventListener("pointerdown", (e) => {
		if (e.target !== track) return;
		const r = thumb.getBoundingClientRect();
		const before = vertical ? e.clientY < r.top : e.clientX < r.left;
		step((before ? -1 : 1) * target[sizeKey] * 0.9);
	});
	thumb.addEventListener("pointerdown", (e) => {
		e.preventDefault();
		thumb.setPointerCapture(e.pointerId);
		const start = vertical ? e.clientY : e.clientX;
		const startPos = target[posKey];
		const trackLen = vertical ? track.clientHeight : track.clientWidth;
		const move = (ev) => {
			const d = (vertical ? ev.clientY : ev.clientX) - start;
			target[posKey] = startPos + (d / Math.max(1, trackLen)) * target[totalKey];
		};
		const up = () => {
			thumb.removeEventListener("pointermove", move);
			thumb.removeEventListener("pointerup", up);
		};
		thumb.addEventListener("pointermove", move);
		thumb.addEventListener("pointerup", up);
	});
	// Scroll, resize and content changes are coalesced into one sync per animation frame.
	let syncQueued = false;
	const scheduleSync = () => {
		if (syncQueued) return;
		syncQueued = true;
		requestAnimationFrame(() => { syncQueued = false; sync(); });
	};
	target.addEventListener("scroll", scheduleSync, { passive: true });
	// Content growth is seen through the target's direct children (e.g. the table whose rows are replaced), so
	// only childList is observed, not the whole subtree.
	const resizeObserver = new ResizeObserver(scheduleSync);
	resizeObserver.observe(target);
	for (const child of target.children) resizeObserver.observe(child);
	new MutationObserver((records) => {
		for (const r of records) for (const n of r.addedNodes) if (n.nodeType === 1) resizeObserver.observe(n);
		scheduleSync();
	}).observe(target, { childList: true });
	sync();
	return bar;
}
