// Small DOM / storage helpers shared by the page modules. Text is only ever set through textContent, so nothing
// here builds HTML from strings.

export const SVG_NS = "http://www.w3.org/2000/svg";

// createElement with an optional class name and optional text content.
export function el(tag, cls, text) {
	const node = document.createElement(tag);
	if (cls) node.className = cls;
	if (text !== undefined) node.textContent = text;
	return node;
}

// Sets an element's text; tolerates a missing element.
export function setText(node, text) {
	if (node) node.textContent = text;
}

// createElementNS for SVG with an attribute map (values are stringified).
export function svgEl(tag, attrs = {}) {
	const node = document.createElementNS(SVG_NS, tag);
	for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
	return node;
}

// localStorage that never throws (storage blocked, full or unavailable): reads fall back, writes are dropped.
export function storageGet(key, fallback = null) {
	try {
		const value = window.localStorage.getItem(key);
		return value === null ? fallback : value;
	} catch {
		return fallback;
	}
}

export function storageSet(key, value) {
	try {
		window.localStorage.setItem(key, value);
	} catch {
		// storage unavailable or full: keep the in-memory state only
	}
}
