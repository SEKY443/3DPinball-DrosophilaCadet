// Low-res toggle: on by default; the Notepad line switches it. The choice is remembered per browser (if storage works).
import { storageGet, storageSet } from "./dom.js?v=1";

const KEY = "flyPinball.lowres";
const toggle = document.getElementById("lowres-toggle");
// The pixel filter runs only on Blink (Chrome/Edge/Brave/Opera) with a 2x screen:
// - WebKit (Safari, Orion) blanks SVG-filtered HTML, and Orion can report Chromium brands / a Chrome UA, so any
//   browser that looks like WebKit (Apple vendor string or the WebKit-only GestureEvent) is excluded first.
// - Firefox renders the filter but too slowly for the live windows, so it gets the plain low-res look too.
// navigator.userAgentData (Chromium-only) is the clean Blink check on https; plain-http LAN pages don't expose it,
// so there the "Chrome/<n>" UA token is used (also matches HeadlessChrome).
const webkit = /Apple/.test(navigator.vendor || "") || "GestureEvent" in window;
const blink = !webkit && (window.isSecureContext
	? !!(navigator.userAgentData && navigator.userAgentData.brands.some((b) => /Chromium/i.test(b.brand)))
	: /Chrome\/\d+/.test(navigator.userAgent));
const dpr = window.devicePixelRatio || 1;
const filterOk = blink && dpr >= 1.75 && dpr < 2.5;
let on = storageGet(KEY) !== "off"; // storage blocked or empty: keep the default (on)
const apply = () => {
	document.documentElement.classList.toggle("lowres", on);
	document.documentElement.classList.toggle("lowres-filter", on && filterOk);
	if (toggle) toggle.textContent = on ? "i wanna retina display" : "take me back to 2003";
};
const flip = () => {
	on = !on;
	storageSet(KEY, on ? "on" : "off");
	apply();
};
toggle?.addEventListener("click", flip);
toggle?.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); flip(); } });
apply();
