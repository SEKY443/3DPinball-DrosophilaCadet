// Low-res toggle: on by default; the Notepad line switches it. The choice is remembered per browser (if storage works).
import { storageGet, storageSet } from "./dom.js?v=1";

const KEY = "flyPinball.lowres";
const toggle = document.getElementById("lowres-toggle");
// The pixel filter is verified only on Blink. navigator.userAgentData (Chromium-only) is the clean check but is
// missing on plain-http pages (secure contexts only), so fall back to the "Chrome/<n>" UA token - WebKit
// browsers (Safari, Orion) and Firefox don't send it. The filter math assumes a 2x screen.
// On https (GitHub Pages) the presence of userAgentData alone decides, so a WebKit browser that spoofs a Chrome
// user-agent string (e.g. Orion's compatibility mode) still never gets the filter.
const blink = window.isSecureContext
	? !!(navigator.userAgentData && navigator.userAgentData.brands.some((b) => /Chromium/i.test(b.brand)))
	: /Chrome\/\d+/.test(navigator.userAgent); // plain-http LAN testing; also matches HeadlessChrome
const dpr = window.devicePixelRatio || 1;
// Firefox (Gecko) also renders this filter family (feFlood/feTile/feComposite pixelation); enabled on request,
// not yet verified on a 2x screen.
const gecko = /Firefox\/\d+/.test(navigator.userAgent);
const filterOk = (blink || gecko) && dpr >= 1.75 && dpr < 2.5;
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
