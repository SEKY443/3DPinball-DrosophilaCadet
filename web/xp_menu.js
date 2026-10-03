// Windows-XP-style HTML menu bar for the game window (Game / Options / Brain / Help), replacing the
// engine's in-canvas ImGui menu (disabled in the browser build, see src_cpp/patches/0008).
// Engine actions go through the pb_menu_command() / pb_menu_state() exports of
// src_cpp/state_export_wasm.cpp; the Brain menu calls back into bridge.js.
// DOM is built with createElement/textContent only; nothing here builds HTML from strings.

import { el } from "./dom.js?v=1";

// pb_menu_command ids - keep in lockstep with state_export_wasm.cpp.
const CMD = { NEW_GAME: 1, LAUNCH: 2, PAUSE: 3, HIGH_SCORES: 4, DEMO: 5, PLAYERS: 6, SOUNDS: 7, MUSIC: 8, CONTROLS: 9 };
// pb_menu_state bit flags.
const FLAG = { PAUSED: 1 << 0, DEMO: 1 << 1, LAUNCH: 1 << 2, SCORES: 1 << 3, SOUNDS: 1 << 4, MUSIC: 1 << 5 };

const STYLE = `
	.xp-menubar { font: 11px var(--font-ui); display: flex; gap: 0; padding: 1px 2px; background: var(--face); border-bottom: 1px solid #d4d0c8; flex: 0 0 auto; position: relative; z-index: 5; user-select: none; }
	.xp-menubar .mb-top { padding: 2px 7px; cursor: default; color: #000; }
	.xp-menubar .mb-top.open, .xp-menubar .mb-top:hover { background: #316ac5; color: #fff; }
	.xp-dropdown { position: absolute; top: 100%; display: none; min-width: 190px; padding: 2px; background: #fff;
		border: 1px solid #848284; box-shadow: 2px 2px 3px rgba(0,0,0,0.4); color: #000; z-index: 50; }
	.xp-dropdown.open { display: block; }
	.xp-dropdown .mi { position: relative; display: flex; align-items: center; padding: 3px 22px 3px 24px; white-space: nowrap; cursor: default; }
	.xp-dropdown .mi .mi-check { position: absolute; left: 7px; width: 12px; text-align: center; }
	.xp-dropdown .mi .mi-label { flex: 1 1 auto; }
	.xp-dropdown .mi .mi-key { margin-left: 28px; color: inherit; }
	.xp-dropdown .mi .mi-arrow { position: absolute; right: 6px; }
	.xp-dropdown .mi:hover:not(.disabled), .xp-dropdown .mi.sub-open { background: #316ac5; color: #fff; }
	.xp-dropdown .mi.disabled { color: #aca899; }
	.xp-dropdown .msep { height: 0; margin: 3px 2px; border-top: 1px solid #aca899; border-bottom: 1px solid #fff; }
	.xp-dropdown .xp-dropdown { top: -3px; left: 100%; }
	.xp-dialog-back { position: absolute; inset: 0; z-index: 100; display: flex; align-items: center; justify-content: center; background: rgba(0,0,0,0.15); }
	.xp-dialog { font: 11px var(--font-ui); width: 330px; background: var(--face); border: 3px solid var(--luna-border); border-top: 0; border-radius: 8px 8px 0 0; box-shadow: 2px 3px 10px rgba(0,0,0,0.5); }
	.xp-dialog .dlg-title { height: 26px; display: flex; align-items: center; padding: 0 8px; margin: 0 -3px; border-radius: 7px 7px 0 0; background: var(--luna-title);
		color: #fff; font: bold 13px var(--font-title); text-shadow: 1px 1px #0f1089; }
	.xp-dialog .dlg-body { padding: 14px 14px 8px; line-height: 1.45; }
	.xp-dialog .dlg-btns { display: flex; justify-content: center; padding: 6px 0 12px; }
	.xp-dialog .dlg-btns button { min-width: 75px; padding: 3px 8px; font: 11px var(--font-ui); }
`;

function showAbout(host, brainLabel) {
	const back = el("div", "xp-dialog-back");
	const dlg = el("div", "xp-dialog");
	dlg.setAttribute("role", "dialog");
	dlg.append(el("div", "dlg-title", "About Drosophila Cadet"));
	const body = el("div", "dlg-body");
	body.append(
		el("div", null, "3D Pinball Drosophila Cadet"),
		el("div", null, "A browser demo of a pinball table played by a fruit-fly connectome (MaleCNS)."),
		el("div", null, `Current brain: ${brainLabel}`),
	);
	const btns = el("div", "dlg-btns");
	const ok = el("button", null, "OK");
	btns.append(ok);
	dlg.append(body, btns);
	back.append(dlg);
	const close = () => { back.remove(); document.removeEventListener("keydown", onKey, true); };
	const onKey = (e) => { if (e.key === "Escape" || e.key === "Enter") { e.stopImmediatePropagation(); e.preventDefault(); close(); } };
	document.addEventListener("keydown", onKey, true);
	ok.addEventListener("click", close);
	back.addEventListener("pointerdown", (e) => e.stopPropagation());
	host.append(back);
	ok.focus();
}

// bar: the <nav> to fill. api: { command(cmd, arg), flags() -> int, brains: [{key, label}],
// brainKey() -> string, selectBrain(key), brainLabel() -> string, manual() -> bool, toggleManual(),
// dialogHost: element }.
export function createMenuBar(bar, api) {
	const style = el("style");
	style.textContent = STYLE;
	document.head.appendChild(style);
	bar.classList.add("xp-menubar");
	bar.setAttribute("role", "menubar");

	const flag = (f) => (api.flags() & f) !== 0;
	const players = () => (api.flags() >> 8) & 7;
	const cmd = (c, a = 0) => () => api.command(c, a);

	const menus = [
		["Game", [
			{ label: "New Game", key: "F2", run: cmd(CMD.NEW_GAME) },
			{ label: "Launch Ball", enabled: () => flag(FLAG.LAUNCH), run: cmd(CMD.LAUNCH) },
			{ label: "Pause/Resume Game", key: "F3", run: cmd(CMD.PAUSE) },
			"sep",
			{ label: "High Scores...", enabled: () => flag(FLAG.SCORES), run: cmd(CMD.HIGH_SCORES) },
			{ label: "Demo", checked: () => flag(FLAG.DEMO), run: cmd(CMD.DEMO) },
		]],
		["Options", [
			{ label: "Select Players", sub: [1, 2, 3, 4].map((n) => ({
				label: n === 1 ? "1 Player" : `${n} Players`, checked: () => players() === n, run: cmd(CMD.PLAYERS, n),
			})) },
			"sep",
			{ label: "Sounds", key: "F5", checked: () => flag(FLAG.SOUNDS), run: cmd(CMD.SOUNDS) },
			{ label: "Music", key: "F6", checked: () => flag(FLAG.MUSIC), run: cmd(CMD.MUSIC) },
			"sep",
			{ label: "Manual Play (Z / slash)", checked: () => api.manual(), run: () => api.toggleManual() },
			"sep",
			{ label: "Player Controls...", key: "F8", run: cmd(CMD.CONTROLS) },
		]],
		["Brain", api.brains.map((b) => ({
			label: b.label, radio: true, checked: () => api.brainKey() === b.key, run: () => api.selectBrain(b.key),
		}))],
		["Help", [
			{ label: "About Drosophila Cadet...", run: () => showAbout(api.dialogHost, api.brainLabel()) },
		]],
	];

	let openTop = null; // { top, drop } while a menu is open
	const refreshers = [];
	const closeAll = () => {
		if (openTop) openTop.top.classList.remove("open");
		openTop = null;
		bar.querySelectorAll(".sub-open").forEach((n) => n.classList.remove("sub-open"));
		bar.querySelectorAll(".xp-dropdown.open").forEach((n) => n.classList.remove("open"));
	};
	const afterAction = () => {
		closeAll();
		if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur();
	};

	function buildDrop(items) {
		const drop = el("div", "xp-dropdown");
		drop.setAttribute("role", "menu");
		for (const item of items) {
			if (item === "sep") { drop.append(el("div", "msep")); continue; }
			const row = el("div", "mi");
			const check = el("span", "mi-check", "");
			row.append(check, el("span", "mi-label", item.label));
			if (item.key) row.append(el("span", "mi-key", item.key));
			row.setAttribute("role", item.radio ? "menuitemradio" : (item.checked ? "menuitemcheckbox" : "menuitem"));
			let subDrop = null;
			if (item.sub) {
				row.append(el("span", "mi-arrow", "▸"));
				subDrop = buildDrop(item.sub);
				row.append(subDrop);
				row.addEventListener("pointerenter", () => { row.classList.add("sub-open"); subDrop.classList.add("open"); });
				row.addEventListener("pointerleave", () => { row.classList.remove("sub-open"); subDrop.classList.remove("open"); });
			}
			refreshers.push(() => {
				const on = item.enabled ? item.enabled() : true;
				row.classList.toggle("disabled", !on);
				row.setAttribute("aria-disabled", String(!on));
				const checked = item.checked ? item.checked() : false;
				check.textContent = checked ? (item.radio ? "●" : "✓") : "";
				if (item.checked) row.setAttribute("aria-checked", String(checked));
			});
			row.addEventListener("click", (e) => {
				e.stopPropagation();
				if (item.sub || row.classList.contains("disabled")) return;
				afterAction();
				item.run();
			});
			drop.append(row);
		}
		return drop;
	}

	function open(entry) {
		closeAll();
		refreshers.forEach((r) => r());
		entry.top.classList.add("open");
		entry.drop.classList.add("open");
		openTop = entry;
	}

	menus.forEach(([title, items]) => {
		const wrap = el("div");
		wrap.style.position = "relative";
		const top = el("div", "mb-top", title);
		top.setAttribute("role", "menuitem");
		const drop = buildDrop(items);
		drop.style.left = "0";
		wrap.append(top, drop);
		bar.append(wrap);
		const entry = { top, drop };
		top.addEventListener("click", (e) => { e.stopPropagation(); if (openTop === entry) closeAll(); else open(entry); });
		top.addEventListener("pointerenter", () => { if (openTop && openTop !== entry) open(entry); });
	});

	// Menus must not take keyboard focus away from the game: keep mouse presses from focusing anything.
	bar.addEventListener("pointerdown", (e) => e.preventDefault());
	document.addEventListener("pointerdown", (e) => { if (openTop && !bar.contains(e.target)) closeAll(); }, true);
	window.addEventListener("blur", closeAll);
	// Capture phase + stopImmediatePropagation so the engine does not also see this Esc (it is bound to Exit).
	window.addEventListener("keydown", (e) => {
		if (e.key === "Escape" && openTop) { e.stopImmediatePropagation(); e.preventDefault(); closeAll(); }
	}, true);
}
