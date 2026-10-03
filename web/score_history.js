// "Score History" window: a Windows-ListView-style table of every finished game (time, player,
// brain, score, high-score marker) plus the all-time top 5. Persisted in localStorage when
// available; every storage access is wrapped so the page works without it. All text is set with
// textContent - nothing here builds HTML from strings.
import { xpScrollbar } from "./xp_scrollbar.js?v=3";
import { el, storageGet, storageSet } from "./dom.js?v=1";

const STORAGE_KEY = "flyPinball.scoreHistory.v1";
const MAX_ENTRIES = 200;
const TOP_COUNT = 5;
// House record: the best game the giant-fiber body played on the author's machine. It always sits in the Top 5
// (also for first-time visitors) until five better games push it out, and it is listed as the oldest row of the game
// list (fixed time label, never stored).
export const HOUSE_RECORD = { name: "xXSpaceFlyXx", score: 6092750, brain: "Giant fiber", timeLabel: "23:38" };

// A game "made the Top 5" when, at the moment it finished, fewer than TOP_COUNT earlier scores existed (house record
// included) or its score beat the current 5th best of them (ties with the 5th do not qualify).
function qualifiesForTop(score, earlierScores) {
	if (earlierScores.length < TOP_COUNT) return true;
	const sorted = [...earlierScores].sort((a, b) => b - a);
	return score > sorted[TOP_COUNT - 1];
}

// Recompute the marker for stored rows (newest first) chronologically, oldest to newest.
function recomputeMarkers(entries) {
	const seen = [HOUSE_RECORD.score];
	for (let i = entries.length - 1; i >= 0; i--) {
		entries[i].highScore = qualifiesForTop(entries[i].score, seen);
		seen.push(entries[i].score);
	}
	return entries;
}

function loadEntries() {
	try {
		const raw = storageGet(STORAGE_KEY);
		const parsed = raw ? JSON.parse(raw) : [];
		if (!Array.isArray(parsed)) return [];
		return recomputeMarkers(parsed
			.filter((e) => e && typeof e.name === "string" && Number.isFinite(e.score) && Number.isFinite(e.time))
			.map((e) => ({
				time: e.time,
				name: e.name.slice(0, 32),
				brain: typeof e.brain === "string" ? e.brain.slice(0, 24) : "",
				score: e.score,
				highScore: e.highScore === true,
			}))
			.slice(0, MAX_ENTRIES));
	} catch {
		return [];
	}
}

function formatTime(ms) {
	const d = new Date(ms);
	return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

function formatScore(score) {
	return String(score).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function buildTable(headers, className) {
	const table = el("table", `sh-table ${className}`);
	const head = el("thead");
	const row = el("tr");
	for (const [label, cls] of headers) row.appendChild(el("th", cls, label));
	head.appendChild(row);
	table.appendChild(head);
	const body = el("tbody");
	table.appendChild(body);
	return { table, body };
}

function appendRow(body, cells) {
	const row = el("tr");
	for (const [text, cls] of cells) row.appendChild(el("td", cls, text));
	body.appendChild(row);
}

// `container` is the .window-body of #win-stats. Returns { add(entry) } where entry is
// { name, brain, score } (time is stamped here; the HS marker is derived from the stored history).
export function createScoreHistory(container) {
	let entries = loadEntries();

	const top = el("div", "sh-top");
	top.appendChild(el("div", "sh-heading", "Top 5"));
	const topTable = buildTable([["#", "num"], ["Player", ""], ["Score", "num"]], "sh-top-table");
	top.appendChild(topTable.table);

	const listWrap = el("div", "sh-list");
	const listTable = buildTable(
		[["Time", ""], ["Player", ""], ["Brain", ""], ["Score", "num"], ["HS", "mark"]], "sh-list-table");
	listWrap.appendChild(listTable.table);

	// XP list view: the native scrollbar is hidden and a live XP Luna one is drawn next to it (xp_scrollbar.js).
	const listFrame = el("div", "sh-list-wrap");
	listFrame.append(listWrap, xpScrollbar("v", { target: listWrap }));
	container.replaceChildren(top, listFrame);

	function render() {
		topTable.body.replaceChildren();
		const best = [...entries, HOUSE_RECORD].sort((a, b) => b.score - a.score).slice(0, TOP_COUNT);
		for (let i = 0; i < TOP_COUNT; i++) {
			const e = best[i];
			appendRow(topTable.body, [[String(i + 1), "num"], [e ? e.name : "", ""], [e ? formatScore(e.score) : "", "num"]]);
		}
		listTable.body.replaceChildren();
		for (const e of entries) {
			appendRow(listTable.body, [
				[formatTime(e.time), ""], [e.name, ""], [e.brain, ""], [formatScore(e.score), "num"],
				[e.highScore ? "★" : "", "mark"],
			]);
		}
		appendRow(listTable.body, [
			[HOUSE_RECORD.timeLabel, ""], [HOUSE_RECORD.name, ""], [HOUSE_RECORD.brain, ""],
			[formatScore(HOUSE_RECORD.score), "num"], ["★", "mark"],
		]);
	}

	render();
	return {
		add({ name, brain, score }) {
			const highScore = qualifiesForTop(score, [HOUSE_RECORD.score, ...entries.map((e) => e.score)]);
			entries.unshift({ time: Date.now(), name: String(name), brain: String(brain), score, highScore });
			entries = entries.slice(0, MAX_ENTRIES);
			storageSet(STORAGE_KEY, JSON.stringify(entries));
			listWrap.scrollTop = 0; // newest first
			render();
		},
	};
}
