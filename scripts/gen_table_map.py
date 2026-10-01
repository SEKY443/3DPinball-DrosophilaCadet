#!/usr/bin/env python3
"""Regenerates agents/table_map.json: every scoring object on the Space Cadet table with its
stable id, engine name, group, position/bounding box and point values.

Two sources are merged:
  1. The engine itself (PINBALL_DUMP_TABLE_MAP, see src_cpp/state_export.cpp::DumpTableMap):
     stable id, C++ class, and the collision AABB in table coordinates - the same frame as
     StateFrame.ball_x/ball_y (both come from the engine's table-space positions).
  2. The engine source (vendor/SpaceCadetPinball/SpaceCadetPinball/control.cpp): control function
     and score array per object, with file:line citations, parsed statically.

The ids are control::score_components index + 1 - fixed by the engine source, not by DEMO.DAT
load order - and are what StateFrame.hit_ids / info["hit_objects"] report.

Usage:
	python scripts/gen_table_map.py [--binary vendor/SpaceCadetPinball/bin/SpaceCadetPinball]
	                                [--out agents/table_map.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

CONTROL_CPP = os.path.join(ROOT, "vendor", "SpaceCadetPinball", "SpaceCadetPinball", "control.cpp")
CONTROL_CPP_REL = "vendor/SpaceCadetPinball/SpaceCadetPinball/control.cpp"

# C++ class -> coarse group used for analysis/rewards.
CLASS_GROUP = {
	"TBumper": "bumper",
	"TRollover": "rollover",
	"TLightRollover": "rollover",
	"TPopupTarget": "target",
	"TSoloTarget": "target",
	"TRamp": "ramp",
	"THole": "hole",
	"TKickout": "kickout",
	"TSink": "sink",
	"TWall": "rebounder",
	"TKickback": "kicker",
	"TGate": "gate",
	"TOneway": "oneway",
	"TFlagSpinner": "spinner",
	"TTripwire": "tripwire",
	"TDrain": "drain",
	"TFlipper": "flipper",
	"TPlunger": "plunger",
	"TBlocker": "blocker",
	"TComponentGroup": "logic",
	"TLight": "logic",
	"TLightGroup": "logic",
}

# Hand-written from reading each handler in control.cpp (see "handler" citation per object): which
# game state changes a hit's payout. Every payout is also multiplied by the table multiplier, and
# control::handler() runs MissionControl() for every hit, so a hit that advances/completes a
# mission is additionally credited with that mission's award (SpecialAddScore, 500k-5M).
VALUE_NOTES = {
	"BumperControl": "pays scores[BmpIndex]; bumper level rises via lane rollovers (Reentry/LaunchLanesRolloverControl)",
	"ReentryLanesRolloverControl": "flat; completing the 3 lanes raises attack-bumper level (future bumper value)",
	"LaunchLanesRolloverControl": "flat; completing the 3 lanes raises launch-bumper level (future bumper value)",
	"FlipperRebounderControl1": "flat", "FlipperRebounderControl2": "flat", "RebounderControl": "flat",
	"DeploymentChuteToEscapeChuteOneWayControl": "pays scores[skill_shot_lights on-count - 1] only when skill-shot lights are on (after a plunger launch), else 0",
	"LaunchRampControl": "base 5000 only when no ramp light is on; with lite54 lit pays ReflexShotScore (SpecialAddScore)",
	"OutLaneRolloverControl": "flat 20000 (outlane = ball about to drain); awards extra ball if lite17/18 lit",
	"ReturnLaneRolloverControl": "5000, or 25000 when its lane light (lite27/lite28) is on",
	"BonusLaneRolloverControl": "10000, or the accumulated BonusScore when lite16 is on",
	"FuelRollover1Control": "flat; refuels bargraph", "FuelRollover2Control": "flat; refuels bargraph",
	"FuelRollover3Control": "flat; refuels bargraph", "FuelRollover4Control": "flat; refuels bargraph",
	"FuelRollover5Control": "flat; refuels bargraph", "FuelRollover6Control": "flat; refuels bargraph",
	"FlagControl": "scores[lite20 on] (500 or 2500) per spinner collision",
	"HyperspaceKickOutControl": "scores[] indexed by hyperspace light level; can pay JackpotScore (SpecialAddScore)",
	"WormHoleControl": "2500 base; 5000/7500 or 10000/50000 + replay when wormhole lights match",
	"BoosterTargetControl": "500 per target; completing the 3-bank pays 5000 and awards flag/jackpot/bonus/bonus-hold",
	"MedalTargetControl": "1500 per target; completing the 3-bank pays 10000/50000 or an extra ball by medal level",
	"MultiplierTargetControl": "500 per target; completing the 3-bank pays 1500 and raises the score multiplier (all future points x2/x3/x5/x10)",
	"FuelSpotTargetControl": "flat 750; refuels", "MissionSpotTargetControl": "flat 1000; advances mission selection",
	"LeftHazardSpotTargetControl": "flat 750; mission progress", "RightHazardSpotTargetControl": "flat 750; mission progress",
	"WormHoleDestinationControl": "flat 750; selects wormhole destination",
	"SpaceWarpRolloverControl": "handler never calls AddScore (the 10000 array is unused); lights lite27/lite28 so the return lanes then pay 25000 instead of 5000 - delayed value",
	"BlackHoleKickoutControl": "flat 20000", "GravityWellKickoutControl": "flat 50000",
	"BallDrainControl": "no score array; drain-time awards (bonus countdown etc.) happen here or on timers",
}

_ARRAY_RE = re.compile(r"^int\s+(\w+)\[\d*\]\s*=\s*\{([^}]*)\};")
_TAG_RE = re.compile(r'^component_tag<(\w+)>\s+(\w+)\s*=\s*\{"([^"]+)"\};')
_INFO_RE = re.compile(r"component_info\{(\w+),\s*\{(\w+),\s*(\d+),\s*(\w+)\}\}")
_FUNC_RE = re.compile(r"^void control::(\w+)\(MessageCode code, TPinballComponent\* caller\)")


def parse_control_cpp(path: str = CONTROL_CPP) -> list[dict]:
	"""Returns one dict per control::score_components entry, in declaration (= id) order."""
	with open(path, encoding="utf-8") as f:
		lines = f.read().splitlines()
	arrays, tags, funcs, infos = {}, {}, {}, []
	in_score_components = False
	for lineno, line in enumerate(lines, start=1):
		s = line.strip()
		m = _ARRAY_RE.match(s)
		if m:
			arrays[m.group(1)] = ([int(v) for v in m.group(2).split(",") if v.strip()], lineno)
			continue
		m = _TAG_RE.match(s)
		if m:
			tags[m.group(2)] = (m.group(3), m.group(1), lineno)
			continue
		m = _FUNC_RE.match(s)
		if m:
			funcs[m.group(1)] = lineno
			continue
		if s.startswith("component_info control::score_components"):
			in_score_components = True
			continue
		if in_score_components:
			if s.startswith("};"):
				in_score_components = False
				continue
			m = _INFO_RE.search(s)
			if m:
				infos.append((m.group(1), m.group(2), int(m.group(3)), m.group(4), lineno))

	out = []
	for idx, (tag_var, func, count, arr, lineno) in enumerate(infos):
		name, tag_class, tag_line = tags[tag_var]
		values, arr_line = arrays.get(arr, ([], None))
		if count and len(values) < count:
			raise ValueError(f"{arr}: {count} scores declared but {len(values)} parsed")
		out.append({
			"id": idx + 1,
			"name": name,
			"tag_class": tag_class,
			"control_func": func,
			"scores": values[:count],
			"source": {
				"score_components": f"{CONTROL_CPP_REL}:{lineno}",
				"tag": f"{CONTROL_CPP_REL}:{tag_line}",
				"scores": f"{CONTROL_CPP_REL}:{arr_line}" if arr_line else None,
				"handler": f"{CONTROL_CPP_REL}:{funcs[func]}" if func in funcs else None,
			},
		})
	return out


def dump_engine_map(binary: str) -> dict:
	"""Launches the engine once with PINBALL_DUMP_TABLE_MAP set and returns the parsed dump."""
	from env_python.pinball_env import PinballEnv

	fd, path = tempfile.mkstemp(suffix=".json", prefix="table_map_dump_")
	os.close(fd)
	os.environ["PINBALL_DUMP_TABLE_MAP"] = path
	env = PinballEnv(binary_path=binary, headless=True)
	try:
		env.reset(seed=0)  # connecting proves Init() (and therefore the dump) has run
	finally:
		env.close()
		del os.environ["PINBALL_DUMP_TABLE_MAP"]
	with open(path, encoding="utf-8") as f:
		data = json.load(f)
	os.remove(path)
	return data


def build_table_map(binary: str) -> dict:
	source = parse_control_cpp()
	engine = {c["id"]: c for c in dump_engine_map(binary)["components"]}
	objects = []
	for src in source:
		eng = engine[src["id"]]
		if eng["name"] != src["name"]:
			raise ValueError(f"id {src['id']}: engine name {eng['name']!r} != source name {src['name']!r}")
		if eng["scores"] != src["scores"]:
			raise ValueError(f"id {src['id']}: engine scores {eng['scores']} != source scores {src['scores']}")
		cls = eng["class"] or src["tag_class"]
		aabb = eng["aabb"]
		center = None
		if aabb is not None:
			center = [round((aabb["x_min"] + aabb["x_max"]) / 2, 4), round((aabb["y_min"] + aabb["y_max"]) / 2, 4)]
		objects.append({
			"id": src["id"],
			"name": src["name"],
			"class": cls,
			"group": CLASS_GROUP.get(cls, "other"),
			"linked": eng["linked"],
			"center": center,
			"aabb": aabb,
			"scores": src["scores"],
			"base_points": src["scores"][0] if src["scores"] else 0,
			"max_listed_points": max(src["scores"]) if src["scores"] else 0,
			"control_func": src["control_func"],
			"value_note": VALUE_NOTES.get(src["control_func"], ""),
			"state_dependent": len(set(src["scores"])) > 1 or src["control_func"] in (
				"LaunchRampControl", "BonusLaneRolloverControl", "DeploymentChuteToEscapeChuteOneWayControl"),
			"source": src["source"],
		})
	return {
		"description": (
			"Scoring objects of 3D Pinball Space Cadet (DEMO.DAT). id = control::score_components index + 1, "
			"as reported in StateFrame.hit_ids / info['hit_objects']. center/aabb are in table coordinates, the "
			"same frame as the ball observation (ball_x, ball_y; y~14 drain, y~-12 top). aabb is the engine's "
			"collision-edge bounding box (edges are already offset by the ball radius, so it bounds the BALL "
			"CENTRE at contact). scores are the object's score array from control.cpp; which entry is paid "
			"depends on handler state (e.g. bumper light level, mission progress) and every AddScore is "
			"multiplied by the table's score multiplier (1/2/3/5/10, TPinballTable.cpp score_multipliers). "
			"base_points = scores[0] is the typical payout; value_note says what state changes it. "
			"control::handler() also runs MissionControl() on every hit, so a hit that completes a mission "
			"is credited at runtime with that award (500k-5M) - not listed here. Use runtime hit_points for "
			"realised value and base_points only as a prior."
		),
		"generated_by": "scripts/gen_table_map.py",
		"objects": objects,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--binary", default=os.path.join(ROOT, "vendor", "SpaceCadetPinball", "bin", "SpaceCadetPinball"))
	parser.add_argument("--out", default=os.path.join(ROOT, "agents", "table_map.json"))
	args = parser.parse_args()
	table_map = build_table_map(args.binary)
	with open(args.out, "w", encoding="utf-8") as f:
		json.dump(table_map, f, indent=1)
		f.write("\n")
	n_pos = sum(o["center"] is not None for o in table_map["objects"])
	print(f"wrote {args.out}: {len(table_map['objects'])} objects ({n_pos} with positions)")


if __name__ == "__main__":
	main()
