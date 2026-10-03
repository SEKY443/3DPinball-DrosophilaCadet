// Early-2000s / Windows-XP-era screen names with a fruit-fly theme (AOL/MSN style: xX_..._Xx,
// leetspeak, ~*sparkles*~, trailing years). Used as the high-score player name (the engine's name
// field holds 31 characters; every entry here stays within MAX_NAME_LENGTH).

export const MAX_NAME_LENGTH = 16;

export const FLY_NAMES = [
	"xX_Drosophila_Xx", "FruitFly2001", "BuzzMelanogaster", "~*GiantFiber*~", "LC4_Sn1p3r",
	"DNp01_Jumper", "Wingz_0f_F1re", "sk8r_fly_99", "Mr.Proboscis", "HaltereHero",
	"OmmatidiaKid", "BananaBreath", "~CompoundEyez~", "L00m1ng_Thr34t", "MaggotMaster3k",
	"x_pupa_x", "Th0raxRox", "Fly_Shady", "N3ctarN1nja", "Larva_Lad_2002",
	"DipteraDude", "BuzzKill2k", "xXSpaceFlyXx", "AntennaAngel", "LPLC2_Lover",
	"RottenBanana88", "WinXP_Wingz", "SwatDodger", "ZzZzBuzZ", "Pr0b0sc1s",
	"VinegarVixen", "Wild_Type_W1118", "~*Ommatidia*~", "xX_LarvaL0rd_Xx", "FlyPaper_Escapee",
	"~*FlyGuy2K*~", "Maggot_Mania_01", "SpaceCadet_Fly",
].filter((name) => name.length <= MAX_NAME_LENGTH);

// Random name, never equal to `previous` (so consecutive games get different names).
export function randomFlyName(previous = "", rng = Math.random) {
	let name = previous;
	while (name === previous) {
		name = FLY_NAMES[Math.floor(rng() * FLY_NAMES.length) % FLY_NAMES.length];
	}
	return name;
}
