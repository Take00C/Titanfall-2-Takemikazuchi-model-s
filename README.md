# Takemikazuchi skins for Ronin (Titanfall 2 / Northstar)

Eight Type-00 Takemikazuchi model's from **Project MIKHAIL**, as replacement models for **Ronin** and **Ronin Prime**.

## Download

Get the zips from the [Releases page](../../releases/latest):

| Skin | Zip |
|---|---|
| Type-00R (Blue) | `TSFRonin.Takemikazuchi.zip` |
| Type-00R (Purple) | `TSFRonin.Takemikazuchi.R_Purple.zip` |
| Type-00F (UN) | `TSFRonin.Takemikazuchi.F_Blue.zip` |
| Type-00F (Red) | `TSFRonin.Takemikazuchi.F_Red.zip` |
| Type-00F (Yellow) | `TSFRonin.Takemikazuchi.F_Yellow.zip` |
| Type-00A (white) | `TSFRonin.Takemikazuchi.A.zip` |
| Type-00C | `TSFRonin.Takemikazuchi.C.zip` |
| Type-00C (UN) | `TSFRonin.Takemikazuchi.C_Blue.zip` |

`Takemikazuchi-Ronin-All-Skins.zip` holds all eight. You do not need Project MIKHAIL to use them.

## Install

1. Unzip so the skin's folder sits in `Titanfall2\R2Northstar\mods\`.
2. Start Northstar and tick the skin in the **Mods** menu.

All the skins replace Ronin, so **only one can be on at a time**. Untick the old one before ticking another, then
restart the game. Reloading mods from the menu after switching skins can crash the game.

## Report a problem

Use the [Issues tab](../../issues/new/choose). Say which skin, what looks wrong, and attach a screenshot.

## Building from source

The scripts and data that make the skins are in this repository (no models or archive key). Building needs both games:
`python build.py` (Python 3.11+). `python build.py --list` shows the stages. Paths are set in `tsfronin/config.py`.
