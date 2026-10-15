# ROHIT JI Banjo - Kontakt library

Indian banjo (bulbul tarang style) sample library made from ROHIT JI's own recordings.
GitHub builds it automatically: clean -> tune -> map -> **Kontakt (.nki)**, **Decent Sampler** and **SFZ**.

## Download
Go to **[Releases](../../releases/latest)** and download **ROHIT-JI-Banjo-Library.zip**. Unzip it and read *HOW TO USE.txt*.

| Instrument (separate files, load the one you need) | What it is |
|---|---|
| **ROHIT JI Banjo - Tremolo** | Sustained tremolo notes, C4 to C6, in tune (A = 440 Hz) |
| **ROHIT JI Banjo - Tremolo Short** | The same notes, short and plucky |
| **ROHIT JI Banjo - Phrases** | The full recorded phrases on the white keys from C3 |

## How it is built
* `raw/` - the original recordings (24-bit WAV).
* `tools/build.py` - removes the 50 Hz electrical hum, rumble and hiss, applies a gentle tone match
  (`tools/ref_eq.json`, measured from reference banjo videos), finds every clean single note above the
  drone strings, tunes it to the exact pitch, adds a smooth sustain loop and writes the SFZ instruments
  and demo audio.
* `.github/workflows/build-library.yml` - runs the build and converts the SFZ instruments to Kontakt
  and Decent Sampler with [ConvertWithMoss](https://github.com/git-moss/ConvertWithMoss), then publishes the release.

To extend the range, add more WAV recordings (one long tremolo note per file) to `raw/` and push.
