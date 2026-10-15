"""Builds the ROHIT JI Banjo library from the raw recordings in ../raw.

Steps: hum + noise removal -> tone match -> note slicing -> pitch detection ->
tuning to the exact note -> SFZ instruments (Tremolo, Tremolo Short, Phrases) + demo audio.
The SFZ files are converted to Kontakt (.nki) and Decent Sampler by ConvertWithMoss in CI.
"""
import os, sys, json, glob, shutil
import numpy as np, scipy.signal as ss
sys.path.insert(0, os.path.dirname(__file__))
from banjo_lib import *

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW = os.path.join(ROOT, 'raw')
OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'build'))
LIB = 'ROHIT JI Banjo'
LOW_KEY, HIGH_KEY = 60, 84            # C4 .. C6 - the keyboard starts on C (the recorded notes sit around A4-F5)
SR = 44100

def fade(x, sr, fin=0.002, fout=0.08):
    x = x.copy(); a = int(fin * sr); b = int(fout * sr)
    if a: x[:a] *= np.linspace(0, 1, a)
    if b and len(x) > b: x[-b:] *= np.cos(np.linspace(0, np.pi / 2, b)) ** 2
    return x

def tone_eq(curve):
    """Linear-phase FIR from a {freq: dB} curve (reference match, already limited)."""
    pts = sorted((float(k), float(v)) for k, v in curve.items() if not k.startswith('_'))
    f = np.array([p[0] for p in pts]); g = np.array([p[1] for p in pts])
    grid = np.linspace(0, SR / 2, 1025)
    gains = 10 ** (np.interp(grid, f, g, left=g[0], right=g[-1]) / 20)
    return ss.firwin2(1023, grid / (SR / 2), gains)

def make_loop(x, xfade=0.12):
    """Sustain loop over the tremolo body with a baked-in cross-fade, so a held key keeps playing."""
    n = len(x); X = int(xfade * SR)
    le = n - int(0.06 * SR)                       # before the end fade
    ls = max(int(0.30 * n), int(0.25 * SR), X + 1)
    if le - ls < int(0.3 * SR): ls = max(X + 1, le - int(0.3 * SR))
    # move the loop end to the zero crossing (rising) closest to the planned point
    zc = np.where((x[ls:le - 1] <= 0) & (x[ls + 1:le] > 0))[0]
    if len(zc): le = ls + 1 + int(zc[-1])
    zs = np.where((x[ls - X:ls + 2000] <= 0) & (x[ls - X + 1:ls + 2001] > 0))[0]
    if len(zs): ls = ls - X + 1 + int(zs[np.argmin(np.abs(zs - X))])
    y = x.copy()
    w = np.sin(np.linspace(0, np.pi / 2, X)) ** 2
    y[le - X:le] = x[le - X:le] * (1 - w) + x[ls - X:ls] * w
    return y, (ls, le - 1)

def main():
    if os.path.isdir(OUT): shutil.rmtree(OUT)
    smp_dir = os.path.join(OUT, 'SFZ', 'Samples'); os.makedirs(smp_dir)
    eq_curve = json.load(open(os.path.join(ROOT, 'tools', 'ref_eq.json')))
    fir = tone_eq(eq_curve)
    report = []
    notes = {}          # midi -> list of candidates
    segments = []
    cleaned = []
    phrases = []

    for path in sorted(glob.glob(os.path.join(RAW, '*.wav'))):
        name = os.path.splitext(os.path.basename(path))[0]
        sr, x = load(path)
        assert sr == SR, f'{name}: expected 44.1 kHz'
        noise_before = np.percentile(frame_db(x), 5)
        hb = hum_frequency(x, sr)
        y = remove_hum(x, sr, hb)
        y0 = denoise(y, sr)                      # cleaned, used for pitch detection
        y = ss.filtfilt(fir, [1.0], y0)          # cleaned + tone matched, used for the samples
        on, db, floor = onsets(y, sr)
        report.append(f'{name}: hum {hb:.2f} Hz, noise floor {noise_before:.1f} dB -> {floor:.1f} dB, {len(on)} attacks')

        # whole phrase (for the Phrases instrument): from first attack to where it dies away
        if on:
            a = max(0, on[0] - int(0.01 * sr))
            tail = np.where(db > floor + 6)[0]
            b = min(len(y), (tail[-1] + 4) * 512 + 2048) if len(tail) else len(y)
            phrases.append((name, fade(y[a:b], sr, 0.002, 0.25)))

        cleaned.append((name, y0, y, floor))

    # ---- melody note every 125 ms, measured above the drone strings ----------------------
    HOP, WIN = 0.125, 0.25
    frames = {}
    allc = []
    for name, y0, y, floor in cleaned:
        fr = []
        for t in np.arange(0, len(y0) / SR - WIN, HOP):
            seg = y0[int(t * SR):int((t + WIN) * SR)]
            lv = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)
            if lv < floor + 15: fr.append(None); continue
            f, S = spectrum(seg, SR, 1 << 14); c = chroma(f, S); fr.append(c); allc.append(c)
        frames[name] = fr
    baseline = np.median(allc, axis=0)
    report.append('Drone strings (always ringing): ' + ', '.join(NAMES[i] for i in np.argsort(-baseline)[:2]))

    for name, y0, y, floor in cleaned:
        lab = []
        for c in frames[name]:
            if c is None: lab.append(-1); continue
            ex = c - baseline; i = int(np.argmax(ex)); lab.append(i if ex[i] >= 0.25 else -2)
        # runs of one note (one odd frame inside a run is tolerated)
        i = 0
        while i < len(lab):
            if lab[i] < 0: i += 1; continue
            pc, j, odd = lab[i], i + 1, 0
            while j < len(lab) and (lab[j] == pc or (odd == 0 and j + 1 < len(lab) and lab[j + 1] == pc and lab[j] != -1)):
                if lab[j] != pc: odd += 1
                j += 1
            n_frames = j - i
            if n_frames >= 6:
                t0, t1 = i * HOP, j * HOP + 0.03
                s0 = int(t0 * SR)
                # exact attack: steepest rise within 80 ms around the start of the run
                env = np.abs(y0[max(0, s0 - int(0.08 * SR)):s0 + int(0.08 * SR)])
                if len(env) > 64:
                    sm = np.convolve(env, np.ones(64) / 64, 'same')
                    s0 = max(0, s0 - int(0.08 * SR)) + int(np.argmax(np.diff(sm))) - int(0.004 * SR)
                # skip the ring of the previous note / grace notes: start where this note leads
                for shift in range(0, int(0.35 * SR), int(0.02 * SR)):
                    fw, Sw = spectrum(y0[s0 + shift:s0 + shift + int(0.15 * SR)], SR, 1 << 14)
                    exw = chroma(fw, Sw) - baseline
                    if int(np.argmax(exw)) == pc: break
                if shift:
                    a0 = s0 + shift - int(0.04 * SR)
                    env = np.convolve(np.abs(y0[a0:a0 + int(0.08 * SR)]), np.ones(64) / 64, 'same')
                    s0 = a0 + int(np.argmax(np.diff(env))) - int(0.004 * SR)
                s1 = min(len(y0), int(t1 * SR), s0 + int(6.0 * SR))
                f, S = spectrum(y0[s0 + int(0.03 * SR):min(s1, s0 + int(1.2 * SR))], SR)
                n, cents, strength, why = melody_note(f, S, chroma(f, S), baseline)
                if n is None or n % 12 != pc or abs(cents) > 30:
                    report.append(f'  skipped {name} @ {t0:.2f}s: {why if n is None else "unclear"}')
                else:
                    L = (s1 - s0) / SR
                    notes.setdefault(n, []).append(dict(src=name, t=t0, length=L, cents=cents, conf=strength,
                                                         score=min(L, 3.0) * (0.5 + strength), audio=y[s0:s1]))
            i = j

    # ---- choose, tune and normalise the note samples ------------------------------------
    chosen = {}
    for n in sorted(notes):
        cands = sorted(notes[n], key=lambda c: -c['score'])[:3]
        outs = []
        for i, c in enumerate(cands):
            ratio = 2 ** (c['cents'] / 1200.0)                 # play faster/slower to land on the note
            seg = ss.resample(c['audio'], int(round(len(c['audio']) * ratio)))
            seg = fade(seg, SR, 0.002, 0.04)
            loop = make_loop(seg)
            rms = np.sqrt(np.mean(seg[:int(0.5 * SR)] ** 2)) + 1e-12
            seg *= 10 ** (-18 / 20) / rms
            pk = np.max(np.abs(seg))
            if pk > 10 ** (-1 / 20): seg *= 10 ** (-1 / 20) / pk
            fn = f'Banjo_Trem_{midi_name(n).replace("#", "s")}_rr{i + 1}.wav'
            seg, (ls, le) = loop
            save24(os.path.join(smp_dir, fn), SR, seg)
            outs.append((fn, ls, le))
            report.append(f'  {midi_name(n):4s} rr{i + 1}: {c["src"]} @ {c["t"]:.2f}s, {c["length"]:.1f}s, tuned {-c["cents"]:+.0f} cents')
        chosen[n] = outs

    phrase_files = []
    for i, (name, a) in enumerate(phrases):
        rms = np.sqrt(np.mean(a[np.abs(a) > 1e-4] ** 2)) + 1e-12
        a = a * 10 ** (-18 / 20) / rms
        pk = np.max(np.abs(a))
        if pk > 10 ** (-1 / 20): a *= 10 ** (-1 / 20) / pk
        fn = f'Banjo_Phrase_{i + 1:02d}_{name}.wav'
        save24(os.path.join(smp_dir, fn), SR, a)
        phrase_files.append(fn)

    # ---- key ranges: every key from C3 to C6 gets the nearest sampled note --------------
    roots = sorted(chosen)
    ranges = {}
    for i, r in enumerate(roots):
        lo = LOW_KEY if i == 0 else (roots[i - 1] + r) // 2 + 1
        hi = HIGH_KEY if i == len(roots) - 1 else (r + roots[i + 1]) // 2
        ranges[r] = (lo, hi)

    white = [0, 2, 4, 5, 7, 9, 11]
    phrase_keys = [48 + 12 * (i // 7) + white[i % 7] for i in range(len(phrase_files))]

    def write_sfz(fname, header, body):
        with open(os.path.join(OUT, 'SFZ', fname), 'w') as f:
            f.write(f'// {LIB} - {header}\n// ROHIT JI | Beat Ki Pathshala\n\n<control>\ndefault_path=Samples/\n\n' + body)

    def tremolo(short, rr):
        out = '<global>\namp_veltrack=70\nampeg_attack=0.002\n'
        out += ('ampeg_decay=0.9\nampeg_sustain=0\nampeg_release=0.12\n\n' if short else 'ampeg_release=0.45\n\n')
        for r in roots:
            lo, hi = ranges[r]
            files = chosen[r] if rr else chosen[r][:1]
            out += f'<group>\nlokey={lo} hikey={hi} pitch_keycenter={r}' + (f' seq_length={len(files)}' if len(files) > 1 else '') + '\n'
            for j, (fn, ls, le) in enumerate(files):
                lp = '' if short else f' loop_mode=loop_sustain loop_start={ls} loop_end={le}'
                out += f'<region> sample={fn}{lp}' + (f' seq_position={j + 1}' if len(files) > 1 else '') + '\n'
            out += '\n'
        return out

    def phrase_body():
        out = '<global>\namp_veltrack=40\nloop_mode=one_shot\npitch_keytrack=0\nampeg_release=0.3\n\n<group>\n'
        for k, fn in zip(phrase_keys, phrase_files):
            out += f'<region> sample={fn} key={k}\n'
        return out

    instruments = [
        (f'{LIB} - Tremolo.sfz', 'Tremolo sustain, round robin', tremolo(False, True)),
        (f'{LIB} - Tremolo Short.sfz', 'Tremolo short, round robin', tremolo(True, True)),
        (f'{LIB} - Phrases.sfz', 'Phrases on the white keys from C3', phrase_body()),
    ]
    for fn, h, body in instruments: write_sfz(fn, h, body)

    # Kontakt 1 NKI cannot do round robin: same instruments, one sample per zone
    kdir = os.path.join(OUT, 'for_kontakt'); os.makedirs(kdir)
    for fn, h, body in [(f'{LIB} - Tremolo.sfz', 'Tremolo sustain', tremolo(False, False)),
                        (f'{LIB} - Tremolo Short.sfz', 'Tremolo short', tremolo(True, False)),
                        (f'{LIB} - Phrases.sfz', 'Phrases', phrase_body())]:
        with open(os.path.join(kdir, fn), 'w') as f:
            f.write(f'// {LIB} - {h}\n\n<control>\ndefault_path=../SFZ/Samples/\n\n' + body)

    # ---- demo: C major scale and a short tune, rendered with the same mapping ----------
    def render(seq, bpm=100):
        beat = 60 / bpm; total = int((sum(d for _, d in seq) * beat + 3) * SR); outb = np.zeros(total)
        t = 0.0; rrc = {}
        for key, dur in seq:
            if key is not None:
                r = min(roots, key=lambda q: (abs(q - key), q))
                for rr in roots:
                    lo, hi = ranges[rr]
                    if lo <= key <= hi: r = rr
                i = rrc.get(r, 0); rrc[r] = i + 1
                fn = chosen[r][i % len(chosen[r])][0]
                _, a = load(os.path.join(smp_dir, fn))
                a = ss.resample(a, int(len(a) / 2 ** ((key - r) / 12)))
                n = min(len(a), int((dur * beat + 0.45) * SR))
                env = np.ones(n); rel = int(0.45 * SR)
                if n > int(dur * beat * SR): env[int(dur * beat * SR):] = np.linspace(1, 0, n - int(dur * beat * SR))
                s = int(t * SR); outb[s:s + n] += a[:n] * env[:len(outb[s:s + n])]
            t += dur * beat
        return outb / (np.max(np.abs(outb)) + 1e-9) * 0.89
    demo_dir = os.path.join(OUT, 'Demo'); os.makedirs(demo_dir)
    scale = [(k, 1) for k in (60, 62, 64, 65, 67, 69, 71, 72)] + [(k, 1) for k in (71, 69, 67, 65, 64, 62)] + [(60, 3)]
    save24(os.path.join(demo_dir, 'Demo - C major scale.wav'), SR, render(scale))
    tune = [(64, .5), (67, .5), (69, 1), (69, .5), (67, .5), (64, 1), (62, .5), (64, .5), (67, 1), (64, 2),
            (69, .5), (72, .5), (71, 1), (69, .5), (67, .5), (64, 1), (62, .5), (60, .5), (62, 1), (60, 3)]
    save24(os.path.join(demo_dir, 'Demo - short tune.wav'), SR, render(tune, 92))
    check = []
    for r in roots: check += [(r, 1.5), (None, 0.5)]
    save24(os.path.join(demo_dir, 'Check - every sampled note low to high.wav'), SR, render(check, 90))

    report.append('')
    report.append('Key map (key range -> sampled note):')
    for r in roots:
        lo, hi = ranges[r]; report.append(f'  {midi_name(lo)}-{midi_name(hi)} -> {midi_name(r)} ({len(chosen[r])} round robin)')
    report.append('Phrases: ' + ', '.join(f'{midi_name(k)} = {fn}' for k, fn in zip(phrase_keys, phrase_files)))
    open(os.path.join(OUT, 'build-report.txt'), 'w').write('\n'.join(report) + '\n')
    print('\n'.join(report))

if __name__ == '__main__':
    main()
