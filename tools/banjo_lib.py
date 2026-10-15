"""Shared DSP for the Banjo library: loading, hum/noise cleaning, onset slicing, pitch detection."""
import numpy as np, scipy.signal as ss, scipy.io.wavfile as wf, warnings
warnings.filterwarnings('ignore', category=wf.WavFileWarning)

NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def midi_name(m): return f"{NAMES[m % 12]}{m // 12 - 1}"
def hz_to_midi(f): return 69 + 12 * np.log2(f / 440.0)

def load(path):
    sr, x = wf.read(path)
    if x.dtype.kind == 'i':
        x = x.astype(np.float64) / float(2 ** (8 * x.dtype.itemsize - 1))
    else:
        x = x.astype(np.float64)
    if x.ndim > 1: x = x.mean(axis=1)
    return sr, x

def save24(path, sr, x):
    x = np.clip(x, -1, 1 - 1e-9)
    v = np.round(x * (2 ** 23 - 1)).astype(np.int32)
    b = np.zeros((len(v), 3), np.uint8)
    b[:, 0] = v & 255; b[:, 1] = (v >> 8) & 255; b[:, 2] = (v >> 16) & 255
    data = b.tobytes()
    import struct
    with open(path, 'wb') as f:
        f.write(b'RIFF' + struct.pack('<I', 36 + len(data)) + b'WAVE')
        f.write(b'fmt ' + struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * 3, 3, 24))
        f.write(b'data' + struct.pack('<I', len(data)) + data)

def frame_db(x, hop=512, win=2048):
    n = max(1, (len(x) - win) // hop)
    fr = np.array([np.sqrt(np.mean(x[i * hop:i * hop + win] ** 2)) for i in range(n)])
    return 20 * np.log10(fr + 1e-12)

def hum_frequency(x, sr):
    """Mains hum base frequency (around 50 or 60 Hz), measured from the strongest low harmonic."""
    f, P = ss.welch(x, sr, nperseg=1 << 16)
    best, bf = -1, 50.0
    for base in list(np.arange(49.0, 51.0, 0.01)) + list(np.arange(59.0, 61.0, 0.01)):
        s = sum(np.interp(base * k, f, P) for k in (1, 3, 5))
        if s > best: best, bf = s, base
    return bf

def remove_hum(x, sr, base):
    b, a = ss.butter(4, 190 / (sr / 2), 'high')          # lowest banjo note is C4 (261 Hz): remove hum + rumble below
    y = ss.filtfilt(b, a, x)
    for k in range(2, 24):                                # harmonics up to ~1.2 kHz
        f0 = base * k
        q = 45 if k % 2 else 30
        bb, aa = ss.iirnotch(f0, q, sr)
        y = ss.filtfilt(bb, aa, y)
    return y

def denoise(x, sr, reduction_db=18.0):
    """Gentle spectral noise reduction with a noise profile from the quietest parts."""
    n, hop = 2048, 512
    f, t, Z = ss.stft(x, sr, nperseg=n, noverlap=n - hop)
    mag = np.abs(Z)
    e = mag.mean(axis=0)
    quiet = e <= np.percentile(e, 12)
    noise = np.median(mag[:, quiet], axis=1, keepdims=True) * 1.6
    floor = 10 ** (-reduction_db / 20)
    g = np.clip(1 - (noise / (mag + 1e-12)) ** 2, 0, 1)
    g = np.maximum(np.sqrt(g), floor)
    g = ss.lfilter([0.4], [1, -0.6], g, axis=1)           # smooth over time to avoid "musical noise"
    g = np.maximum(g, floor)
    _, y = ss.istft(Z * g, sr, nperseg=n, noverlap=n - hop)
    return y[:len(x)]

def onsets(x, sr, hop=512):
    db = frame_db(x, hop)
    floor = np.percentile(db, 5)
    out = []
    for i in range(4, len(db)):
        if db[i] > floor + 20 and db[i] - db[i - 4] > 9 and (not out or i - out[-1] > int(0.25 * sr / hop)):
            # move back to where the attack starts
            j = i
            while j > 0 and db[j - 1] < db[j] - 0.5 and db[j - 1] > floor + 6: j -= 1
            out.append(j)
    return [o * hop for o in out], db, floor

def pitch(seg, sr, fmin=180.0, fmax=1100.0):
    """Melody pitch of a tremolo note (with drone strings ringing): harmonic salience search.
    Returns (midi_float, confidence)."""
    N = 1 << 16
    w = seg * np.hanning(len(seg))
    S = np.abs(np.fft.rfft(w, N)); fr = np.fft.rfftfreq(N, 1 / sr)
    S = S / (S.max() + 1e-12)
    def mag(f): return np.interp(f, fr, S) if f < sr / 2 else 0.0
    cands = 440 * 2 ** ((np.arange(hz_to_midi(fmin), hz_to_midi(fmax), 0.05) - 69) / 12)
    sal = []
    for c in cands:
        s = sum(mag(c * k) / k ** 0.6 for k in range(1, 9))
        s -= 0.5 * sum(mag(c * (k + 0.5)) / k ** 0.6 for k in range(1, 6))   # penalise f0/2 errors
        s *= mag(c) ** 0.25                                                   # fundamental must exist
        sal.append(s)
    sal = np.array(sal)
    i = int(np.argmax(sal))
    best = cands[i]
    # refine on the actual fundamental peak
    band = (fr > best * 0.97) & (fr < best * 1.03)
    fpk = fr[band][np.argmax(S[band])] if band.any() else best
    # confidence: best vs best candidate in another pitch class
    m = hz_to_midi(cands)
    other = sal[np.abs(((m - hz_to_midi(best)) + 6) % 12 - 6) > 0.6]
    conf = sal[i] / (other.max() + 1e-12) if len(other) else 9
    return hz_to_midi(fpk), conf

def spectrum(seg, sr, N=1 << 16):
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), N)); f = np.fft.rfftfreq(N, 1 / sr)
    return f, S

def chroma(f, S, lo=180.0, hi=2500.0):
    m = (f > lo) & (f < hi); mid = hz_to_midi(f[m]); w = S[m] ** 2
    c = np.zeros(12)
    for k in range(12):
        d = np.abs(((mid - k) + 6) % 12 - 6); c[k] = np.sum(w * np.exp(-(d / 0.25) ** 2))
    return c / (c.sum() + 1e-12)

def melody_note(f, S, c, baseline, lo_midi=60, hi_midi=83):
    """Melody note of a banjo tremolo note: pitch class with the most energy above the drone
    strings (baseline chroma), octave = strongest partial of that class inside the melody register.
    Returns (midi, cents, strength, reason) or (None, ..., reason)."""
    ex = c - baseline; o = np.argsort(-ex); pc = int(o[0])
    if ex[pc] < 0.35: return None, 0, 0, 'no clear note'
    if ex[pc] - ex[o[1]] < 0.15: return None, 0, 0, 'two notes (chord)'
    best = None; levels = []
    for midi in range(lo_midi, hi_midi + 1):
        if midi % 12 != pc: continue
        fc = 440 * 2 ** ((midi - 69) / 12)
        band = (f > fc * 2 ** (-0.35 / 12)) & (f < fc * 2 ** (0.35 / 12))
        i = np.argmax(S[band]); lvl = 20 * np.log10(S[band][i] / S.max() + 1e-12)
        # parabolic peak interpolation for the exact frequency
        idx = np.where(band)[0][i]
        a, b, cc = np.log(S[idx - 1] + 1e-12), np.log(S[idx] + 1e-12), np.log(S[idx + 1] + 1e-12)
        d = 0.5 * (a - cc) / (a - 2 * b + cc) if (a - 2 * b + cc) != 0 else 0
        fpk = f[idx] + d * (f[1] - f[0])
        levels.append(lvl)
        if best is None or lvl > best[1]: best = (midi, lvl, fpk)
    if best is None or best[1] < -12: return None, 0, 0, 'note too weak'
    others = sorted(levels)[:-1]
    if best[1] < -8 and others and best[1] - others[-1] < 6: return None, 0, 0, 'octave unclear'
    m = hz_to_midi(best[2])
    return best[0], 100 * (m - best[0]), float(ex[pc]), 'ok'
