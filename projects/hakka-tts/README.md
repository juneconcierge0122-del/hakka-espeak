# hakka-espeak

**客語漢字 → Speech** — A lightweight Hakka TTS pipeline using [formog2p](https://github.com/hungshinlee/formospeech-g2p) for G2P and [espeak-ng](https://github.com/espeak-ng/espeak-ng) for synthesis.

## Features

- 六腔支援 (Six Hakka dialects)：四縣、南四縣、海陸、大埔、饒平、詔安
- IPA-based phoneme pipeline with tone pitch contour injection
- No external TTS API required — runs fully offline
- Simple CLI and importable Python module

## Supported Dialects

| Code      | 腔調   |
|-----------|--------|
| `hak_sx`  | 四縣   |
| `hak_nsx` | 南四縣 |
| `hak_hl`  | 海陸   |
| `hak_dp`  | 大埔   |
| `hak_rp`  | 饒平   |
| `hak_za`  | 詔安   |

## Requirements

- Ubuntu/Debian with `espeak-ng` data installed:
  ```bash
  sudo apt install espeak-ng-data libespeak-ng1
  ```
- Python dependencies:
  ```bash
  pip install formog2p
  ```

## Usage

### CLI

```bash
# Synthesize text to WAV
python hakka_tts.py --text "天公落水" --dialect hak_sx --out rain.wav

# Run built-in test suite (3 dialects)
python hakka_tts.py --test

# Print IPA phonemes only (no synthesis)
python hakka_tts.py --phonemes-only --text "天公落水" --dialect hak_sx

# List supported dialects
python hakka_tts.py --list-dialects

# Adjust speech rate and pitch
python hakka_tts.py --text "客家人愛唱歌" --dialect hak_hl --rate 250 --pitch 60 --out song.wav
```

### Python API

```python
from hakka_tts import synthesize

result = synthesize("天公落水", dialect="hak_sx")
result.save_wav("output.wav")

print(result.phoneme_str)   # IPA phoneme string with tone markers
print(result.duration_sec())  # audio duration in seconds
print(result.unknown_words)   # list of OOV words (if any)
```

## Pipeline

```
漢字
 ↓
formog2p  (G2P, IPA per syllable, e.g. "tʰ-ien_24")
 ↓
tone pitch injection  ([[t:start,end]] espeak markers)
 ↓
espeak-ng IPA mode  (libespeak-ng.so via ctypes)
 ↓
WAV (22050 Hz, 16-bit mono)
```

### Tone Contours

Tone numbers from formog2p are mapped to espeak pitch targets (0–100 scale):

| Tone | Contour | Pitch |
|------|---------|-------|
| 24   | Rising  | 20→60 |
| 31   | Falling | 50→10 |
| 55   | High    | 90→90 |
| 11   | Low     | 10→10 |
| 53   | H-Fall  | 80→40 |
| 5    | 入聲 H  | 80→80 |
| 2    | 入聲 L  | 20→20 |
| ...  | ...     | ...   |

## Test Suite

```bash
python hakka_tts.py --test --test-dir ./out
```

| ID | Text      | Dialect | Expected |
|----|-----------|---------|----------|
| T1 | 天公落水  | hak_sx  | 4 syllables, rising+falling tones |
| T2 | 客家人愛唱歌 | hak_hl | 6 syllables, mixed tones |
| T3 | 阿爸食飯無？ | hak_dp | 5 syllables, stop finals |

## Known Limitations

- espeak-ng uses an English phoneme engine base; Hakka phonology is approximated via IPA passthrough — audio sounds synthetic, not natural Hakka speech.
- OOV words (not in formog2p's dictionary) are skipped.
- For higher-quality synthesis, a Hakka-specific acoustic model (e.g. VITS/FastSpeech2) would be needed.

## License

MIT
