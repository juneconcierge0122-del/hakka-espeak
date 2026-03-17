# hakka-espeak

**客語漢字 → Speech** — Hakka TTS using [formog2p](https://github.com/hungshinlee/formospeech-g2p) G2P + [espeak-ng](https://github.com/espeak-ng/espeak-ng) synthesis.

## v3 Changes

- **Fixed**: tone numbers no longer read aloud (was: espeak IPA mode treating `[[t:...]]` tags as text)
- **Per-syllable synthesis**: each syllable rendered individually with tone-appropriate pitch
- Phoneme display: human-readable `tʰien(24) kuŋ(24) lok(5) sui(31)`

## Supported Dialects

| Code | 腔調 | Code | 腔調 |
|------|------|------|------|
| `hak_sx` | 四縣 | `hak_dp` | 大埔 |
| `hak_nsx` | 南四縣 | `hak_rp` | 饒平 |
| `hak_hl` | 海陸 | `hak_za` | 詔安 |

## Setup

```bash
sudo apt install espeak-ng-data libespeak-ng1
pip install formog2p
```

## Usage

```bash
# Synthesize
python hakka_tts.py --text "天公落水" --dialect hak_sx --out out.wav

# Phonemes only
python hakka_tts.py --phonemes-only --text "天公落水" --dialect hak_sx

# Test suite
python hakka_tts.py --test
```

```python
from hakka_tts import synthesize

result = synthesize("天公落水", dialect="hak_sx")
result.save_wav("output.wav")
print(result.phoneme_str)    # tʰien(24) kuŋ(24) lok(5) sui(31)
print(result.duration_sec()) # seconds
```

## Pipeline (v3)

```
漢字 → formog2p(IPA) → SyllableInfo[] → per-syllable espeak_Synth(IPA, pitch) → concat → WAV
```

Each syllable's tone contour is mapped to a pitch value (0-100). Synthesis is done one syllable at a time to avoid the v2 bug where espeak-ng IPA mode misinterpreted pitch control tags.

## Test Results

| Text | Dialect | Phonemes | Duration |
|------|---------|----------|----------|
| 天公落水 | hak_sx | `tʰien(24) kuŋ(24) lok(5) sui(31)` | 1.32s |
| 客家人愛唱歌 | hak_hl | `hak(5) ka(53) ŋin(55) oi(11) ʈ͡ʂʰoŋ(11) ko(53)` | 3.35s |
| 阿爸食飯無？ | hak_dp | `a(33) pa(33) ʂit(54) pʰon(53) mo(113)` | 1.88s |

## Limitations

- Per-syllable concatenation lacks co-articulation smoothing
- espeak English voice base — sounds synthetic, not natural Hakka
- formog2p OOV coverage low for some dialects (南四縣, 饒平)

## License

MIT
