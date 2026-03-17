"""
Hakka TTS – 客語漢字 → Speech
==============================
Pipeline:
  漢字 → formog2p (IPA per syllable) → per-syllable espeak-ng synthesis with pitch control → WAV

Supported dialects (腔調):
  hak_sx   四縣 (Sixian)
  hak_nsx  南四縣 (South Sixian)
  hak_hl   海陸 (Hailu)
  hak_dp   大埔 (Dapu)
  hak_rp   饒平 (Raoping)
  hak_za   詔安 (Zhaoan)

Usage:
  python hakka_tts.py --text "天公落水" --dialect hak_sx --out out.wav
  python hakka_tts.py --test [--test-dir ./out]
  python hakka_tts.py --list-dialects
  python hakka_tts.py --phonemes-only --text "天公落水" --dialect hak_sx
"""

import argparse
import ctypes
import os
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# formog2p
# ---------------------------------------------------------------------------
try:
    from formog2p.hakka import g2p, G2PResult
except ImportError:
    sys.exit("formog2p not installed – run: pip install formog2p")

# ---------------------------------------------------------------------------
# Dialect registry
# ---------------------------------------------------------------------------
DIALECTS = {
    "hak_sx":  "四縣",
    "hak_nsx": "南四縣",
    "hak_hl":  "海陸",
    "hak_dp":  "大埔",
    "hak_rp":  "饒平",
    "hak_za":  "詔安",
}

# ---------------------------------------------------------------------------
# IPA + tone → espeak IPA phoneme string
#
# formog2p IPA format per syllable: "onset-rime_tone"
# e.g. "tʰ-ien_24" → onset=tʰ, rime=ien, tone=24
#
# espeak-ng IPA mode (espeakPHONEMES_IPA = 0x10):
#   - Feed raw IPA characters (space-separated phonemes)
#   - Use [[t:start,end]] for pitch targets (0-100)
#   - Syllable boundary: space
#
# Tone contours → (pitch_start, pitch_end) on 0-100 scale
# ---------------------------------------------------------------------------

_TONE_PITCH: dict[str, tuple[int, int]] = {
    "11": (10, 10),  "13": (10, 40),  "24": (20, 60),
    "31": (50, 10),  "33": (40, 40),  "35": (30, 70),
    "42": (65, 25),  "44": (70, 70),  "53": (80, 40),
    "55": (90, 90),
    # 入聲 (short stop-final)
    "2":  (20, 20),  "5":  (80, 80),
    "21": (25, 10),  "32": (45, 25),  "43": (60, 40),
    "54": (80, 60),
}


@dataclass
class SyllableInfo:
    """Parsed syllable with IPA body and tone pitch."""
    ipa: str
    tone: str
    pitch: int  # average pitch 0-100 for this tone

    @staticmethod
    def from_token(token: str) -> "SyllableInfo":
        """Parse a formog2p syllable token like 'tʰ-ien_24'."""
        tone_str = ""
        if "_" in token:
            body, tone_str = token.rsplit("_", 1)
        else:
            body = token
        ipa_body = body.replace("-", "")
        if tone_str in _TONE_PITCH:
            p0, p1 = _TONE_PITCH[tone_str]
            pitch = (p0 + p1) // 2
        else:
            pitch = 50  # default mid pitch
        return SyllableInfo(ipa=ipa_body, tone=tone_str, pitch=pitch)


def text_to_syllables(text: str, dialect: str) -> Tuple[List[SyllableInfo], List[str]]:
    """
    Convert Hakka Chinese text to a list of SyllableInfo objects.
    Returns: (syllables, unknown_words)
    """
    result: G2PResult = g2p(text, dialect, "ipa")

    syllables: list[SyllableInfo] = []
    for entry in result.pronunciations:
        if entry in ("，", "。", "？", "！", "、"):
            continue
        for syl in entry.split():
            syllables.append(SyllableInfo.from_token(syl))

    return syllables, result.unknown_words


def text_to_phoneme_str(text: str, dialect: str) -> Tuple[str, List[str]]:
    """
    Convert Hakka Chinese text to a display-friendly IPA phoneme string.
    Returns: (phoneme_str, unknown_words)
    """
    syllables, unknown = text_to_syllables(text, dialect)
    parts = []
    for s in syllables:
        if s.tone:
            parts.append(f"{s.ipa}({s.tone})")
        else:
            parts.append(s.ipa)
    return " ".join(parts), unknown


# ---------------------------------------------------------------------------
# espeak-ng ctypes interface
# ---------------------------------------------------------------------------

_ESPEAK_LIB  = "libespeak-ng.so.1"
_ESPEAK_DATA = "/usr/lib/x86_64-linux-gnu/espeak-ng-data"

# espeak constants
_AUDIO_OUTPUT_SYNCHRONOUS = 2
_espeakCHARS_UTF8         = 0x1
_espeakPHONEMES_IPA       = 0x10  # IPA phoneme string input
_espeakRATE               = 1
_espeakPITCH              = 3

_lib = None
_lib_initialized = False
_cb_ref = None          # keep callback alive across calls
_pcm_chunks: list[bytes] = []


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    lib = ctypes.cdll.LoadLibrary(_ESPEAK_LIB)

    lib.espeak_Initialize.restype  = ctypes.c_int
    lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]

    lib.espeak_SetVoiceByName.restype  = ctypes.c_int
    lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]

    lib.espeak_SetParameter.restype  = ctypes.c_int
    lib.espeak_SetParameter.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

    lib.espeak_SetSynthCallback.restype  = None
    lib.espeak_SetSynthCallback.argtypes = [ctypes.c_void_p]

    lib.espeak_Synth.restype  = ctypes.c_int
    lib.espeak_Synth.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_uint, ctypes.c_int, ctypes.c_uint,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.c_void_p,
    ]

    lib.espeak_Synchronize.restype = ctypes.c_int
    lib.espeak_Terminate.restype   = ctypes.c_int

    _lib = lib
    return lib


def _ensure_espeak_init():
    """Initialize espeak-ng once per process."""
    global _lib_initialized, _cb_ref
    lib = _get_lib()

    if not _lib_initialized:
        lib.espeak_Initialize(_AUDIO_OUTPUT_SYNCHRONOUS, 500, _ESPEAK_DATA.encode(), 0)
        _lib_initialized = True

    # Register PCM callback (keep reference alive)
    SYNTH_CB = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_short),
        ctypes.c_int,
        ctypes.c_void_p,
    )

    def _cb(wav_ptr, numsamples, events_ptr):
        if wav_ptr and numsamples > 0:
            arr = (ctypes.c_short * numsamples).from_address(
                ctypes.cast(wav_ptr, ctypes.c_void_p).value
            )
            _pcm_chunks.append(bytes(arr))
        return 0

    _cb_ref = SYNTH_CB(_cb)
    lib.espeak_SetSynthCallback(_cb_ref)

    # Use English voice as IPA phoneme renderer base
    lib.espeak_SetVoiceByName(b"en")

    return lib


def _synth_one_syllable(lib, ipa: str, rate: int, pitch: int) -> bytes:
    """Synthesize a single IPA syllable with specific pitch. Returns raw PCM."""
    _pcm_chunks.clear()
    lib.espeak_SetParameter(_espeakRATE, rate, 0)
    lib.espeak_SetParameter(_espeakPITCH, pitch, 0)

    encoded = ipa.encode("utf-8")
    buf = ctypes.create_string_buffer(encoded)
    lib.espeak_Synth(
        buf, len(encoded) + 1,
        0, 0, 0,
        _espeakPHONEMES_IPA | _espeakCHARS_UTF8,
        None, None,
    )
    lib.espeak_Synchronize()
    return b"".join(_pcm_chunks)


def synthesize(text: str, dialect: str, rate: int = 300, pitch: int = 50) -> "TTSResult":
    """
    Synthesize Hakka Chinese text to PCM audio.

    Uses per-syllable synthesis: each syllable is rendered individually with
    its tone-appropriate pitch value, then concatenated. This avoids the bug
    where espeak-ng IPA mode reads pitch control tags as literal text.

    Args:
        text:    漢字
        dialect: one of DIALECTS keys (e.g. "hak_sx")
        rate:    speech rate wpm 80-450 (default 300)
        pitch:   base pitch 0-100 (default 50, modulated by tone)

    Returns:
        TTSResult
    """
    if dialect not in DIALECTS:
        raise ValueError(f"Unknown dialect '{dialect}'. Valid: {list(DIALECTS)}")

    # Step 1: G2P → syllable list with tone info
    syllables, unknown = text_to_syllables(text, dialect)
    phoneme_str, _ = text_to_phoneme_str(text, dialect)

    # Step 2: Init espeak
    lib = _ensure_espeak_init()

    # Step 3: Synthesize each syllable with tone-appropriate pitch
    all_pcm = b""
    for syl in syllables:
        # Blend the syllable's tone pitch with the user-requested base pitch
        # tone pitch is 0-100; base pitch shifts the center
        effective_pitch = max(0, min(99, syl.pitch + (pitch - 50)))
        pcm = _synth_one_syllable(lib, syl.ipa, rate, effective_pitch)
        all_pcm += pcm

    return TTSResult(
        sample_rate=22050,
        pcm=all_pcm,
        unknown_words=unknown,
        phoneme_str=phoneme_str,
        dialect=dialect,
        dialect_name=DIALECTS[dialect],
    )


@dataclass
class TTSResult:
    sample_rate: int
    pcm: bytes
    unknown_words: List[str]
    phoneme_str: str
    dialect: str
    dialect_name: str

    def save_wav(self, path: str) -> str:
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)        # 16-bit LE
            wf.setframerate(self.sample_rate)
            wf.writeframes(self.pcm)
        return path

    def duration_sec(self) -> float:
        if self.sample_rate <= 0 or len(self.pcm) == 0:
            return 0.0
        return len(self.pcm) / 2 / self.sample_rate


# ---------------------------------------------------------------------------
# Built-in test suite
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "id": "T1",
        "text": "天公落水",
        "dialect": "hak_sx",
        "desc": "四縣腔・常見詞・含入聲",
    },
    {
        "id": "T2",
        "text": "客家人愛唱歌",
        "dialect": "hak_hl",
        "desc": "海陸腔・一般句子",
    },
    {
        "id": "T3",
        "text": "阿爸食飯無？",
        "dialect": "hak_dp",
        "desc": "大埔腔・日常問句",
    },
]


def run_tests(out_dir: str = "test_output"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sep = "=" * 60
    print(f"\n{sep}")
    print("  Hakka TTS – Test Suite (3 cases)")
    print(f"{sep}\n")

    passed = 0
    for tc in TEST_CASES:
        print(f"[{tc['id']}] {tc['desc']}")
        print(f"  text:    {tc['text']}")
        print(f"  dialect: {tc['dialect']} ({DIALECTS[tc['dialect']]})")
        try:
            result = synthesize(tc["text"], tc["dialect"])
            print(f"  phonemes: {result.phoneme_str}")
            if result.unknown_words:
                print(f"  ⚠ unknown: {result.unknown_words}")
            wav_path = os.path.join(out_dir, f"{tc['id']}_{tc['dialect']}.wav")
            result.save_wav(wav_path)
            dur = result.duration_sec()
            kb  = len(result.pcm) / 1024
            print(f"  ✅ {wav_path}  [{dur:.2f}s | {kb:.1f} KB | {result.sample_rate} Hz]")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  ❌ {e}")
            traceback.print_exc()
        print()

    print(f"Results: {passed}/{len(TEST_CASES)} passed")
    if passed == len(TEST_CASES):
        print("All tests passed ✅")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hakka TTS – 客語漢字 → Speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hakka_tts.py --text "天公落水" --dialect hak_sx --out rain.wav
  python hakka_tts.py --text "客家人愛唱歌" --dialect hak_hl --out song.wav
  python hakka_tts.py --test
  python hakka_tts.py --list-dialects
  python hakka_tts.py --phonemes-only --text "天公落水" --dialect hak_sx
        """,
    )
    parser.add_argument("--text",    help="Hakka Chinese text (漢字)")
    parser.add_argument(
        "--dialect", default="hak_sx", choices=list(DIALECTS), metavar="DIALECT",
        help=f"Dialect 腔調 (default: hak_sx). Options: {', '.join(DIALECTS)}",
    )
    parser.add_argument("--out",      default="output.wav", help="Output WAV path")
    parser.add_argument("--rate",     type=int, default=300, help="Speech rate wpm (default 300)")
    parser.add_argument("--pitch",    type=int, default=50,  help="Pitch 0-100 (default 50)")
    parser.add_argument("--test",     action="store_true",   help="Run built-in test suite")
    parser.add_argument("--test-dir", default="test_output", help="Dir for test WAVs")
    parser.add_argument("--list-dialects", action="store_true", help="Print supported dialects")
    parser.add_argument("--phonemes-only", action="store_true", help="Print phonemes only, skip synthesis")

    args = parser.parse_args()

    if args.list_dialects:
        print("Supported dialects / 支援腔調:")
        for k, v in DIALECTS.items():
            print(f"  {k:<12} {v}")
        return

    if args.test:
        run_tests(args.test_dir)
        return

    if not args.text:
        parser.print_help()
        return

    if args.phonemes_only:
        ph, unk = text_to_phoneme_str(args.text, args.dialect)
        print(f"Dialect:  {args.dialect} ({DIALECTS[args.dialect]})")
        print(f"Phonemes: {ph}")
        if unk:
            print(f"Unknown:  {unk}")
        return

    result = synthesize(args.text, args.dialect, rate=args.rate, pitch=args.pitch)
    result.save_wav(args.out)
    print(f"Text:     {args.text}")
    print(f"Dialect:  {result.dialect} ({result.dialect_name})")
    print(f"Phonemes: {result.phoneme_str}")
    if result.unknown_words:
        print(f"Unknown:  {result.unknown_words}")
    print(f"Output:   {args.out}  [{result.duration_sec():.2f}s | {len(result.pcm)//1024} KB | {result.sample_rate} Hz]")


if __name__ == "__main__":
    main()
