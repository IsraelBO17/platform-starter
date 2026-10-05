# tests/test_local_speech.py
import wave
from types import SimpleNamespace

from speech_pipeline.local_engine import LocalSpeechEngine


def test_local_engine_transcribes_and_speaks(tmp_path):
    engine = LocalSpeechEngine(voice_dir=str(tmp_path))

    class FakeWhisper:
        def transcribe(self, path, language=None):
            return iter([SimpleNamespace(text=" Hello"), SimpleNamespace(text=" class. ")]), None

    class FakeVoice:
        def synthesize_wav(self, text, wav_file):
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(b"\x00\x00" * 100)

    engine.__dict__["_whisper"] = FakeWhisper()   # replaces the cached_property
    engine.__dict__["_voice"] = FakeVoice()
    assert engine.transcribe("meeting.mp3") == "Hello class."
    out = engine.synthesize("Hi", str(tmp_path / "hi.wav"))
    with wave.open(out) as w:
        assert w.getnframes() == 100


def test_missing_voice_explains_download(tmp_path):
    engine = LocalSpeechEngine(voice_dir=str(tmp_path))
    try:
        engine.synthesize("Hi", str(tmp_path / "x.wav"))
    except FileNotFoundError as exc:
        assert "piper.download_voices" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
