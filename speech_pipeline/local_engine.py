# speech_pipeline/local_engine.py
"""Open-source speech models running on your own machine: free, private, no cloud account.

  Speech to text: Whisper (OpenAI's open-weights model) through the faster-whisper package
  Text to speech: Piper (a small, fast open-source voice model)
"""
import wave
from functools import cached_property
from pathlib import Path


class LocalSpeechEngine:
    def __init__(self, whisper_size: str = "small", voice: str = "en_US-lessac-medium",
                 voice_dir: str = "voices"):
        self.whisper_size = whisper_size          # tiny, base, small, medium, large-v3
        self.voice_path = Path(voice_dir) / f"{voice}.onnx"

    @cached_property
    def _whisper(self):
        from faster_whisper import WhisperModel

        # int8 on CPU keeps memory low; the model downloads once, then loads from cache.
        return WhisperModel(self.whisper_size, device="cpu", compute_type="int8")

    @cached_property
    def _voice(self):
        if not self.voice_path.exists():
            raise FileNotFoundError(
                f"{self.voice_path} not found. Download it with: "
                f"python -m piper.download_voices {self.voice_path.stem} --download-dir {self.voice_path.parent}"
            )
        from piper import PiperVoice

        return PiperVoice.load(str(self.voice_path))

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """Audio file in, text out. language=None lets Whisper detect it."""
        segments, info = self._whisper.transcribe(audio_path, language=language)
        return " ".join(segment.text.strip() for segment in segments)

    def synthesize(self, text: str, out_path: str) -> str:
        """Text in, a .wav file out."""
        voice = self._voice  # load (or fail clearly) before creating the output file
        with wave.open(out_path, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        return out_path
