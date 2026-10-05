# scripts/speech_local.py
"""Try open-source speech models locally.

  python -m scripts.speech_local listen outputs/meeting.mp3
  python -m scripts.speech_local speak "Your order has shipped." outputs/notice.wav
"""
import sys
import time

from speech_pipeline.local_engine import LocalSpeechEngine


def main():
    engine = LocalSpeechEngine()
    command = sys.argv[1]
    started = time.perf_counter()
    if command == "listen":
        print(engine.transcribe(sys.argv[2]))
    elif command == "speak":
        print("Saved", engine.synthesize(sys.argv[2], sys.argv[3]))
    else:
        raise SystemExit("usage: listen <audio file> | speak <text> <out.wav>")
    print(f"took {time.perf_counter() - started:.1f} s on this machine")


if __name__ == "__main__":
    main()
