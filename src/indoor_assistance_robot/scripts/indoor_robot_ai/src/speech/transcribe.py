"""CLI: transcribe one audio file or a microphone recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from src.config_loader import PROJECT_ROOT
from src.logging_utils import setup_logging
from src.speech.audio_capture import record_microphone, save_audio_file
from src.speech.pipeline import transcribe_array, transcribe_file
from src.speech.types import Transcript


def _to_dict(transcript: Transcript) -> dict:
    return {
        "text": transcript.text,
        "language": transcript.language,
        "speech_detected": transcript.speech_detected,
        "latency_seconds": transcript.latency_seconds,
        "pipeline_latency_seconds": transcript.pipeline_latency_seconds,
        "source": transcript.source,
        "audio_path": transcript.audio_path,
        "segments": [
            {"start": seg.start, "end": seg.end, "text": seg.text}
            for seg in transcript.segments
        ],
    }


def main(argv: List[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Transcribe speech with Faster-Whisper. Output is text only."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        help="Path to a recorded audio file (preferred reproducible path).",
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Record from the default microphone instead of a file.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Microphone duration in seconds.",
    )
    parser.add_argument(
        "--save-recording",
        default=None,
        help="If --mic, write the captured waveform to this path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full Transcript object as JSON.",
    )
    args = parser.parse_args(argv)

    if args.mic:
        audio, source_rate = record_microphone(duration_seconds=args.duration)
        saved_path = None
        if args.save_recording:
            save_path = Path(args.save_recording)
            if not save_path.is_absolute():
                save_path = PROJECT_ROOT / save_path
            save_audio_file(save_path, audio, source_rate)
            saved_path = str(save_path)
        transcript = transcribe_array(
            audio,
            source_rate,
            source="microphone",
            audio_path=saved_path,
        )
    elif args.audio:
        transcript = transcribe_file(args.audio)
    else:
        parser.error("Provide an audio file or pass --mic.")
        return 2

    if args.json:
        print(json.dumps(_to_dict(transcript), indent=2))
    elif not transcript.speech_detected:
        print("")
        print("No speech detected.")
    else:
        print(transcript.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
