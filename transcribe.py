"""Transcribe local MP4 files with faster-whisper."""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from faster_whisper import WhisperModel


def stamp(seconds):
    milliseconds = round(seconds * 1000)
    minutes, ms = divmod(milliseconds, 60_000)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{ms // 1000:02},{ms % 1000:03}"


def signature(path, args):
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns, args.model, args.language]


def transcribe(path, folder, args, sig, model):
    folder.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=folder) as staging:
        tmp = Path(staging)
        segments, _ = model.transcribe(str(path), language=args.language, beam_size=1, vad_filter=True)
        count = 0
        with (tmp / "Transcript.txt").open("w", encoding="utf-8") as txt, (tmp / "Subtitles.srt").open("w", encoding="utf-8") as srt:
            for count, segment in enumerate(segments, 1):
                start, end = stamp(segment.start), stamp(segment.end)
                line = segment.text.strip()
                txt.write(f"[{start[:8]}] {line}\n")
                srt.write(f"{count}\n{start} --> {end}\n{line}\n\n")
        if not count or signature(path, args) != sig:
            raise ValueError("Empty transcript or source changed during transcription")
        (tmp / "source.json").write_text(json.dumps(sig), encoding="utf-8")
        for name in ("Transcript.txt", "Subtitles.srt", "source.json"):
            os.replace(tmp / name, folder / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--model", default="base.en")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_dir() and not (source.is_file() and source.suffix.lower() == ".mp4"):
        parser.error("Provide an existing MP4 file or directory")
    if args.watch and not source.is_dir():
        parser.error("--watch requires a directory")
    model = None
    previous = {}
    while True:
        current = {}
        errors = False
        paths = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.suffix.lower() == ".mp4" and p.is_file())
        for path in paths:
            try:
                sig = signature(path, args)
                current[path] = sig
                if args.watch and previous.get(path) != sig:
                    continue
                relative = path.relative_to(source) if source.is_dir() else Path(path.name)
                folder = args.output / relative.with_suffix("")
                marker = folder / "source.json"
                if marker.is_file() and marker.read_text(encoding="utf-8") == json.dumps(sig) and all((folder / name).is_file() for name in ("Transcript.txt", "Subtitles.srt")):
                    continue
                if model is None:
                    model = WhisperModel(args.model, device="cpu", compute_type="int8", local_files_only=args.offline)
                print(f"Transcribing {path}", flush=True)
                transcribe(path, folder, args, sig, model)
                print(f"Saved {folder}", flush=True)
            except Exception as error:
                errors = True
                print(f"Failed {path}: {error}", flush=True)
        if not args.watch:
            return int(errors)
        previous = current
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
