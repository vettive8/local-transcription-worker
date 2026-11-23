# Local Transcription Worker

A small command-line worker for turning MP4 recordings into timestamped text and SRT subtitles. It uses faster-whisper on the CPU. Audio stays on the machine. The first run downloads the selected model unless it is already cached.

## Run

Tested with Python 3.12 on Windows. In a virtual environment:

    python -m pip install -r requirements.txt
    python transcribe.py interview.mp4 --output transcripts
    python transcribe.py recordings --output transcripts
    python transcribe.py recordings --output transcripts --watch

A directory input includes its subdirectories. Each recording gets its own output directory with Transcript.txt, Subtitles.srt, and source.json. Use a dedicated output directory per input collection and run one worker per output directory.

Watch mode checks every 30 seconds and waits for an unchanged file across two checks. For downloads, finish the download before placing the file in the watched directory. Keep the process running to continue watching. Stop it with Ctrl+C. Restarting skips completed recordings with matching source and model settings. Errors are printed and retried on later watch passes.

The default model is base.en. For other languages select a multilingual model and language, for example --model small --language pl. Use --offline after downloading the model to require local inference assets.
