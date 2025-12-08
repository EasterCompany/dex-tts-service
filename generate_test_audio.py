#!/usr/bin/env python3

import requests
import argparse
import os
import sys


def generate_audio(
    text: str,
    language: str = "en",
    speaker_wav_path: str = None,
    output_filename: str = "output.wav",
):
    """
    Generates audio from the Dexter TTS service.
    """
    service_url = "http://127.0.0.1:8200/generate"

    # Resolve the speaker_wav_path if not provided
    if speaker_wav_path is None:
        script_dir = os.path.dirname(__file__)
        speaker_wav_path = os.path.join(script_dir, "assets", "reference.wav")

    if not os.path.exists(speaker_wav_path):
        print(
            f"Error: Speaker WAV file not found at '{speaker_wav_path}'.",
            file=sys.stderr,
        )
        print(
            "Please ensure your 'reference.wav' is in the 'assets' directory or provide a valid path.",
            file=sys.stderr,
        )
        sys.exit(1)

    payload = {
        "text": text,
        "language": language,
        "speaker_wav": speaker_wav_path,
    }

    print(f"Sending request to TTS service for text: '{text}'")
    try:
        response = requests.post(service_url, json=payload, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

        if response.headers.get("Content-Type") == "audio/wav":
            with open(output_filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Audio successfully generated and saved to '{output_filename}'")
        else:
            print(
                f"Error: Unexpected content type received: {response.headers.get('Content-Type')}",
                file=sys.stderr,
            )
            print(f"Response content: {response.text}", file=sys.stderr)

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to TTS service: {e}", file=sys.stderr)
        print("Please ensure the Dexter TTS service is running.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate sample audio using Dexter TTS Service."
    )
    parser.add_argument(
        "text",
        type=str,
        nargs="?",
        help="The text to convert to speech.",
        default="Hello, this is Dexter speaking. I am now fully operational. Hannah, prepare yourself for tactical penis insertion.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="The language to generate the speech in (e.g., 'en').",
    )
    parser.add_argument(
        "--speaker_wav",
        type=str,
        help="Path to the speaker reference WAV file. Defaults to assets/reference.wav.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output filename for the generated audio.",
    )

    args = parser.parse_args()

    # Ensure the script is run from the venv if not already
    if "VIRTUAL_ENV" not in os.environ:
        script_dir = os.path.dirname(__file__)
        venv_activate = os.path.join(script_dir, "venv", "bin", "activate")
        print(
            f"Warning: Not in virtual environment. Please activate it first: source {venv_activate}",
            file=sys.stderr,
        )
        print(
            "Attempting to proceed, but it might fail without venv active.",
            file=sys.stderr,
        )
        # This is a bit tricky; subprocess.run(['bash', '-c', f'source {venv_activate} && python {sys.argv[0]} ...'])
        # is complex. For a simple helper, warning is enough.

    generate_audio(args.text, args.lang, args.speaker_wav, args.output)
