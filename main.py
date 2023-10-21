import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from logzero import logger
from pyannote.audio import Pipeline
from pydub import AudioSegment

load_dotenv()
ROOT_DIR = Path(__file__).parent
ENV_PATH = Path(ROOT_DIR, ".env")
AUDIO_PATH = Path(ROOT_DIR, "data/audio.wav")
OUTPUT_DIARIZATION_PATH = Path(ROOT_DIR, "data/diarization.txt")
HF_TOKEN = os.environ.get("HF_TOKEN")
NUM_SPEAKERS = 5


class SpeakerDiarization:
    def __init__(self) -> None:
        """Initialize SpeakerDiarization class."""
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token=HF_TOKEN)

    def run_diarization(self) -> None:
        """Run speaker diarization."""
        diarization = self.pipeline(AUDIO_PATH, num_speakers=NUM_SPEAKERS)
        logger.info(f"done: run speaker diarization")

        data = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            data.append([{"start": turn.start, "end": turn.end}, speaker])
            logger.info(f"start={turn.start:.1f}s end={turn.end:.1f}s {speaker}")

        df = pd.DataFrame(data, columns=["segment", "label"])
        self.output_audio_groups(df)

    def output_audio_groups(self, df: pd.DataFrame) -> None:
        audio_segment = AudioSegment.from_file(AUDIO_PATH, format="wav")

        for label in sorted(df["label"].unique()):
            filtered_df = df[df["label"] == label]
            audio_segment_each_speaker = AudioSegment.silent(duration=0)

            previous_end = 0
            for _, segment in enumerate(filtered_df["segment"]):
                start = segment["start"] * 1000
                end = segment["end"] * 1000
                duration = start - previous_end
                audio_segment_each_speaker += audio_segment[previous_end:start] - 12
                audio_segment_each_speaker += audio_segment[start:end]
                previous_end = end

            audio_segment_each_speaker += audio_segment[previous_end:]

            logger.info(f"{label}, {audio_segment_each_speaker.duration_seconds}s")
            audio_segment_each_speaker.export(f"outputs/{label}.wav", format="wav")


def main() -> None:
    speaker_diarization = SpeakerDiarization()
    logger.info("done: initialize SpeakerDiarization class")

    speaker_diarization.run_diarization()
    logger.info("done: run speaker diarization")


if __name__ == "__main__":
    main()
