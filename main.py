import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pydub import AudioSegment

load_dotenv()
ROOT_DIR = Path(__file__).parent
ENV_PATH = Path(ROOT_DIR, ".env")
AUDIO_PATH = Path(ROOT_DIR, "data/audio.wav")
OUTPUT_IMAGE_PATH = Path(ROOT_DIR, "outputs/speaker_diarization_result.png")
HF_TOKEN = os.environ.get("HF_TOKEN")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeakerDiarization:
    def __init__(self):
        """Initialize SpeakerDiarization class."""
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HF_TOKEN)

    def run_diarization(self):
        """Run speaker diarization."""
        diarization = self.pipeline({"audio": AUDIO_PATH}, num_speakers=9)
        df = pl.DataFrame(diarization.for_json()["content"])
        audio_segment = AudioSegment.from_file(AUDIO_PATH, format="wav")

        for label in sorted(df["label"].unique()):
            _df = df.filter(pl.col("label") == label)
            audio_segment_each_speaker = AudioSegment.silent()

            for _, segment in _df["segment"]:
                start = segment["start"] * 1000
                end = segment["end"] * 1000
                audio_segment_each_speaker += audio_segment[start:end]

            logger.info(f"{label=}, duration={audio_segment_each_speaker.duration_seconds}")
            audio_segment_each_speaker.export(f"outputs/{label}.wav", format="wav")

    def save_diarization_image(self):
        diarization = self.pipeline({"audio": AUDIO_PATH}, num_speakers=9)
        plt.figure(figsize=(10, 3))

        for segment, label in diarization.itertracks(yield_label=True):
            speaker_id = label.split(" ")[0]
            color = int(speaker_id) % 10
            plt.barh(0, segment.duration, left=segment.start, color=plt.cm.tab10(color), label=speaker_id)

        plt.yticks([])
        plt.savefig(OUTPUT_IMAGE_PATH, bbox_inches="tight", pad_inches=0.1)
        plt.show()


def main():
    speaker_diarization = SpeakerDiarization()
    speaker_diarization.save_diarization_image()
    speaker_diarization.run_diarization()


if __name__ == "__main__":
    main()
