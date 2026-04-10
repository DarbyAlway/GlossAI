import tqdm
import json
from faster_whisper import WhisperModel
from pathlib import Path

class SubtitleExtractor:
    def __init__(self, model_size="large-v3", device="cuda", compute_type="float16"):
        """
        Initializes the Whisper model. 
        Doing this in __init__ ensures the model stays in memory for multiple files.
        """
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def generate_subtitles(self, input_file, output_json="subtitles.json"):
        """
        Processes a media file and exports speech to a formatted JSON file.

        :param input_file: Path to the source video or audio file (e.g., 'video.mp4').
        :param output_json: The filename/path where the JSON data will be saved.

        """
        # 1. Transcribe the video/audio
        segments, info = self.model.transcribe(input_file, task="transcribe", beam_size=5)

        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        
        pbar = tqdm.tqdm(total=info.duration, unit="sec", desc=f"Processing {input_file}")

        subtitle_data = []

        # 2. Process segments
        for segment in segments:
            # Update progress bar based on the end timestamp of the current segment
            pbar.update(segment.end - pbar.n)
            
            entry = {
                "subtitle_id": len(subtitle_data) + 1,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text.strip()
            }
            
            subtitle_data.append(entry)

        pbar.close()

        # 3. Save to JSON file
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(subtitle_data, f, indent=4, ensure_ascii=False)

        print(f"Successfully saved {len(subtitle_data)} segments to {output_json}")
        return subtitle_data

# --- Example Usage ---
if __name__ == "__main__":
    # Initialize the engine once
    transcriber = SubtitleExtractor(model_size="large-v3")
    
    # Use it to process a file
    transcriber.generate_subtitles(
        input_file="../data/test_vid.mp4", 
        output_json=f"../subtitles/{Path('../data/test_vid.mp4').stem}_subtitles.json"
    )