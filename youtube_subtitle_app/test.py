import time
from pydub import AudioSegment
import numpy as np
import soundfile as sf
import io

from youtube_subtitle_app.transcription.nemo_transcriber import (
    transcribe_audio_word_level,
)

from youtube_subtitle_app.config import DEFAULT_MODEL

# Parameters
chunk_size_ms = 1600  # Length of each chunk in milliseconds (e.g., 1.6 seconds)
file_path = "tests/data/long.mp3"  # Path to your MP3 file


# Function to simulate live streaming
def simulate_live_streaming(file_path, chunk_size_ms):
    # Load audio file using pydub
    audio = AudioSegment.from_mp3(file_path)

    # Total length of the audio
    audio_length = len(audio)  # in milliseconds

    print(f"Total audio length: {audio_length / 1000} seconds")

    # Simulate real-time streaming by processing in chunks
    start_time = time.time()

    for i in range(0, audio_length, chunk_size_ms):
        # Extract the current chunk
        chunk = audio[i : i + chunk_size_ms]

        # Convert chunk to a numpy array for the model
        chunk_data = np.array(chunk.get_array_of_samples())
        chunk_data = chunk_data.astype(
            np.float32
        )  # Make sure the data type is suitable for your model

        # Optionally save the chunk as a WAV file to pass to your model
        # Here, we're using soundfile to save as a temporary buffer
        buffer = io.BytesIO()
        sf.write(buffer, chunk_data, chunk.frame_rate, format="WAV")

        # Simulate the chunk passing to the model
        # You can replace this with your model inference call
        print(
            f"Processing chunk {i // chunk_size_ms + 1}/{audio_length // chunk_size_ms}..."
        )
        # Insert your transcription code here (e.g., feeding the chunk to the model)

        word_segments = transcribe_audio_word_level(processed_audio, DEFAULT_MODEL)

        # Simulate real-time delay for each chunk (to mimic streaming)
        # time.sleep(chunk_size_ms / 1000)  # Delay for chunk length (e.g., 1.6 seconds)

        # Optionally, print or process the transcription here
        # For example:
        # transcription = your_model.transcribe(chunk_data)
        # print(f"Transcription for chunk {i // chunk_size_ms + 1}: {transcription}")

    end_time = time.time()
    print(f"Finished processing. Total time: {end_time - start_time:.2f} seconds")


# Call the function to start the simulated live streaming
simulate_live_streaming(file_path, chunk_size_ms)
