import os
import wave
import struct
from pathlib import Path

# Create bonafide and spoof subfolders with 50 files each
for category in ["bonafide", "spoof"]:
    folder = Path("dataset") / category
    folder.mkdir(parents=True, exist_ok=True)
    
    for i in range(50):
        filepath = folder / f"sample_{i}.wav"
        sample_rate = 16000
        duration_seconds = 1
        num_samples = sample_rate * duration_seconds

        with wave.open(str(filepath), "w") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit PCM
            wav_file.setframerate(sample_rate)
            for _ in range(num_samples):
                wav_file.writeframes(struct.pack('<h', 0))

print("Large dummy dataset created successfully in 'dataset/' folder!")