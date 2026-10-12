import wave
import struct

sample_rate = 16000
duration_seconds = 2  # 2 seconds of audio
num_samples = sample_rate * duration_seconds

with wave.open("sample.wav", "w") as wav_file:
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 16-bit PCM
    wav_file.setframerate(sample_rate)
    
    # Write dummy audio frames (silence or simple tone)
    for i in range(num_samples):
        value = 0
        data = struct.pack('<h', value)
        wav_file.writeframes(data)

print("Successfully created sample.wav")