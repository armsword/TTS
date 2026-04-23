"""Generate mock data for testing training pipeline"""
import numpy as np
import scipy.io.wavfile as wavfile
import os

# Create mock LJSpeech-like dataset
output_dir = "data/mock_ljspeech"
os.makedirs(f"{output_dir}/wavs", exist_ok=True)

# Create metadata with 20 samples
texts = [
    "Printing, in the only sense with which we are at present concerned, differs from most if not from all the other arts and crafts",
    "The beauty of the place has no effect on my mother",
    "I would not look so far as to seek for any",
    "The information which we have borrowed from no other source",
    "In the meantime, the new year was approaching",
    "The question is whether we can find such a person",
    "The book you lent me is very interesting",
    "I have lived in this city for many years",
    "The weather is beautiful today",
    "The quick brown fox jumps over the lazy dog",
    "A journey of a thousand miles begins with a single step",
    "To be or not to be, that is the question",
    "All that glitters is not gold",
    "Actions speak louder than words",
    "Better late than never",
    "Knowledge is power",
    "Practice makes perfect",
    "Time and tide wait for no man",
    "Where there is a will, there is a way",
    "You cannot judge a book by its cover"
]

# Write metadata.csv
with open(f"{output_dir}/metadata.csv", "w") as f:
    for i, text in enumerate(texts, 1):
        f.write(f"{i}|{text}\n")

# Create wav files (440 Hz sine wave for 1 second)
sample_rate = 22050
for i in range(1, 21):
    t = np.linspace(0, 1, sample_rate)
    # Add some variation
    freq = 440 + (i * 10)
    waveform = np.sin(2 * np.pi * freq * t).astype(np.float32)
    wavfile.write(f"{output_dir}/wavs/{i}.wav", sample_rate, waveform)

print(f"Created {len(texts)} mock samples in {output_dir}")
