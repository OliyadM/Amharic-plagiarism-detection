# Make sure MODEL_PATH points to your actual file location
# Example: if it's in the same folder → "cc.am.300.bin"
# If it's elsewhere → full path like r"C:\Users\oliya\Downloads\cc.am.300.bin"

import fasttext
import numpy as np

MODEL_PATH = "cc.am.300.bin"   # ← change this if needed

try:
    print("Starting to load model... (expect 20–90 seconds first time)")
    model = fasttext.load_model(MODEL_PATH)
    print("Model loaded successfully!")
    print("Dimension:", model.get_dimension())  # Should be 300

    test_word = "ተማሪ"
    vector = model.get_word_vector(test_word)
    print(f"Vector for '{test_word}' (first 10 values):", vector[:10])
    print("Vector shape:", vector.shape)

except Exception as e:
    print("Error:", str(e))