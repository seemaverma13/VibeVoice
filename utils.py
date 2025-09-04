# app/utils.py
"""
Utility functions for VibeVoice project.
Keeps inference.py clean and modular.
"""

import os
from datetime import datetime

def clean_text(text: str) -> str:
    """
    Cleans and formats text before passing to the model.
    """
    text = text.strip()
    return " ".join(text.split())  # Remove extra spaces

def create_output_dir(dir_name="outputs"):
    """
    Create outputs directory if it doesn't exist.
    """
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def generate_filename(prefix="podcast"):
    """
    Generate a unique filename for each podcast audio file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.wav"

def save_audio_file(audio_data, filename, output_dir="outputs"):
    """
    Saves audio data to a file.
    For now, just simulates by writing a placeholder text file.
    When real model is loaded, write actual audio bytes here.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    # Simulate with placeholder
    with open(filepath, "wb") as f:
        if isinstance(audio_data, bytes):
            f.write(audio_data)
        else:
            f.write(b"Simulated audio data")

    return filepath
