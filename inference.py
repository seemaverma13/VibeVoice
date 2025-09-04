"""
VibeVoice Inference Script (Local Mode)
This version runs locally using your downloaded VibeVoice model.
It generates podcast audio from text input.
"""

import os
import torch
from utils import clean_text, create_output_dir, generate_filename, save_audio_file

# ======== MODEL LOADING ========
def load_vibevoice_model(model_path="models/vibevoice-1.5b"):
    """
    Load the VibeVoice model from a local directory.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model path '{model_path}' not found! Please download the model first."
        )

    print(f"[INFO] Loading VibeVoice model from {model_path} ...")

    # Import actual model & processor
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, processor

# ======== REAL GENERATION PIPELINE ========
def generate_podcast(content: str, model, processor) -> str:
    """
    Generate podcast audio from text using VibeVoice.
    Returns the path to the saved audio file.
    """
    cleaned_text = clean_text(content)

    # Convert text to input tokens for the model
    inputs = processor(text=cleaned_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        audio_output = model.generate(**inputs)  # Generates waveform tensor

    # Convert waveform to bytes if needed
    if isinstance(audio_output, torch.Tensor):
        audio_output = audio_output.cpu().numpy().tobytes()

    # Save generated audio
    filename = generate_filename()
    filepath = save_audio_file(audio_output, filename)

    return filepath

# ======== MAIN (Quick Test) ========
if __name__ == "__main__":
    # Example podcast script
    podcast_script = """
    Welcome to today's podcast! 
    In this episode, we'll explore how artificial intelligence
    is changing the way we create, consume, and share stories.
    Stay tuned for some fun facts and quick insights!
    """

    # Step 1: Create output folder
    create_output_dir()

    # Step 2: Load model locally
    model, processor = load_vibevoice_model()

    # Step 3: Generate podcast audio
    output_audio_path = generate_podcast(podcast_script, model, processor)

    print(f"[INFO] Podcast generated at: {output_audio_path}")
