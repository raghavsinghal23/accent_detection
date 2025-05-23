import os
import subprocess
from yt_dlp import YoutubeDL
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch
import torch.nn.functional as F
import numpy as np
import streamlit as st

# Load pretrained model and processor from HuggingFace
@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model.eval()
    return processor, model

processor, model = load_model()

# Fixed accent prototypes (replace with your actual trained centroids)
accent_prototypes = {
    "Indian English": np.random.rand(768) * 0.1 + 0.9,  # Sample values - replace with real data
    "American English": np.random.rand(768) * 0.1 + 0.7,
    "British English": np.random.rand(768) * 0.1 + 0.8,
    "Australian English": np.random.rand(768) * 0.1 + 0.6,
}

def download_audio_from_url(video_url, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    base_path = os.path.join(output_dir, 'audio')
    mp3_path = f"{base_path}.mp3"
    wav_path = f"{base_path}.wav"

    ydl_opts = {
        'ffmpeg_location': '/usr/bin/ffmpeg',  # Force Colab's FFmpeg path
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
        'extractor_args': {'youtube': {'skip': ['dash', 'hls']}},
        'format': 'bestaudio/best',
        'outtmpl': base_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'extract_audio': True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        st.error(f"Failed to download audio: {e}")
        return None

    if os.path.exists(mp3_path):
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', mp3_path,
                '-ac', '1',
                '-ar', '16000',
                wav_path
            ], check=True, stderr=subprocess.PIPE)
            os.remove(mp3_path)
            return wav_path
        except subprocess.CalledProcessError as e:
            st.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
            return None
        except Exception as e:
            st.error(f"Error during conversion: {e}")
            return None
    else:
        st.error(f"MP3 file not found at {mp3_path}")
        return None

def detect_accent_wav2vec2(wav_path):
    # Load and preprocess audio
    speech, sr = torchaudio.load(wav_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        speech = resampler(speech)

    input_values = processor(speech.squeeze(), return_tensors="pt", sampling_rate=16000).input_values

    # Get embeddings
    with torch.no_grad():
        outputs = model(input_values)
        hidden_states = outputs.last_hidden_state
        mean_embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()

    # Find closest prototype
    similarities = {}
    for accent, proto in accent_prototypes.items():
        cosine_sim = F.cosine_similarity(
            torch.tensor(mean_embedding), 
            torch.tensor(proto), 
            dim=0
        ).item()
        similarities[accent] = cosine_sim

    # Softmax normalization for confidence scores
    similarity_scores = np.array(list(similarities.values()))
    softmax_scores = np.exp(similarity_scores) / np.sum(np.exp(similarity_scores))
    
    # Get best match
    best_idx = np.argmax(softmax_scores)
    best_accent = list(accent_prototypes.keys())[best_idx]
    confidence = softmax_scores[best_idx] * 100

    return best_accent, confidence

def main():
    st.title("YouTube Accent Detection")
    st.write("This app detects the English accent from a YouTube video using Wav2Vec2 model.")
    
    video_url = st.text_input("Enter YouTube video URL:")
    
    if st.button("Detect Accent"):
        if video_url:
            with st.spinner("Processing..."):
                output_dir = "./output"
                
                try:
                    audio_file = download_audio_from_url(video_url, output_dir)
                    if audio_file:
                        st.success("Audio extracted successfully!")
                        
                        accent, confidence = detect_accent_wav2vec2(audio_file)
                        
                        st.subheader("Results:")
                        col1, col2 = st.columns(2)
                        col1.metric("Detected Accent", accent)
                        col2.metric("Confidence", f"{confidence:.2f}%")
                        
                        os.remove(audio_file)
                    else:
                        st.error("Failed to extract audio.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a YouTube URL")

if __name__ == "__main__":
    main()
