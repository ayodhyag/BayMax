import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoModelForSequenceClassification, AutoTokenizer
import sounddevice as sd
import soundfile as sf
import torch
import librosa
import noisereduce as nr

# Initialize the model and processor
model_name = "facebook/wav2vec2-large-960h"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Initialize the emotion detection model and tokenizer
emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)

# Initialize the emotion detection pipeline
# emotion_recognition = pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base")

# Directory fro saving files
base_dir = "C:\\Users\\Auburn\\OneDrive\\Desktop\\Emotion Recognition\\"


# Recording Audio
def record_audio(duration=5, sr=16000):
    mp3_path = os.path.join(base_dir, "audio.mp3")
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    sf.write(mp3_path, audio, sr)
    print(f"\nAudio is saved!!!")
    return mp3_path


def convert_mp3_to_wav(mp3_path):
    wav_path = os.path.join(base_dir, "audio.wav")
    audio, sr = sf.read(mp3_path)
    sf.write(wav_path, audio, sr)
    print("\nConverted mp3 file to wav format")
    return wav_path

# function to preprocess audio
def preprocess_audio(wav_path):
    # Load audio
    audio, sr = librosa.load(wav_path, sr=16000)
    # Noise reduction
    audio = nr.reduce_noise(y=audio, sr=sr)
    # Normalize audio
    audio = librosa.util.normalize(audio)
    # Save preprocessed audio
    preprocessed_wav_path = wav_path.replace(".wav", "_preprocessed.wav")
    sf.write(preprocessed_wav_path, audio, sr)
    return preprocessed_wav_path

# Perform speech-to-text
def perform_speech_to_text(wav_path):
    # Load audio file
    audio, _ = sf.read(wav_path)

    # Tokenize audio to features
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    # Perform speech-to-text
    with torch.no_grad():
        logits = model(input_values=inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription

# Perform Sentiment Analysis
def detect_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt")
    outputs = emotion_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    label_ids = torch.argmax(probs, dim=1)
    labels = ["anger", "disgust", "fear", "happy", "neutral", "sadness", "surprise"]
    return labels[label_ids]

#record audio
mp3_path = record_audio()

#convert mp3 to wav
wav_path = convert_mp3_to_wav(mp3_path)

# Perform emotion detection
transcription = perform_speech_to_text(wav_path)
print("\nTranscription:", transcription)
emotion = detect_emotion(transcription)
print("\nDetected Emotion:", emotion)

print("\n")


