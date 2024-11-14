from flask import Flask, render_template, request, jsonify
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
)
import torch
import os
import librosa
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max file size

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

device = "cuda"
# device = "cpu"

print("Loading Whisper model...")
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
whisper_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3-turbo"
).to(device)

print("Loading MBart model...")
translation_model = MBartForConditionalGeneration.from_pretrained(
    "facebook/mbart-large-50-many-to-many-mmt"
).to(device)
tokenizer = MBart50TokenizerFast.from_pretrained(
    "facebook/mbart-large-50-many-to-many-mmt"
)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "wav",
        "mp3",
        "m4a",
        "ogg",
    }


def split_audio(audio, sr, chunk_duration=30):
    """Split audio into chunks of specified duration in seconds."""
    chunk_length = int(sr * chunk_duration)
    chunks = []
    for i in range(0, len(audio), chunk_length):
        chunk = audio[i : i + chunk_length]
        if len(chunk) < chunk_length:
            chunk = np.pad(chunk, (0, chunk_length - len(chunk)))
        chunks.append(chunk)
    return chunks


def process_audio_chunk(audio_chunk, sr):
    """Process a single audio chunk and return transcribed text."""
    try:
        input_features = whisper_processor(
            audio_chunk, sampling_rate=sr, return_tensors="pt"
        ).input_features.to(device)
        generated_ids = whisper_model.generate(
            input_features, max_length=448, num_beams=1
        )
        transcription = whisper_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return transcription.strip()
    except Exception as e:
        print(f"Error in process_audio_chunk: {str(e)}")
        return ""


def process_audio(audio_path):
    """Process audio file in chunks and return combined transcribed text."""
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        chunks = split_audio(audio, sr)
        transcriptions = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            chunk_transcription = process_audio_chunk(chunk, sr)
            transcriptions.append(chunk_transcription)
        full_transcription = " ".join(transcriptions)
        return full_transcription
    except Exception as e:
        print(f"Error in process_audio: {str(e)}")
        return ""


def save_transcription_file(full_transcription, filename):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(str(full_transcription))


def translate_text(text, target_language_code, chunk_size=200):
    """Translate text to a specified language using MBart model with chunking."""
    try:
        tokenizer.src_lang = "en_XX"
        text_chunks = [
            text[i : i + chunk_size] for i in range(0, len(text), chunk_size)
        ]

        translated_chunks = []
        for idx, chunk in enumerate(text_chunks):
            if not chunk.strip():
                continue

            try:
                print(f"Translating chunk {idx+1}/{len(text_chunks)}: {chunk[:50]}...")
                encoded_text = tokenizer(chunk, return_tensors="pt").to(device)
                generated_tokens = translation_model.generate(
                    **encoded_text,
                    forced_bos_token_id=tokenizer.lang_code_to_id[target_language_code],
                )
                translated_chunk = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )[0]
                translated_chunks.append(translated_chunk)
            except Exception as e:
                print(f"Error in translating chunk {idx+1}: {str(e)}")
                translated_chunks.append("")

        translated_text = " ".join(translated_chunks).strip()
        return translated_text if translated_text else "Translation Error"

    except Exception as e:
        print(f"Error in translate_text function: {str(e)}")
        return "Translation Error"


def save_translate_file(translated_text, filename):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(str(translated_text))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        file = request.files["audio"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        transcribed_text = process_audio(filepath)
        save_transcription_file(transcribed_text, "transcribe.txt")
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({"transcribed_text": transcribed_text})
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get("text", "")
    target_language = data.get("target_language", "fr_XX")

    if not text:
        return jsonify({"error": "No text provided for translation"}), 400

    translated_text = translate_text(text, target_language)
    save_translate_file(translated_text, "translated.txt")
    return jsonify({"translated_text": translated_text})


if __name__ == "__main__":
    app.run(debug=True)
