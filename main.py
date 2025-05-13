import requests
from fastapi import FastAPI, UploadFile, File
# from pydub import AudioSegment
# import speech_recognition as sr
import tempfile
import whisper
from transformers import pipeline

app = FastAPI()
model_whisper = whisper.load_model("base")
pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", trust_remote_code=True)

# response = requests.post(url, headers=headers, json=payload)

@app.post("/process/")
async def process_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
        tmp_mp3.write(audio_bytes)
        mp3_path = tmp_mp3.name

    result = model_whisper.transcribe(mp3_path, language="pt")
    # result["text"]
    # processed_result = pipe(result["text"])
    
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]

    pipe(messages)
    
    return {"transcription": ''}