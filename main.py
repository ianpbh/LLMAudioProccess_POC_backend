from fastapi import FastAPI, UploadFile, File
from pydub import AudioSegment
import speech_recognition as sr
import tempfile

app = FastAPI()

@app.post("/process/")
async def process_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
        tmp_mp3.write(audio_bytes)
        mp3_path = tmp_mp3.name

    wav_path = mp3_path.replace(".mp3", ".wav")
    AudioSegment.from_file(mp3_path).export(wav_path, format="wav")

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language="pt-BR")

    return {"transcription": text}