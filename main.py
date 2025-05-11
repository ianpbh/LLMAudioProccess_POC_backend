import requests
from fastapi import FastAPI, UploadFile, File
# from pydub import AudioSegment
# import speech_recognition as sr
import tempfile
import whisper
from transformers import pipeline

app = FastAPI()
model = whisper.load_model("base")
# pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Zero", trust_remote_code=True)


# @app.post("/processold/")
# async def process_audio(file: UploadFile = File(...)):
#     audio_bytes = await file.read()

#     with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
#         tmp_mp3.write(audio_bytes)
#         mp3_path = tmp_mp3.name

#     wav_path = mp3_path.replace(".mp3", ".wav")
#     AudioSegment.from_file(mp3_path).export(wav_path, format="wav")

#     recognizer = sr.Recognizer()
#     with sr.AudioFile(wav_path) as source:
#         audio_data = recognizer.record(source)
#         text = recognizer.recognize_google(audio_data, language="pt-BR")

#     return {"transcription": text}


url = "http://localhost:1234/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}

# response = requests.post(url, headers=headers, json=payload)

@app.post("/process/")
async def process_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
        tmp_mp3.write(audio_bytes)
        mp3_path = tmp_mp3.name

    result = model.transcribe(mp3_path, language="pt")
    # processed_result = pipe(result["text"])
    payload = {
        "model": "deepseek-r1-distill-qwen-7b",
        "messages": [
            {
                "role": "system",
                "content": ""
            },
            { "role": "user", "content": result["text"]}
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False
    }
    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    print(data)
    return {"transcription": data["choices"][0]["message"]["content"]}