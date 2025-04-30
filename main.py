from fastapi import FastAPI, File, UploadFile
import whisper
import shutil
import os
import audio2numpy as a2n
import numpy as np
import ast

app = FastAPI()
model = whisper.load_model("base")

@app.post("/process/")
async def process_audio(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"

    print(file)
    import tempfile
    byteszao = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(byteszao)
        tmp_path = tmp.name
    
    
    result = model.transcribe(tmp_path, language="pt", fp16=False, task="transcribe")
    # os.remove(temp_file_path)

    return {"text": result["text"]}

@app.get("/teste/")
async def process_text():
    return {"text": "relou"}