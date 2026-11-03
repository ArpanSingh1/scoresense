from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
from core.registry import OCRPipelineManager
os.environ["FLAGS_use_pir_api"] = "0"

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

app = FastAPI(title="Unified OCR Pipeline API")
manager = OCRPipelineManager()
UPLOAD_DIR = "./temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def health_check():
    return {"status": "healthy", "engines": list(manager.engines.keys())}

@app.post("/process/{engine_name}")
async def process_document(engine_name: str, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        result = manager.run(engine_name, file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)