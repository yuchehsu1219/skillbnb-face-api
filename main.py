import os, numpy as np, cv2
from fastapi import FastAPI, UploadFile, HTTPException, Header
from insightface.app import FaceAnalysis

face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(640,640))

app = FastAPI()
API_KEY = os.environ.get("API_KEY", "")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/embed")
async def embed(file: UploadFile, x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "unauthorized")
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "invalid image")
    faces = face_app.get(img)
    if not faces:
        raise HTTPException(422, "no face detected")
    f = max(faces, key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    w = int(f.bbox[2]-f.bbox[0]); h = int(f.bbox[3]-f.bbox[1])
    if w < 80 or h < 80:
        raise HTTPException(422, "face too small")
    return {
        "embedding": f.normed_embedding.tolist(),
        "gender": "male" if f.sex=='M' else "female",
        "face_size": [w, h],
    }
