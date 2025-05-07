from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import base64
import time

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face API details
HF_API_KEY = "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"
HF_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

class TextPrompt(BaseModel):
    prompt: str

@app.post("/generate-image")
async def generate_image(prompt: TextPrompt):
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "image/png"
    }
    payload = {"inputs": prompt.prompt}
    max_retries = 3
    for attempt in range(max_retries):
        response = requests.post(HF_URL, headers=headers, json=payload)
        print("Hugging Face API status:", response.status_code)
        print("Hugging Face API headers:", response.headers)
        try:
            print("Hugging Face API body:", response.json())
        except Exception:
            print("Hugging Face API body (raw):", response.content)
        if response.status_code == 504:
            print("504 Gateway Timeout, retrying...")
            time.sleep(5)  # Wait 5 seconds before retrying
            continue
        break
    # If the response is JSON, it's likely an error
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            error = response.json()
            raise HTTPException(status_code=500, detail=error.get("error", "Failed to generate image"))
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to generate image (unknown error)")
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to generate image")
    # The response is an image (bytes)
    image_bytes = response.content
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return {"image_base64": image_base64} 