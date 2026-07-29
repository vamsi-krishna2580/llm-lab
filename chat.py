import os
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ============================================================
# Load Environment
# ============================================================

load_dotenv()

MODEL = os.getenv("MODEL", "Qwen/Qwen3-8B")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="LLM Lab API",
    version="1.0.0",
)

# ============================================================
# Request / Response Models
# ============================================================

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


# ============================================================
# GPU Info
# ============================================================

print("=" * 60)
print("Loading Model")
print("=" * 60)

print("Model :", MODEL)
print("Torch :", torch.__version__)
print("CUDA  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU   :", torch.cuda.get_device_name(0))
    print(
        "VRAM  : {:.2f} GB".format(
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        )
    )

print("=" * 60)

# ============================================================
# Load Tokenizer
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# ============================================================
# Quantization
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ============================================================
# Load Model
# ============================================================

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Model Loaded Successfully!")

# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
You are a helpful, friendly and knowledgeable AI assistant.

Give accurate and concise answers.

If you don't know something, say you don't know.

Do not make up facts.
"""

# ============================================================
# Helper Function
# ============================================================

def generate(prompt: str):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.05,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    return response

# ============================================================
# Routes
# ============================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "model": MODEL,
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    if not req.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    answer = generate(req.message)

    return ChatResponse(
        response=answer
    )


@app.get("/info")
def info():

    return {
        "model": MODEL,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
    }

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "chat:app",
        host=HOST,
        port=PORT,
        reload=False,
    )