import json
import os
import time
import uuid
from threading import Thread
from typing import List, Literal, Optional, Union

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def openai_error(message: str, err_type: str = "invalid_request_error", param: Optional[str] = None, code: Optional[str] = None, status_code: int = 400):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": err_type,
                "param": param,
                "code": code,
            }
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return openai_error(message=str(exc.detail), status_code=exc.status_code)


from fastapi.exceptions import RequestValidationError


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    param = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    return openai_error(
        message=first.get("msg", "Invalid request"),
        param=param or None,
        status_code=422,
    )


# ============================================================
# Optional API Key Auth (set API_KEY in .env to enable)
# ============================================================

API_KEY = os.getenv("API_KEY")


def check_api_key(request: Request):
    if not API_KEY:
        return
    auth = request.headers.get("authorization", "")
    token = auth.replace("Bearer ", "").strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Incorrect API key provided.")

# ============================================================
# OpenAI-Compatible Request / Response Models
# ============================================================

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=MODEL)
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 512
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls", "function_call"]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class CompletionRequest(BaseModel):
    model: str = Field(default=MODEL)
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    n: Optional[int] = 1


class CompletionChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[dict] = None
    finish_reason: Literal["stop", "length", "content_filter"]


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


# ---- Responses API (/v1/responses) models ----

class ResponseInputItem(BaseModel):
    role: Literal["system", "user", "assistant", "developer"]
    content: Union[str, List[dict]]


class ResponsesRequest(BaseModel):
    model: str = Field(default=MODEL)
    input: Union[str, List[ResponseInputItem], List[dict]]
    instructions: Optional[str] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_output_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    previous_response_id: Optional[str] = None
    store: Optional[bool] = True


class ResponseOutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: List[dict] = []


class ResponseOutputMessage(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    status: Literal["completed", "incomplete"] = "completed"
    content: List[ResponseOutputText]


class ResponsesUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ResponsesResponse(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int
    model: str
    status: Literal["completed", "incomplete", "failed"] = "completed"
    output: List[ResponseOutputMessage]
    output_text: str
    usage: ResponsesUsage


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
# System Prompt (fallback, only used if no system message is provided)
# ============================================================

DEFAULT_SYSTEM_PROMPT = """
You are a helpful, friendly and knowledgeable AI assistant.

Give accurate and concise answers.

If you don't know something, say you don't know.

Do not make up facts.
"""

# ============================================================
# Helper Function
# ============================================================

def build_inputs(messages: List[ChatMessage]):
    # Ensure there is a system message; otherwise fall back to default
    chat_messages = [m.dict() for m in messages]
    if not any(m["role"] == "system" for m in chat_messages):
        chat_messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + chat_messages

    inputs = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    return inputs


def apply_stop_sequences(text: str, stop: Optional[Union[str, List[str]]]):
    if not stop:
        return text, False
    stop_list = [stop] if isinstance(stop, str) else stop
    for s in stop_list:
        if s and s in text:
            return text.split(s)[0], True
    return text, False


def generate(
    messages: List[ChatMessage],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
    stop: Optional[Union[str, List[str]]] = None,
):
    inputs = build_inputs(messages)
    prompt_tokens = inputs["input_ids"].shape[1]

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=50,
            repetition_penalty=1.05,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][prompt_tokens:]
    completion_tokens = generated_ids.shape[0]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    finish_reason = "length" if completion_tokens >= max_tokens else "stop"

    response, hit_stop = apply_stop_sequences(response, stop)
    if hit_stop:
        finish_reason = "stop"

    return response, prompt_tokens, completion_tokens, finish_reason


def generate_stream(
    messages: List[ChatMessage],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
    stop: Optional[Union[str, List[str]]] = None,
):
    """Yields text chunks as they are generated, using a background thread."""
    inputs = build_inputs(messages)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=50,
        repetition_penalty=1.05,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    stop_list = [stop] if isinstance(stop, str) else (stop or [])
    buffer = ""
    stopped = False

    for token_text in streamer:
        if stopped:
            break
        buffer += token_text
        for s in stop_list:
            if s and s in buffer:
                buffer = buffer.split(s)[0]
                stopped = True
                break
        yield token_text if not stopped else buffer
        if stopped:
            break

    thread.join()

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


def _models_payload():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.get("/v1/models")
def list_models(request: Request):
    check_api_key(request)
    return _models_payload()


# Alias: some clients (and the user's own probe) hit /models without the /v1 prefix
@app.get("/models")
def list_models_alias(request: Request):
    check_api_key(request)
    return _models_payload()


@app.get("/v1/models/{model_id}")
def retrieve_model(model_id: str, request: Request):
    check_api_key(request)
    if model_id != MODEL:
        raise HTTPException(status_code=404, detail=f"The model '{model_id}' does not exist.")
    return {
        "id": MODEL,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "local",
    }


def _chat_completion_chunk(chunk_id: str, model_name: str, delta: dict, finish_reason: Optional[str]):
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


def stream_chat_response(req: ChatCompletionRequest):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"

    # First chunk announces the assistant role, as OpenAI does
    yield _chat_completion_chunk(chunk_id, req.model, {"role": "assistant"}, None)

    finished_early = False
    for token_text in generate_stream(
        messages=req.messages,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        stop=req.stop,
    ):
        if token_text:
            yield _chat_completion_chunk(chunk_id, req.model, {"content": token_text}, None)

    yield _chat_completion_chunk(chunk_id, req.model, {}, "stop")
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, request: Request):
    check_api_key(request)
    return _handle_chat_completions(req)


# Alias without the /v1 prefix
@app.post("/chat/completions")
def chat_completions_alias(req: ChatCompletionRequest, request: Request):
    check_api_key(request)
    return _handle_chat_completions(req)


def _handle_chat_completions(req: ChatCompletionRequest):

    if not req.messages:
        raise HTTPException(
            status_code=400,
            detail="messages cannot be empty."
        )

    if req.model != MODEL:
        # OpenAI-compatible servers generally accept any model string but
        # flag a mismatch here since only one model is actually loaded.
        pass

    if req.stream:
        return StreamingResponse(
            stream_chat_response(req),
            media_type="text/event-stream",
        )

    answer, prompt_tokens, completion_tokens, finish_reason = generate(
        messages=req.messages,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        stop=req.stop,
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=answer),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# ============================================================
# Legacy /v1/completions (text-in, text-out) endpoint
# ============================================================

@app.post("/v1/completions")
def completions(req: CompletionRequest, request: Request):
    check_api_key(request)
    return _handle_completions(req)


# Alias without the /v1 prefix
@app.post("/completions")
def completions_alias(req: CompletionRequest, request: Request):
    check_api_key(request)
    return _handle_completions(req)


def _handle_completions(req: CompletionRequest):

    prompts = [req.prompt] if isinstance(req.prompt, str) else req.prompt
    if not prompts:
        raise HTTPException(status_code=400, detail="prompt cannot be empty.")

    if req.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming is not supported on /v1/completions; use /v1/chat/completions."
        )

    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, p in enumerate(prompts):
        answer, prompt_tokens, completion_tokens, finish_reason = generate(
            messages=[ChatMessage(role="user", content=p)],
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            stop=req.stop,
        )
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        choices.append(
            CompletionChoice(
                index=i,
                text=answer,
                finish_reason=finish_reason,
            )
        )

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model,
        choices=choices,
        usage=Usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        ),
    )


# ============================================================
# Responses API (/v1/responses) — newer OpenAI endpoint
# ============================================================

def _responses_input_to_messages(req: ResponsesRequest) -> List[ChatMessage]:
    messages: List[ChatMessage] = []

    if req.instructions:
        messages.append(ChatMessage(role="system", content=req.instructions))

    if isinstance(req.input, str):
        messages.append(ChatMessage(role="user", content=req.input))
        return messages

    for item in req.input:
        if isinstance(item, ResponseInputItem):
            role = item.role
            content = item.content
        else:
            role = item.get("role", "user")
            content = item.get("content", "")

        # content can be a plain string or a list of
        # {"type": "input_text"/"output_text", "text": "..."} blocks
        if isinstance(content, list):
            text = "".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        else:
            text = content

        # Responses API uses "developer" in place of "system" in some SDKs
        normalized_role = "system" if role == "developer" else role
        if normalized_role not in ("system", "user", "assistant"):
            normalized_role = "user"

        messages.append(ChatMessage(role=normalized_role, content=text))

    return messages


def _build_responses_object(req: ResponsesRequest, answer: str, input_tokens: int, output_tokens: int, status: str = "completed"):
    resp_id = f"resp_{uuid.uuid4().hex}"
    msg_id = f"msg_{uuid.uuid4().hex}"

    return ResponsesResponse(
        id=resp_id,
        created_at=int(time.time()),
        model=req.model,
        status=status,
        output=[
            ResponseOutputMessage(
                id=msg_id,
                content=[ResponseOutputText(text=answer)],
            )
        ],
        output_text=answer,
        usage=ResponsesUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


def stream_responses_response(req: ResponsesRequest):
    messages = _responses_input_to_messages(req)
    resp_id = f"resp_{uuid.uuid4().hex}"
    msg_id = f"msg_{uuid.uuid4().hex}"

    def event(event_type: str, data: dict):
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    created_payload = {
        "type": "response.created",
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": int(time.time()),
            "model": req.model,
            "status": "in_progress",
        },
    }
    yield event("response.created", created_payload)

    full_text = ""
    for token_text in generate_stream(
        messages=messages,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_output_tokens,
    ):
        if token_text:
            full_text += token_text
            yield event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": msg_id,
                    "delta": token_text,
                },
            )

    completed_payload = {
        "type": "response.completed",
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": int(time.time()),
            "model": req.model,
            "status": "completed",
            "output": [
                {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": full_text, "annotations": []}],
                }
            ],
            "output_text": full_text,
        },
    }
    yield event("response.completed", completed_payload)


def _handle_responses(req: ResponsesRequest):
    if req.stream:
        return StreamingResponse(
            stream_responses_response(req),
            media_type="text/event-stream",
        )

    messages = _responses_input_to_messages(req)
    answer, prompt_tokens, completion_tokens, _finish_reason = generate(
        messages=messages,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_output_tokens,
    )

    return _build_responses_object(req, answer, prompt_tokens, completion_tokens)


@app.post("/v1/responses", response_model=None)
def responses_v1(req: ResponsesRequest, request: Request):
    check_api_key(request)
    return _handle_responses(req)


# Alias without the /v1 prefix, since some clients/proxies omit it
@app.post("/responses", response_model=None)
def responses_alias(req: ResponsesRequest, request: Request):
    check_api_key(request)
    return _handle_responses(req)
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