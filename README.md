# LLM Lab API

Simple FastAPI backend powered by Qwen.

---

# Setup

```bash
git clone https://github.com/YOUR_USERNAME/llm-lab.git

cd llm-lab

bash install.sh
```

Server:

```
http://localhost:8000
```

Swagger UI:

```
http://localhost:8000/docs
```

ReDoc:

```
http://localhost:8000/redoc
```

---

# Endpoints

## GET /

Returns server status.

### Request

```
GET /
```

### Response

```json
{
    "status": "running",
    "model": "Qwen/Qwen3-8B"
}
```

---

## GET /health

Returns health information.

### Request

```
GET /health
```

### Response

```json
{
    "status": "healthy",
    "cuda": true,
    "gpu": "Tesla T4"
}
```

---

## GET /info

Returns model information.

### Request

```
GET /info
```

### Response

```json
{
    "model": "Qwen/Qwen3-8B",
    "torch": "2.x.x",
    "cuda": true
}
```

---

## POST /chat

Chat with the model.

### Request

```
POST /chat
Content-Type: application/json
```

Body

```json
{
    "message": "What is Artificial Intelligence?"
}
```

### Response

```json
{
    "response": "Artificial Intelligence (AI) is..."
}
```

---

# Sample Requests

## cURL

```bash
curl -X POST http://localhost:8000/chat \
-H "Content-Type: application/json" \
-d '{
    "message":"Explain Machine Learning"
}'
```

---

## Python

```python
import requests

r = requests.post(
    "http://localhost:8000/chat",
    json={
        "message":"Hello"
    }
)

print(r.json())
```

---

## JavaScript

```javascript
const response = await fetch("http://localhost:8000/chat", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify({
        message: "Hello"
    })
});

console.log(await response.json());
```

---

## Java

```java
HttpClient client = HttpClient.newHttpClient();

String json = """
{
    "message":"Hello"
}
""";

HttpRequest request = HttpRequest.newBuilder()
        .uri(URI.create("http://localhost:8000/chat"))
        .header("Content-Type","application/json")
        .POST(HttpRequest.BodyPublishers.ofString(json))
        .build();

HttpResponse<String> response =
        client.send(request, HttpResponse.BodyHandlers.ofString());

System.out.println(response.body());
```

---

# Response Codes

| Code | Meaning |
|------|---------|
|200|Success|
|400|Invalid Request|
|500|Internal Server Error|

---

# Project Structure

```
llm-lab/
│
├── chat.py
├── install.sh
├── requirements.txt
└── README.md
```

---

# Future Features

- Conversation History
- Session Management
- Streaming Responses
- Tool Calling
- SQL Database
- Memory
- RAG
- MCP
- Fine-tuning Support
