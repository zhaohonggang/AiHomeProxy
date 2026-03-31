from flask import Flask, jsonify, request
from flasgger import Swagger
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import config

try:
    import config_local

    _config = config_local
except ImportError:
    _config = config

app = Flask(__name__)
swagger = Swagger(app)

MODEL_NAME = "Home-0.0.1"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", _config.GOOGLE_API_KEY)
GOOGLE_MODEL = "gemma-3-1b-it"

llm = ChatGoogleGenerativeAI(
    model=GOOGLE_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    convert_system_message_to_human=True,
)


@app.route("/v1/models", methods=["GET"])
def list_models():
    """
    List available models
    ---
    tags:
      - Models
    responses:
      200:
        description: List of available models
        schema:
          type: object
          properties:
            object:
              type: string
            data:
              type: array
              items:
                properties:
                  id:
                    type: string
                  object:
                    type: string
                  created:
                    type: integer
                  owned_by:
                    type: string
    """
    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": MODEL_NAME,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "local",
                },
                {
                    "id": "gemma-3-1b-it",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "google",
                    "provider": "google",
                    "rpm": 30,
                    "rpd": 14400,
                },
                {
                    "id": "gemini-2.0-flash",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "google",
                    "provider": "google",
                    "rpm": 30,
                    "rpd": 14400,
                },
            ],
        }
    )


@app.route("/v1/models/<model_id>", methods=["GET"])
def get_model(model_id):
    """
    Get model info
    ---
    tags:
      - Models
    parameters:
      - name: model_id
        in: path
        required: true
        type: string
    responses:
      200:
        description: Model info
    """
    return jsonify(
        {
            "id": model_id,
            "object": "model",
            "created": 1700000000,
            "owned_by": "local",
            "permission": [],
            "root": model_id,
            "parent": None,
        }
    )


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """
    Chat completions
    ---
    tags:
      - Completions
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - model
            - messages
          properties:
            model:
              type: string
              default: Home-0.0.1
            messages:
              type: array
              items:
                properties:
                  role:
                    type: string
                  content:
                    type: string
    responses:
      200:
        description: Chat completion response
    """
    data = request.json
    messages = data.get("messages", [])

    if not messages:
        return jsonify({"error": "messages is required"}), 400

    try:
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not content:
                continue
            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            elif role == "system":
                langchain_messages.append(SystemMessage(content=content))
            else:
                langchain_messages.append(HumanMessage(content=content))

        result = llm.invoke(langchain_messages)
        assistant_content = result.content

    except Exception as e:
        assistant_content = f"Error: {str(e)}"

    response = {
        "id": f"chatcmpl-{hash(str(messages)) % 1000000}",
        "object": "chat.completion",
        "created": 1700000000,
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": assistant_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    return jsonify(response)


@app.route("/v1/completions", methods=["POST"])
def completions():
    """
    Text completions
    ---
    tags:
      - Completions
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - model
            - prompt
          properties:
            model:
              type: string
              default: Home-0.0.1
            prompt:
              type: string
    responses:
      200:
        description: Text completion response
    """
    data = request.json

    # Extract prompt (for future use when we add real logic)
    prompt = data.get("prompt", "")

    # Build response in OpenAI format
    response = {
        "id": f"cmpl-{hash(prompt) % 1000000}",
        "object": "text_completion",
        "created": 1700000000,
        "model": MODEL_NAME,
        "choices": [
            {"text": "Hello, I am Home 0.0.1", "index": 0, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    return jsonify(response)


@app.route("/health", methods=["GET"])
def health():
    """
    Health check
    ---
    tags:
      - Health
    responses:
      200:
        description: Server health status
    """
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print(f"Starting Home 0.0.1 API server on port 5010...")
    app.run(host="0.0.0.0", port=5010, debug=True)
