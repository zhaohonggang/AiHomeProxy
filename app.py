from flask import Flask, jsonify, request
from flasgger import Swagger
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import config
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    import config_local

    _config = config_local
except ImportError:
    _config = config

app = Flask(__name__)
swagger = Swagger(app)

MODEL_NAME = "Home-0.0.1"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", _config.GOOGLE_API_KEY)
from datetime import datetime, timedelta
import json


def get_google_models():
    with open("google_models.json", "r") as f:
        return json.load(f)


_current_model_index = 0
_model_usage = {}


def get_google_model(models=None, skip_on_limit=True):
    global _current_model_index
    if models is None:
        models = get_google_models()

    start_index = _current_model_index
    attempts = 0

    while attempts < len(models):
        model_id = models[_current_model_index % len(models)]["id"]
        rpm = get_model_rpm(model_id)
        usage = get_usage_count_this_minute(model_id, count_only=True)

        if not skip_on_limit or usage < rpm - 2:
            _current_model_index += 1
            return model_id

        _current_model_index += 1
        attempts += 1

    return None


def get_model_rpm(model_id):
    models = get_google_models()
    for m in models:
        if m["id"] == model_id:
            return m.get("rpm", 0)
    return 0


def get_usage_count_this_minute(model_id, count_only=False):
    global _model_usage
    now = datetime.now()
    if model_id not in _model_usage:
        _model_usage[model_id] = []
    _model_usage[model_id] = [
        ts for ts in _model_usage[model_id] if now - ts < timedelta(minutes=1)
    ]
    if not count_only:
        _model_usage[model_id].append(now)
    return len(_model_usage[model_id])


GOOGLE_MODEL = get_google_model()

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
    selected_model = data.get("model", "Home-0.0.1")
    logger.info(
        f"Request | Model: {selected_model} | Messages: {len(messages)} | Data: {data}"
    )

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
                content = content.split("\n\n[")[0]
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                content = content.split("\n\n[")[0]
                langchain_messages.append(AIMessage(content=content))
            elif role == "system":
                langchain_messages.append(SystemMessage(content=content))
            else:
                langchain_messages.append(HumanMessage(content=content))

        if selected_model == "Home-0.0.1":
            selected_model = get_google_model()

        assistant_content = None
        last_error = None
        tried_models = set()
        max_retries = 5

        while assistant_content is None and len(tried_models) < max_retries:
            if selected_model is None:
                assistant_content = (
                    "All models are at capacity. Please try again later."
                )
                break

            if selected_model in tried_models:
                selected_model = get_google_model(skip_on_limit=True)
                if selected_model is None or selected_model in tried_models:
                    assistant_content = "All models failed. Please try again later."
                    break

            tried_models.add(selected_model)

            llm_temp = ChatGoogleGenerativeAI(
                model=selected_model,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.7,
                convert_system_message_to_human=True,
            )
            try:
                result = llm_temp.invoke(langchain_messages, timeout=30)
                assistant_content = result.content
                logger.info(
                    f"Chat success | Model: {selected_model} | Prompt: {messages[-1].get('content', '')[:100]} | Response: {assistant_content[:200]}"
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Chat failed | Model: {selected_model} | Error: {last_error}"
                )
                selected_model = get_google_model(skip_on_limit=True)

    except Exception as e:
        assistant_content = f"Error: {str(e)}"

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if selected_model:
        usage_count = get_usage_count_this_minute(selected_model, count_only=False)
        model_rpm = get_model_rpm(selected_model)
        metadata = f"[{current_time}] Model: {selected_model} | Usage: {usage_count}/{model_rpm} per minute"
    else:
        metadata = f"[{current_time}] Model: N/A"

    response = {
        "id": f"chatcmpl-{hash(str(messages)) % 1000000}",
        "object": "chat.completion",
        "created": 1700000000,
        "model": selected_model if selected_model else "N/A",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"{assistant_content}\n\n{metadata}",
                },
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
