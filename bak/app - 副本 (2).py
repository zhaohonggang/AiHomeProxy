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
            tools:
              type: array
            tool_choice:
              type: object
            response_format:
              type: object
              description: JSON schema for structured output
    responses:
      200:
        description: Chat completion response
    """
    data = request.json
    messages = data.get("messages", [])
    selected_model = data.get("model", "Home-0.0.1")
    tools = data.get("tools", [])
    tool_choice = data.get("tool_choice", None)
    response_format = data.get("response_format", None)
    logger.info(
        f"Request | Model: {selected_model} | Messages: {len(messages)} | Tools: {len(tools)} | ResponseFormat: {response_format is not None} | Data: {data}"
    )

    if not messages:
        return jsonify({"error": "messages is required"}), 400

    try:
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        from langchain_core.utils.function_calling import convert_to_openai_function

        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list) and len(content) > 0:
                content = content[0].get("text", "")
            elif not isinstance(content, str):
                content = str(content)

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
        result = None

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

            try:
                tool_bindings = {}
                if tools:
                    processed_tools = []
                    for t in tools:
                        if isinstance(t, dict) and "function" in t:
                            func = t["function"]
                            if "title" not in t:
                                t["title"] = func.get("name", "function")
                            processed_tools.append(t)
                        else:
                            processed_tools.append(t)
                    tool_bindings["functions"] = [
                        convert_to_openai_function(t) for t in processed_tools
                    ]
                    tool_bindings["model_kwargs"] = {
                        "functions": tool_bindings.pop("functions")
                    }
                    if tool_choice:
                        tool_bindings["function_call"] = (
                            tool_choice.get("name")
                            if isinstance(tool_choice, dict)
                            else tool_choice
                        )

                if response_format:
                    if "json_schema" in response_format or "schema" in response_format:
                        tool_bindings["response_schema"] = response_format.get(
                            "json_schema", response_format.get("schema")
                        )

                llm_temp = ChatGoogleGenerativeAI(
                    model=selected_model,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.7,
                    convert_system_message_to_human=True,
                    **tool_bindings,
                )
                result = llm_temp.invoke(langchain_messages, timeout=30)
                logger.info(f"LLM invoke done | Model: {selected_model}")

                if isinstance(result.content, list):
                    for item in result.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            item["text"] = item.get("text", "")
                elif isinstance(result.content, str):
                    result.content = result.content

                assistant_content = result.content
                break
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Chat failed | Model: {selected_model} | Error: {last_error}"
                )
                selected_model = get_google_model(skip_on_limit=True)

    except Exception as e:
        assistant_content = f"Error: {str(e)}"

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    message_content = assistant_content
    if isinstance(message_content, list):
        text_parts = []
        for item in message_content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "thinking":
                    continue
                else:
                    text_parts.append(str(item))
        message_content = "".join(text_parts)
    elif not isinstance(message_content, str):
        message_content = str(message_content)

    message = {
        "role": "assistant",
        "content": "message_content",
    }

    tool_calls_result = None
    if result is not None:
        tool_calls_result = getattr(result, "tool_calls", None)
        if tool_calls_result and isinstance(tool_calls_result, list):
            message["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{hash(str(tc)) % 1000000}"),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", tc.get("function", {}).get("name", "")),
                        "arguments": json.dumps(
                            tc.get(
                                "arguments", tc.get("function", {}).get("arguments", {})
                            )
                        ),
                    },
                }
                for tc in tool_calls_result
            ]

    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if result is not None:
        usage_metadata = getattr(result, "usage_metadata", None)
        if usage_metadata:
            usage_info = {
                "prompt_tokens": usage_metadata.get("input_tokens", 0),
                "completion_tokens": usage_metadata.get("output_tokens", 0),
                "total_tokens": usage_metadata.get("total_tokens", 0),
            }

    response = {
        "id": "chatcmpl-unique-test-001",
        "object": "chat.completion",
        "created": 1712300000,
        "model": selected_model if selected_model else "N/A",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls_result else "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
    }

    logger.info(f"Full response: {response}")
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
