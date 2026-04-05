# AGENTS.md - AI Home Proxy

This project is a Flask API that acts as a proxy to wrap Google Gemini models.

## Project Overview

- **Purpose**: AI Proxy API - Flask API to create home models to serve other API callers. The home models run as proxies by calling existing LLM models, then transfer and modify the result as the return of home models to the caller.
- **Tech Stack**: Python, Flask, LangChain, Google Gemini
- **API Structure**: REST API with `/v1/chat/completions`, `/v1/models`, etc.

## Behavior

Follow Standard OpenClaw patterns:
- Read `SOUL.md` for persona
- Read `USER.md` for user context
- Use memory files for continuity

## Rules

1. **Confirm before exec**: Don't execute external commands without explicit approval
2. **Error handling**: Always use proper error handling and logging
3. **Keep it simple**: Prefer clean, simple solutions over complex ones
4. **Follow patterns**: Follow existing code patterns in the project
5. **Massive changes**: When AI wants to delete more than 5 lines, explain the plan and ask explicit approval before making changes
6. **Logging format**: When adding log statements, always wrap the log description with `********************...description...********************` at both beginning and end
7. **Model naming**: Never expose Google model names to callers. Always return defined Model Names (e.g., "Home-0.0.1") in API responses
8. **Check AGENTS.md**: Always scan AGENTS.md before making changes. If any change requests conflict with AGENTS.md rules, explain the conflict and ask for confirmation before proceeding.
9. **Review and test**: Always scan the code twice to ensure correctness. After applying changes, verify the modified code works as expected (run tests or verify functionality).
10. **File logging**: When logging, always write to both console and a log file under `log/` subfolder with date in filename (e.g., `app_20260405.log`).
11. **Test with start.ps1**: When testing the Flask API, use `start.ps1` to open a new cmd window so you can see the output and verify functionality.

## Key Files

- `app.py` - Main Flask application
- `config.py` - Default configuration
- `config_local.py` - Local overrides (API keys)
- `google_models.json` - Available Google models
- `requirements.txt` - Python dependencies

## API Endpoints

- `GET /v1/models` - List available models
- `GET /v1/models/<model_id>` - Get model info
- `POST /v1/chat/completions` - Chat completions
- `POST /v1/completions` - Text completions
- `GET /health` - Health check