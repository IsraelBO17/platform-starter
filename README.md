# Modular AI Integration Platform — starter and Lab 1 solution

Reference code for the Foundation Models and Large Language Models class notes, Lab 1 and the Voice Notes Assistant project: the starting point of the Modular AI Integration Platform. Every full code listing in the Class Notes is a file in this folder.

**For instructors.** Give students `tests/test_platform.py` and the three scripts the Lab refers to (`scripts/compare_prompts.py`, `scripts/compare_models.py`, `scripts/generate_image.py`) at the start of the Lab. Share the whole folder as the Project starter after the Lab is submitted.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/), which manages the Python version, the virtual environment and the dependencies:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS / Linux
  ```
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
  ```
  You do not need to install Python yourself: uv downloads Python 3.12 (pinned in `.python-version`) if it isn't already on your machine.
- A free API key from [Groq](https://console.groq.com/keys) or [Google AI Studio (Gemini)](https://aistudio.google.com/apikey).
- Optional: PostgreSQL, for the stateful conversation endpoint (see [Conversation history](#conversation-history)).
- Optional: an AWS account, for Bedrock, Polly, Transcribe and the Lambda deployment.

## Quick start

```bash
git clone https://github.com/IsraelBO17/platform-starter.git
cd platform-starter

uv sync                         # creates .venv and installs everything from uv.lock
cp .env.example .env            # then open .env and paste your GROQ_API_KEY (or GEMINI_API_KEY)

uv run pytest -q                # 16 tests, no network or API key needed
USE_MEMORY_HISTORY=1 uv run uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs> for the interactive API docs.

On Windows (PowerShell), set the variable on its own line first:

```powershell
$env:USE_MEMORY_HISTORY = "1"
uv run uvicorn app.main:app --reload
```

`uv run` always uses the project's `.venv`, so you don't need to activate it. If you prefer to, run `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows) and drop the `uv run` prefix.

## Configuration

All settings come from environment variables or the `.env` file (see `app/config.py`). Only the key for the provider you pick needs a value.

| Variable | Default in `.env.example` | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `groq` | `groq`, `gemini`, `openai`, `xai`, `ollama` or `bedrock` |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Model name as the provider spells it |
| `GROQ_API_KEY` / `GEMINI_API_KEY` | empty | Key for the chosen provider (`OPENAI_API_KEY` and `XAI_API_KEY` also work) |
| `APP_API_KEY` | not set | If set, every request must send it in an `X-API-Key` header. Leave it unset for local development. |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/platform` | PostgreSQL for conversation history |
| `USE_MEMORY_HISTORY` | not set | Set to `1` to keep conversation history in memory instead of PostgreSQL |
| `AWS_PROFILE` / `AWS_REGION` | `univaciti` / `us-east-1` | Only for Bedrock, Polly, Transcribe and image generation on AWS. Change the profile to one in your `~/.aws/config`. |
| `HF_TOKEN` | empty | Hugging Face token for `scripts/generate_image.py hf` |

Example: to use Gemini instead of Groq, set `LLM_PROVIDER=gemini`, `GEMINI_API_KEY=...` and a Gemini model name in `LLM_MODEL`.

To run against a local [Ollama](https://ollama.com) model, set `LLM_PROVIDER=ollama` and `LLM_MODEL` to a model you've pulled. No key is needed.

## Conversation history

`POST /v1/conversations/{id}/messages` stores history server-side. The app picks the store at startup:

- **`USE_MEMORY_HISTORY=1`**: history lives in memory and disappears on restart. Use this if you don't have PostgreSQL.
- **Otherwise**: the app connects to `DATABASE_URL` at startup and creates its table automatically. If PostgreSQL isn't reachable, the server fails to start.

The quickest way to get a matching PostgreSQL is Docker:

```bash
docker run -d --name platform-db -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=platform postgres:16
uv run uvicorn app.main:app --reload
```

## Try the API

```bash
# Stateless chat: the client sends the whole history
curl -s localhost:8000/v1/chat -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in Yoruba"}]}'

# Stateful chat: the server keeps the history
curl -s localhost:8000/v1/conversations/demo/messages -H 'Content-Type: application/json' \
  -d '{"message":"My name is Ada."}'

# Streaming (Server-Sent Events)
curl -N localhost:8000/v1/chat/stream -H 'Content-Type: application/json' \
  -d '{"message":"Write a haiku about Lagos traffic"}'

# Structured output
curl -s localhost:8000/v1/triage -H 'Content-Type: application/json' \
  -d '{"text":"I was charged twice for my card this month"}'
```

If you set `APP_API_KEY`, add `-H "X-API-Key: <your key>"` to each request.

## Scripts

Run scripts as modules from the project root so `app` can be imported:

```bash
uv run python -m scripts.first_call            # optional args: temperature max_tokens
uv run python -m scripts.ttft
uv run python -m scripts.try_triage
uv run python -m scripts.compare_prompts
uv run python -m scripts.compare_models
uv run python -m scripts.generate_image hf "a lighthouse at dawn"        # needs HF_TOKEN
uv run python -m scripts.generate_image bedrock "a lighthouse at dawn"   # needs AWS credentials
```

Some scripts need packages that aren't installed by default. `uv run --with` adds them for that run only:

```bash
uv run --with sentence-transformers python -m scripts.embeddings_demo
uv run --with torch --with diffusers --with transformers --with accelerate \
  python -m scripts.generate_image local "a lighthouse at dawn"
```

## Local speech (optional)

The open-source Whisper and Piper models are in the optional `speech` dependency group:

```bash
uv sync --group speech
```

The Piper voice files are expected in `voices/`. If `voices/en_US-lessac-medium.onnx` is missing, download it:

```bash
uv run --group speech python -m piper.download_voices en_US-lessac-medium --download-dir voices
```

Then:

```bash
mkdir -p outputs
uv run --group speech python -m scripts.speech_local speak "Your order has shipped." outputs/notice.wav
uv run --group speech python -m scripts.speech_local listen outputs/notice.wav
```

Whisper downloads its model on first use.

## Deploying to AWS Lambda

`build_lambda.sh` builds `function.zip` from `requirements.txt` for Python 3.12 on x86_64. `run.sh` is the start script the Lambda Web Adapter runs. Deploy the zip in the console behind a Function URL.

```bash
./build_lambda.sh
```

`requirements.txt` holds the Lambda's runtime dependencies. If you add a runtime dependency with `uv add`, add it to `requirements.txt` as well.

On Lambda, any environment variable ending in `_PARAM` names an SSM parameter to load. For example, `GROQ_API_KEY_PARAM=/platform/groq-key` fills `GROQ_API_KEY` at startup (see `app/secrets.py`).

## Layout

- `app/` — the platform: provider interface and adapters, services, FastAPI endpoints
- `scripts/` — first_call, ttft, try_triage, compare_prompts, compare_models, generate_image, embeddings_demo, speech_local
- `speech_pipeline/` — `local_engine.py` (open-source Whisper + Piper on your machine) and `handler.py` (S3 → Transcribe / Polly Lambda handler, deployed in the console)
- `voices/` — Piper voice model used by `local_engine.py`
- `run.sh`, `build_lambda.sh` — start script for the Lambda Web Adapter and the build for `function.zip`
- `tests/` — offline tests using a fake provider and botocore Stubbers

## Troubleshooting

- **`uv run uvicorn` fails with a traceback from another Python (e.g. pyenv)**: run `uv sync` first. Without it, `uvicorn` isn't in `.venv` and uv falls back to whatever is on your `PATH`.
- **`warning: VIRTUAL_ENV=... does not match the project environment`**: another virtualenv is active in your shell. It's harmless, since uv uses `.venv` anyway. Run `deactivate` or `unset VIRTUAL_ENV` to silence it.
- **`RuntimeError: Set GROQ_API_KEY to use the groq provider.`**: `.env` is missing or the key for your `LLM_PROVIDER` is empty.
- **Server won't start and the error mentions a connection to port 5432**: PostgreSQL isn't running. Start it (see above) or set `USE_MEMORY_HISTORY=1`.
- **`401 Missing or invalid X-API-Key header`**: `APP_API_KEY` is set in your environment or `.env`. Send the header or unset it.
