"""
Unified LLM client for Erebus.

Provider selection (first match wins):
  1. LOCAL_MODEL: true in config.json  → local llama_cpp Qwen (explicit override)
  2. GOOGLE_API_KEY env var + GOOGLE.MODEL_NAME in config → Google Gemini (OpenAI-compatible)
  3. OPENAI_API_KEY + OPENAI.MODEL_NAME → OpenAI
  4. ANTHROPIC_API_KEY + ANTHROPIC.MODEL_NAME → Anthropic (via OpenAI-compatible proxy)
  5. OLLAMA.BASE_URL + OLLAMA.MODEL_NAME in config → Ollama (no key needed)
  6. Fallback → local llama_cpp Qwen

Usage:
  load_llm_client()     # once at startup
  get_llm_client()      # everywhere else
"""

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = Path(os.environ.get("EREBUS_CONFIG", BASE_DIR / "config.json"))

_KEYED_PROVIDERS = ["GOOGLE", "OPENAI", "ANTHROPIC"]

_client = None


def _load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return json.load(f)


class _LocalBackend:
    """Grammar-constrained JSON inference via llama_cpp — fully offline."""

    def __init__(self, config: dict):
        self._config = config
        self._llm = None

    def load(self):
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download, snapshot_download

        much_ram = self._config.get("MUCH_RAM", False)
        use_gpu = self._config.get("USE_GPU", False)
        model_dir = BASE_DIR / "Embedding"

        if much_ram:
            model_path = model_dir / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
            if not model_path.exists():
                print("[LLM] Downloading Qwen2.5-7B...")
                snapshot_download(
                    repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
                    local_dir=model_dir,
                    allow_patterns=["qwen2.5-7b-instruct-q4_k_m*"],
                )
        else:
            model_path = model_dir / "qwen2.5-3b-instruct-q4_k_m.gguf"
            if not model_path.exists():
                print("[LLM] Downloading Qwen2.5-3B...")
                hf_hub_download(
                    repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                    filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                    local_dir=model_dir,
                )

        print("[LLM] Loading local Qwen model into RAM...")
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_gpu_layers=-1 if use_gpu else 0,
            use_mlock=True,
            verbose=False,
            chat_format="chatml",
        )

    def chat_json(self, messages: list[dict], schema: dict, temperature: float = 0.1) -> str:
        response = self._llm.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object", "schema": schema},
            temperature=temperature,
        )
        return response["choices"][0]["message"]["content"]

    def chat_text(self, messages: list[dict], temperature: float = 0.3) -> str:
        response = self._llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
        )
        return response["choices"][0]["message"]["content"]


class _OpenAICompatibleBackend:
    """Any OpenAI-compatible endpoint: Google Gemini, OpenAI, Ollama, or proxy."""

    def __init__(self, base_url: str | None, api_key: str, model: str):
        from openai import OpenAI
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._openai = OpenAI(**kwargs)
        self._model = model

    def load(self):
        pass  # no local resources to acquire

    def chat_json(self, messages: list[dict], schema: dict, temperature: float = 0.1) -> str:
        # json_schema passes the exact Pydantic schema to the API so the model knows
        # the required field names. strict is omitted (defaults to false) because
        # Pydantic schemas have optional fields that aren't in "required", which
        # OpenAI strict mode rejects client-side.
        response = self._openai.chat.completions.create(
            model=self._model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema},
            },
            temperature=temperature,
        )
        return response.choices[0].message.content

    def chat_text(self, messages: list[dict], temperature: float = 0.3) -> str:
        response = self._openai.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content


def _select_backend(config: dict) -> _LocalBackend | _OpenAICompatibleBackend:
    if config.get("LOCAL_MODEL"):
        return _LocalBackend(config)

    for provider in _KEYED_PROVIDERS:
        provider_conf = config.get(provider, {})
        model = provider_conf.get("MODEL_NAME", "")
        if not model:
            continue
        api_key = os.environ.get(f"{provider}_API_KEY", "")
        if not api_key:
            continue
        base_url = provider_conf.get("BASE_URL") or None
        return _OpenAICompatibleBackend(base_url=base_url, api_key=api_key, model=model)

    # Ollama: no API key required, just base_url + model_name in config
    ollama_conf = config.get("OLLAMA", {})
    ollama_url = ollama_conf.get("BASE_URL", "")
    ollama_model = ollama_conf.get("MODEL_NAME", "")
    if ollama_url and ollama_model:
        return _OpenAICompatibleBackend(base_url=ollama_url, api_key="ollama", model=ollama_model)

    return _LocalBackend(config)


def load_llm_client():
    """Initialize the global LLM backend. Called once at server startup."""
    global _client
    config = _load_config()
    backend = _select_backend(config)
    backend.load()
    _client = backend


def get_llm_client() -> _LocalBackend | _OpenAICompatibleBackend:
    """Return the initialized LLM backend. Raises if load_llm_client() was not called."""
    if _client is None:
        raise RuntimeError("LLM client not initialized — call load_llm_client() first.")
    return _client
