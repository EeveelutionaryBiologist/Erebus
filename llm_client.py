
from pathlib import Path
import json
import os


CONFIG_FILE = os.environ.get("EREBUS_CONFIG")

BASE_DIR = Path(__file__).resolve().parent
PROVIDERS = ["GOOGLE", "OPENAI", "ANTHROPIC"]

if not CONFIG_FILE:
    CONFIG_FILE = BASE_DIR / "config.json" 


def load_environment_dict() -> dict:
    # TODO: Decoding from encrypt later - right now irrelevant as on local machine
    with open(CONFIG_FILE) as json_file:
        data = json.load(json_file)

    return data


def obtain_client():
    config_dict = load_environment_dict()

    for provider in PROVIDERS:
        key_name = f"{provider}_API_KEY"
        api_key = os.environ.get(key_name)

        if not key_name:
            continue

        base_url = config_dict.get(provider).get("BASE_URL")
        model_name = config_dict.get(provider).get("MODEL_NAME")

        # TODO setup client
        pass

        # TODO Fallback to llama_cpp local Qwen
        pass

