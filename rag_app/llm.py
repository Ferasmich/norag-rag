"""Which language model the app talks to, read from .env.

LLM_URL      base URL of any OpenAI-compatible chat endpoint
LLM_API_KEY  its key (may stay empty for a local server such as Ollama)
LLM_MODEL    the model name
LLM_TIMEOUT  seconds to wait for one model call (default 45; a slow local model needs more)

Settings are read from the real environment first, then .env.local, then .env.

Google Gemini stays the default: when none of the three is set, the older
GOOGLE_API_KEY / GOOGLE_MODEL variables and the Google endpoint are used,
so an existing .env keeps working unchanged.

GOOGLE_API_KEY is a Google credential. It is only ever sent to Google's own
endpoint (https, generativelanguage.googleapis.com). With any other LLM_URL,
only LLM_API_KEY is used; without it the request carries no real key, which is
what a local server such as Ollama expects.
"""
import os
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse

import dotenv

ROOT = Path(__file__).resolve().parents[1]
GOOGLE_URL = 'https://generativelanguage.googleapis.com/v1beta/openai/'
GOOGLE_MODEL = 'gemini-3.5-flash-lite'
GOOGLE_HOST = 'generativelanguage.googleapis.com'
PLACEHOLDERS = {'', 'your_google_api_key_here', 'your_api_key_here'}
DEFAULT_TIMEOUT = 45


def is_google(url):
    """True only for Google's own endpoint over https (a look-alike host does not count)."""
    try:
        parts = urlparse(url)
    except ValueError:
        return False
    return parts.scheme == 'https' and parts.hostname == GOOGLE_HOST


class LLMConfig(NamedTuple):
    url: str
    key: str
    model: str

    @property
    def usable(self):
        """A key is needed for Google, not for a custom LLM_URL (local servers often take none)."""
        return self.key not in PLACEHOLDERS or not is_google(self.url)


def load_config():
    # dotenv never replaces a variable that is already set, so the first source wins:
    # real environment, then .env.local (personal overrides), then .env.
    dotenv.load_dotenv(ROOT / '.env.local')
    dotenv.load_dotenv(ROOT / '.env')
    url = (os.getenv('LLM_URL') or '').strip() or GOOGLE_URL
    key = (os.getenv('LLM_API_KEY') or '').strip()
    if key in PLACEHOLDERS:
        key = ''
        # The older Google variable may only travel to Google, never to another provider.
        if is_google(url):
            key = (os.getenv('GOOGLE_API_KEY') or '').strip()
    model = (os.getenv('LLM_MODEL') or os.getenv('GOOGLE_MODEL') or '').strip() or GOOGLE_MODEL
    return LLMConfig(url, key, model)


MISSING_KEY = 'Set LLM_API_KEY (or GOOGLE_API_KEY) in .env to generate answers.'


def hide_secrets(text):
    """Replace the configured API keys in a text before it is stored or shown."""
    for name in ('LLM_API_KEY', 'GOOGLE_API_KEY'):
        value = (os.getenv(name) or '').strip()
        if len(value) >= 8 and value not in PLACEHOLDERS:
            text = text.replace(value, '[hidden key]')
    return text


def timeout_seconds():
    """LLM_TIMEOUT in seconds. Falls back to the default when missing or not a positive number."""
    try:
        value = float(os.getenv('LLM_TIMEOUT') or DEFAULT_TIMEOUT)
    except ValueError:
        return DEFAULT_TIMEOUT
    return value if value > 0 else DEFAULT_TIMEOUT


def client(config, timeout=None):
    from openai import OpenAI
    timeout = timeout or timeout_seconds()
    key = config.key if config.key not in PLACEHOLDERS else 'not-needed'
    return OpenAI(api_key=key, base_url=config.url, timeout=timeout, max_retries=0)


def complete_json(config, system, user, timeout=None):
    """One JSON-mode chat call. Returns (text, usage) with usage = model and token counts."""
    response = client(config, timeout).chat.completions.create(
        model=config.model, temperature=0, response_format={'type': 'json_object'},
        messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}])
    usage = response.usage
    return (response.choices[0].message.content or '',
            dict(model=config.model,
                 input_tokens=usage.prompt_tokens if usage else None,
                 output_tokens=usage.completion_tokens if usage else None))
