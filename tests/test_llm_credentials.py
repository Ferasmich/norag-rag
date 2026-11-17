"""Which key is sent to which LLM endpoint, and keys never leak into traces or messages.

GOOGLE_API_KEY may only reach Google's own endpoint. The OpenAI client is replaced by a
fake that records what it was given, so no request leaves the machine.
"""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rag_app import llm
from rag_app.api import create_app
from rag_app.service import PipelineError, RagService
from rag_app.tracing import Trace, TraceStore

GOOGLE_SECRET = 'AIza-test-GOOGLE-SECRET-0123456789'
OTHER_SECRET = 'sk-test-OTHER-PROVIDER-SECRET-98765'
NAMES = ('LLM_URL', 'LLM_API_KEY', 'LLM_MODEL', 'LLM_TIMEOUT', 'GOOGLE_API_KEY', 'GOOGLE_MODEL')
TEXT = 'The water tank holds 1.8 litres. ' * 12


@pytest.fixture
def env(monkeypatch):
    """No .env file is read; every LLM variable starts unset."""
    monkeypatch.setattr('dotenv.load_dotenv', lambda *a, **k: None)
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def provider(monkeypatch):
    """A fake OpenAI client. `provider.sent` lists the api_key and base_url of every client made."""
    state = SimpleNamespace(sent=[], fail_with=None)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            state.sent.append(dict(api_key=kwargs['api_key'], base_url=str(kwargs['base_url'])))
            self.key = kwargs['api_key']
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            if state.fail_with:
                # Some providers quote the key they received in their error message.
                raise RuntimeError(state.fail_with.format(key=self.key))
            evidence = json.loads(kwargs['messages'][1]['content'])['evidence']
            answer = dict(answer='It holds 1.8 litres.', answerable=True, citations=[evidence[0]['id']])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer)))],
                                   usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))
    monkeypatch.setattr('openai.OpenAI', FakeOpenAI)
    return state


def ask(tmp_path, question='How much water does the tank hold?'):
    client = TestClient(create_app(RagService(tmp_path)))
    doc = client.post('/api/documents', files={'file': ('tank.txt', TEXT.encode(), 'text/plain')}).json()['id']
    response = client.post('/api/ask', json={'document_ids': [doc], 'question': question,
                                             'retrieval': {'retrievers': ['bm25']}})
    stored = (tmp_path / 'traces' / 'traces.jsonl').read_text(encoding='utf-8')
    return response, stored


# ---------- Google (default, backward compatible)

def test_google_default_uses_the_google_key(env, provider):
    env.setenv('GOOGLE_API_KEY', GOOGLE_SECRET)
    config = llm.load_config()
    assert config.url == llm.GOOGLE_URL and config.key == GOOGLE_SECRET and config.usable
    llm.client(config)
    assert provider.sent[-1] == dict(api_key=GOOGLE_SECRET, base_url=llm.GOOGLE_URL)


@pytest.mark.parametrize('url', ['https://generativelanguage.googleapis.com/v1beta/openai/',
                                 'https://generativelanguage.googleapis.com/v1beta/openai'])
def test_google_url_written_explicitly_still_uses_the_google_key(env, url):
    env.setenv('LLM_URL', url)
    env.setenv('GOOGLE_API_KEY', GOOGLE_SECRET)
    assert llm.load_config().key == GOOGLE_SECRET


def test_the_env_example_placeholder_does_not_hide_the_google_key(env):
    env.setenv('LLM_API_KEY', 'your_api_key_here')  # copied from .env.example and not edited
    env.setenv('GOOGLE_API_KEY', GOOGLE_SECRET)
    assert llm.load_config().key == GOOGLE_SECRET


def test_llm_api_key_wins_over_the_google_key_for_google(env):
    env.setenv('LLM_API_KEY', 'new-google-key-123')
    env.setenv('GOOGLE_API_KEY', GOOGLE_SECRET)
    assert llm.load_config().key == 'new-google-key-123'


# ---------- another hosted provider

def test_another_provider_never_receives_the_google_key(env, provider, tmp_path):
    env.setenv('LLM_URL', 'https://api.openai.com/v1')
    env.setenv('LLM_MODEL', 'gpt-4o-mini')
    env.setenv('GOOGLE_API_KEY', GOOGLE_SECRET)
    assert llm.load_config().key == ''
    response, stored = ask(tmp_path)
    assert response.status_code == 200
    assert provider.sent and all(GOOGLE_SECRET not in sent['api_key'] for sent in provider.sent)
    assert provider.sent[-1]['base_url'].startswith('https://api.openai.com/v1')


def test_another_provider_gets_its_own_key(env, provider):
    env.setenv('LLM_URL', 'https://api.openai.com/v1')
    env.setenv('LLM_API_KEY', OTHER_SECRET)
    env.setenv('GOOGLE_API_KEY', GOOGLE_SECRET)
    llm.client(llm.load_config())
    assert provider.sent[-1]['api_key'] == OTHER_SECRET


@pytest.mark.parametrize('url', [
    'https://generativelanguage.googleapis.com.attacker.example/v1',   # look-alike host
    'https://attacker.example/generativelanguage.googleapis.com/v1',   # Google name in the path
    'http://generativelanguage.googleapis.com/v1beta/openai/',         # plain http
])
def test_look_alike_google_urls_do_not_receive_the_google_key(env, provider, url):
    env.setenv('LLM_URL', url)
    env.setenv('GOOGLE_API_KEY', GOOGLE_SECRET)
    config = llm.load_config()
    assert config.key == ''
    llm.client(config)
    assert provider.sent[-1]['api_key'] != GOOGLE_SECRET


# ---------- local Ollama

def test_local_ollama_works_without_any_key(env, provider, tmp_path):
    env.setenv('LLM_URL', 'http://localhost:11434/v1')
    env.setenv('LLM_MODEL', 'llama3.1')
    config = llm.load_config()
    assert config.usable and config.key == ''
    response, _ = ask(tmp_path)
    assert response.status_code == 200 and response.json()['answerable']
    assert provider.sent[-1]['api_key'] == 'not-needed'


def test_local_ollama_does_not_receive_the_google_key_from_an_old_env(env, provider):
    env.setenv('LLM_URL', 'http://localhost:11434/v1')
    env.setenv('GOOGLE_API_KEY', GOOGLE_SECRET)  # still in .env from the Gemini days
    llm.client(llm.load_config())
    assert provider.sent[-1]['api_key'] == 'not-needed'


# ---------- missing credentials

@pytest.mark.parametrize('settings', [{}, {'LLM_API_KEY': 'your_api_key_here'},
                                      {'GOOGLE_API_KEY': 'your_google_api_key_here'}])
def test_google_without_a_real_key_is_refused_before_any_request(env, provider, tmp_path, settings):
    for name, value in settings.items():
        env.setenv(name, value)
    assert not llm.load_config().usable
    with pytest.raises(PipelineError, match='LLM_API_KEY'):
        RagService(tmp_path).generate('q', [{'id': 'a', 'text': 't'}])
    assert provider.sent == []  # nothing was sent anywhere


def test_missing_key_is_reported_through_the_api_and_the_trace(env, provider, tmp_path):
    response, stored = ask(tmp_path)
    assert response.status_code == 503 and 'LLM_API_KEY' in response.json()['error']
    assert '"outcome": "error"' in stored and provider.sent == []


# ---------- keys never reach traces or error messages

@pytest.mark.parametrize('url, key_name, secret', [
    ('https://api.openai.com/v1', 'LLM_API_KEY', OTHER_SECRET),
    ('', 'GOOGLE_API_KEY', GOOGLE_SECRET),
])
def test_a_provider_error_that_quotes_the_key_does_not_leak_it(env, provider, tmp_path, url, key_name, secret):
    env.setenv('LLM_URL', url)
    env.setenv(key_name, secret)
    env.setenv('GOOGLE_API_KEY', GOOGLE_SECRET)  # also present, must not leak either
    provider.fail_with = '401 Incorrect API key provided: {key}'
    response, stored = ask(tmp_path)
    assert response.status_code == 503
    for value in (OTHER_SECRET, GOOGLE_SECRET):
        assert value not in response.text and value not in stored


def test_a_successful_answer_leaves_no_key_in_the_trace(env, provider, tmp_path):
    env.setenv('LLM_URL', 'https://api.openai.com/v1')
    env.setenv('LLM_API_KEY', OTHER_SECRET)
    env.setenv('GOOGLE_API_KEY', GOOGLE_SECRET)
    response, stored = ask(tmp_path)
    assert response.status_code == 200
    assert OTHER_SECRET not in response.text and OTHER_SECRET not in stored and GOOGLE_SECRET not in stored


def test_a_failed_graph_build_leaves_no_key_in_the_trace_or_the_answer(env, provider, tmp_path):
    env.setenv('LLM_URL', 'https://api.openai.com/v1')
    env.setenv('LLM_API_KEY', OTHER_SECRET)
    provider.fail_with = '401 Incorrect API key provided: {key}'
    client = TestClient(create_app(RagService(tmp_path)))
    doc = client.post('/api/documents', files={'file': ('tank.txt', TEXT.encode(), 'text/plain')}).json()['id']
    response = client.post(f'/api/documents/{doc}/graph')
    assert response.status_code == 503 and OTHER_SECRET not in response.text
    assert OTHER_SECRET not in (tmp_path / 'traces' / 'traces.jsonl').read_text(encoding='utf-8')


def test_the_trace_store_hides_a_key_even_if_a_message_contains_one(env, tmp_path):
    env.setenv('LLM_API_KEY', OTHER_SECRET)
    trace = Trace('question')
    trace.error = f'provider said: bad key {OTHER_SECRET}'
    TraceStore(tmp_path).save(trace)
    stored = (tmp_path / 'traces.jsonl').read_text(encoding='utf-8')
    assert OTHER_SECRET not in stored and '[hidden key]' in stored


def test_hide_secrets_ignores_placeholders_and_short_values(env):
    env.setenv('LLM_API_KEY', 'your_api_key_here')
    env.setenv('GOOGLE_API_KEY', 'abc')
    assert llm.hide_secrets('your_api_key_here abc') == 'your_api_key_here abc'
