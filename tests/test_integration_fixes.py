"""Fixes made while integrating the uploaded-document graphs (16 November 2026).

Timeout setting, .env.local, explicit graph building, and mixed selections.
No real model is called.
"""
import json

import pytest
from fastapi.testclient import TestClient

from rag_app import llm
from rag_app.api import create_app
from rag_app.models import RetrievalConfig, Settings
from rag_app.service import ROOT, RagService
from tests.test_doc_graph import MERIDIAN, fake_extractor

LLM_NAMES = ('LLM_URL', 'LLM_API_KEY', 'LLM_MODEL', 'LLM_TIMEOUT', 'GOOGLE_API_KEY', 'GOOGLE_MODEL')


def stub_answer(question, evidence):
    return json.dumps(dict(answer='ok', answerable=True, citations=[evidence[0]['id']]))


# ---------- 1. timeout

@pytest.mark.parametrize('value, expected', [(None, 45), ('300', 300), ('12.5', 12.5), ('abc', 45), ('0', 45), ('-3', 45)])
def test_llm_timeout_setting(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv('LLM_TIMEOUT', raising=False)
    else:
        monkeypatch.setenv('LLM_TIMEOUT', value)
    assert llm.timeout_seconds() == expected


def test_the_model_client_uses_the_configured_timeout(monkeypatch):
    monkeypatch.setenv('LLM_TIMEOUT', '300')
    config = llm.LLMConfig('http://localhost:11434/v1', '', 'local-model')
    assert llm.client(config).timeout == 300
    assert llm.client(config, 20).timeout == 20  # an explicit value still wins (Kestrel graph CLI)


# ---------- 2. .env.local

def test_env_local_overrides_env_and_the_real_environment_overrides_both(monkeypatch, tmp_path):
    for name in LLM_NAMES:  # set then delete, so pytest restores the variable after the test
        monkeypatch.setenv(name, 'x')
        monkeypatch.delenv(name)
    (tmp_path / '.env').write_text('LLM_URL=https://from-env.example/v1\nLLM_MODEL=env-model\nLLM_API_KEY=env-key\n')
    (tmp_path / '.env.local').write_text('LLM_URL=http://localhost:11434/v1\nLLM_MODEL=local-model\n')
    monkeypatch.setattr(llm, 'ROOT', tmp_path)
    assert tuple(llm.load_config()) == ('http://localhost:11434/v1', 'env-key', 'local-model')
    monkeypatch.setenv('LLM_MODEL', 'shell-model')
    assert llm.load_config().model == 'shell-model'


# ---------- 4. building a graph is an explicit action

def test_upload_sends_nothing_to_the_model_until_build_graph_is_clicked(tmp_path, monkeypatch):
    monkeypatch.setenv('RAG_GRAPH_AUTO', '0')
    seen = []
    client = TestClient(create_app(RagService(tmp_path, generator=stub_answer, graph_extractor=fake_extractor(seen))))
    result = client.post('/api/documents', files={'file': ('meridian.txt', MERIDIAN.encode(), 'text/plain')}).json()
    assert result['graph']['status'] == 'manual' and 'Build graph' in result['graph']['reason']
    assert seen == [] and client.get('/api/graphs').json() == {}
    built = client.post(f"/api/documents/{result['id']}/graph").json()
    assert built['built'] and built['concepts'] == 4 and len(seen) == 1
    trace = client.get('/api/traces/' + built['trace_id']).json()
    assert trace['kind'] == 'graph' and trace['outcome'] == 'built'


def test_building_on_upload_is_off_unless_asked_for(monkeypatch):
    monkeypatch.setattr('dotenv.load_dotenv', lambda *a, **k: None)
    monkeypatch.delenv('RAG_GRAPH_AUTO', raising=False)
    assert not RagService._auto_graph_enabled()
    for value, expected in (('1', True), ('true', True), ('0', False), ('', False), ('no', False)):
        monkeypatch.setenv('RAG_GRAPH_AUTO', value)
        assert RagService._auto_graph_enabled() is expected


# ---------- 3. mixed selection

def test_an_unrelated_uploaded_graph_does_not_change_a_kestrel_answer(tmp_path):
    """Found in the integration review: the uploaded graph was seeded by loose word matching and
    pushed the passage naming the owning team out of the top 5."""
    service = RagService(tmp_path, generator=stub_answer, graph_extractor=fake_extractor())
    kestrel = [service.upload(p.name, p.read_bytes())['id'] for p in sorted((ROOT / 'docs' / 'kestrel').glob('*.md'))]
    other = service.upload('meridian.txt', (MERIDIAN + 'The project lead reports to the team that owns the budget.\n').encode())['id']
    service.build_graph(other)
    # "solar project" also matches words of the uploaded graph, as an unrelated file can in real use
    question = 'Error KST-5031 happened on the scheduled reports of the solar project. Which team should support contact?'
    config = RetrievalConfig(retrievers=['graph'])
    alone = service.retrieve_configured(question, kestrel, Settings(), config)
    mixed = service.retrieve_configured(question, kestrel + [other], Settings(), config)
    assert [e['label'] for e in mixed['graph_trace']['matched_entities']] == ['KST-5031']
    assert [r['id'] for r in mixed['results']] == [r['id'] for r in alone['results']]
    assert all(r['document_id'] != other for r in mixed['results'])
