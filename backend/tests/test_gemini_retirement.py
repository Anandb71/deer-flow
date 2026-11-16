"""Offline wire checks for the native Gemini setup provider."""

import asyncio
import importlib
import importlib.util
import json
import socket
from pathlib import Path

import httpx
import pytest
from google import genai
from google.genai import types
from wizard.providers import LLM_PROVIDERS


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def reject_connection(*args, **kwargs):
        raise AssertionError("This test must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", reject_connection)


def _provider_class():
    provider = next(p for p in LLM_PROVIDERS if p.name == "google")
    module_name, class_name = provider.use.split(":")
    if module_name.startswith("deerflow."):
        # Load the real adapter without importing unrelated harness services.
        path = Path(__file__).parents[1] / "packages" / "harness" / Path(*module_name.split("."))
        spec = importlib.util.spec_from_file_location(module_name, path.with_suffix(".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    return getattr(module, class_name)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("model_name,n", [("gemini-3.8-flash", 1), ("gemini-3.8-flash", 2), ("gemini-2.5-flash-lite", 1)])
def test_native_gemini_request_controls(asynchronous, model_name, n):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}]})

    transport = httpx.MockTransport(respond)
    client = genai.Client(api_key="offline-test-key", http_options=types.HttpOptions(client_args={"transport": transport}, async_client_args={"transport": transport}))
    model = _provider_class()(model=model_name, api_key="offline-test-key", n=n, temperature=0.2, top_p=0.8, top_k=10)
    # LangChain rebuilds client during validation, so replace it afterwards.
    model.client.close()
    asyncio.run(model.client.aio.aclose())
    model.client = client
    assert model.client is client
    try:
        if n > 1:
            with pytest.raises(ValueError, match="only one candidate"):
                asyncio.run(model.ainvoke("hello")) if asynchronous else model.invoke("hello")
            assert requests == []
            return
        answer = asyncio.run(model.ainvoke("hello")) if asynchronous else model.invoke("hello")
        assert answer.content == "ok"
        assert len(requests) == 1
        config = requests[0]["generationConfig"]
        if model_name == "gemini-3.8-flash":
            assert not {"candidateCount", "temperature", "topP", "topK"}.intersection(config)
        else:
            assert config["candidateCount"] == 1
            assert config["temperature"] == 0.2
            assert config["topP"] == 0.8
            assert config["topK"] == 10
    finally:
        client.close()
        asyncio.run(client.aio.aclose())
