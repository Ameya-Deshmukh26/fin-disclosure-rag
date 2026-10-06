"""
Pluggable model backends for the generator and the judge.

A model is named by a spec string "provider:model":
  hf:meta-llama/Llama-3.1-8B-Instruct                  Hugging Face Inference Providers
  bedrock:us.meta.llama3-1-8b-instruct-v1:0             Amazon Bedrock (Converse API)
  bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0
  claude-cli:claude-sonnet-4-5                          Claude Code CLI on a subscription

Pick them with environment variables, no code change:
  EVAL_GENERATOR=bedrock:us.meta.llama3-1-8b-instruct-v1:0
  EVAL_JUDGE=bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0

Every backend returns (text, usage) so token counts end up in the run report.
"""
import json
import os
import shutil
import subprocess

from dotenv import load_dotenv

load_dotenv("hugging.env")

_clients = {}


def _hf(model, messages, max_tokens, temperature):
    from huggingface_hub import InferenceClient
    c = _clients.setdefault("hf", InferenceClient(token=os.environ["HF_TOKEN"]))
    r = c.chat_completion(messages=messages, model=model, max_tokens=max_tokens, temperature=temperature)
    u = getattr(r, "usage", None)
    usage = {"input_tokens": getattr(u, "prompt_tokens", 0) or 0,
             "output_tokens": getattr(u, "completion_tokens", 0) or 0}
    return r.choices[0].message.content, usage


def _bedrock(model, messages, max_tokens, temperature):
    import boto3
    c = _clients.setdefault("bedrock", boto3.client(
        "bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1")))
    system = [{"text": m["content"]} for m in messages if m["role"] == "system"]
    convo = [{"role": m["role"], "content": [{"text": m["content"]}]}
             for m in messages if m["role"] != "system"]
    kwargs = {"modelId": model, "messages": convo,
              "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature}}
    if system:
        kwargs["system"] = system
    r = c.converse(**kwargs)
    text = "".join(b.get("text", "") for b in r["output"]["message"]["content"])
    u = r.get("usage", {})
    return text, {"input_tokens": u.get("inputTokens", 0), "output_tokens": u.get("outputTokens", 0)}


def _claude_cli(model, messages, max_tokens, temperature):
    exe = shutil.which("claude") or os.path.expandvars(r"%APPDATA%\npm\claude.cmd")
    prompt = "\n\n".join(
        (f"[SYSTEM INSTRUCTIONS]\n{m['content']}" if m["role"] == "system" else m["content"])
        for m in messages)
    p = subprocess.run([exe, "-p", "--model", model, "--output-format", "json", "--max-turns", "1"],
                       input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=300)
    d = json.loads(p.stdout)
    if d.get("is_error"):
        raise RuntimeError(f"claude-cli: {d.get('result')}")
    u = d.get("usage", {})
    return d.get("result", ""), {"input_tokens": u.get("input_tokens", 0),
                                 "output_tokens": u.get("output_tokens", 0)}


BACKENDS = {"hf": _hf, "bedrock": _bedrock, "claude-cli": _claude_cli}


def complete(spec: str, messages: list[dict], max_tokens: int = 1024,
             temperature: float = 0.0) -> tuple[str, dict]:
    provider, model = spec.split(":", 1)
    return BACKENDS[provider](model, messages, max_tokens, temperature)
