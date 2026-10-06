"""
LLM-as-a-judge model for DeepEval.

DeepEval's metrics (Faithfulness, AnswerRelevancy, GEval) are judge-agnostic: they
build the grading prompts and call `generate(prompt, schema=...)` on whatever model
you hand them. This class is that model, on top of any backend in evals/llm.py
(Hugging Face, Amazon Bedrock, or the Claude CLI).

Design choices:
  - temperature 0: a judge should give the same verdict twice
  - structured output: when DeepEval passes a pydantic schema, we return a validated
    instance, so a malformed judge reply fails loudly instead of scoring silently
  - retries with backoff: a provider hiccup should not turn into a 0 score
  - usage counters: judge calls and tokens are part of what an eval run costs
"""
import json
import random
import re
import threading
import time

from deepeval.models import DeepEvalBaseLLM

from evals.config import JUDGE_MODEL
from evals.llm import complete


def extract_json(text: str):
    """Pull the first balanced JSON object/array out of a model reply."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError(f"no JSON in judge output: {text[:200]!r}")
    start = min(starts)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON in judge output")


class Judge(DeepEvalBaseLLM):
    def __init__(self, spec: str = JUDGE_MODEL, max_retries: int = 6):
        self.spec = spec
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        super().__init__(spec)

    def load_model(self):
        return self.spec

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            prompt += "\n\nReturn ONLY valid JSON. No prose, no markdown."
        last_err = None
        for attempt in range(self.max_retries):
            try:
                text, usage = complete(self.spec, [{"role": "user", "content": prompt}],
                                       max_tokens=1024, temperature=0.0)
                with self._lock:
                    self.calls += 1
                    self.input_tokens += usage.get("input_tokens", 0)
                    self.output_tokens += usage.get("output_tokens", 0)
                if schema is None:
                    return text
                return schema.model_validate(extract_json(text))
            except Exception as e:  # provider error or unparseable reply: retry
                last_err = e
                # exponential backoff with jitter; Bedrock throttles bursts from parallel workers
                time.sleep(min(30, 2 ** attempt) + random.random())
        raise RuntimeError(f"judge failed after {self.max_retries} attempts: {last_err}")

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema=schema)

    def get_model_name(self):
        return self.spec

    def usage(self) -> dict:
        return {"judge_calls": self.calls, "judge_input_tokens": self.input_tokens,
                "judge_output_tokens": self.output_tokens}
