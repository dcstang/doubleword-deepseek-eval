"""
Model wrappers for DeepSeek (via Doubleword OpenAI-compatible API) and Gemini.

Both classes expose:
  - chat(prompt, system, max_tokens) → ModelResponse
  - batch_chat(questions, system, max_tokens) → List[ModelResponse]
  - chat_with_tools(prompt, tool_names, system, max_tokens) → ModelResponse

DeepSeek batch uses the OpenAI Batch API with completion_window="1h" (Doubleword
delayed mode) — roughly 50% cheaper than synchronous calls.

Gemini batch uses the google-genai batches API when available (requires the model
to support it); otherwise falls back to concurrent asyncio requests which still
parallelises the wall-clock cost without the 1-hr queue.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    result: Any


@dataclass
class ModelResponse:
    model_name: str
    final_text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    batch_mode: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "final_text": self.final_text,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments, "result": tc.result}
                for tc in self.tool_calls
            ],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_s": round(self.latency_s, 2),
            "batch_mode": self.batch_mode,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# DeepSeek via Doubleword (OpenAI-compatible)
# ---------------------------------------------------------------------------

class DeepSeekModel:
    """
    Wraps the Doubleword OpenAI-compatible endpoint for DeepSeek.

    Supports two modes:
      - Synchronous: standard chat completions
      - Batch (delayed, 1h): OpenAI Batch API — cheaper but asynchronous
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    # ------------------------------------------------------------------
    # Synchronous single-turn chat
    # ------------------------------------------------------------------

    def chat(
        self,
        prompt: str,
        system: str = "You are a knowledgeable medical assistant.",
        max_tokens: int = 2048,
    ) -> ModelResponse:
        start = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            elapsed = time.time() - start
            choice = resp.choices[0]
            return ModelResponse(
                model_name=self.model,
                final_text=choice.message.content or "",
                input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                output_tokens=resp.usage.completion_tokens if resp.usage else 0,
                latency_s=elapsed,
                batch_mode=False,
            )
        except Exception as exc:
            return ModelResponse(
                model_name=self.model,
                final_text="",
                latency_s=time.time() - start,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Batch / delayed mode (OpenAI Batch API, completion_window="1h")
    # ------------------------------------------------------------------

    def batch_chat(
        self,
        questions: List[Dict],   # each dict: {"id": str, "prompt": str}
        system: str = "You are a knowledgeable medical assistant.",
        max_tokens: int = 2048,
        poll_interval_s: int = 30,
    ) -> List[ModelResponse]:
        """
        Submit all questions as one OpenAI batch job with completion_window="1h".
        This is Doubleword's delayed/cheap mode — typically ~50% off list price.
        Blocks until the batch completes (up to ~1 hour) then returns all responses.
        """
        import io

        print(f"    [batch] Submitting {len(questions)} requests to Doubleword delayed API...")
        start = time.time()

        # Build JSONL request file
        batch_requests = [
            {
                "custom_id": q["id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": q["prompt"]},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                },
            }
            for q in questions
        ]
        jsonl_bytes = "\n".join(json.dumps(r) for r in batch_requests).encode()

        try:
            # Upload the request file
            upload = self.client.files.create(
                file=("batch_requests.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
                purpose="batch",
            )
            print(f"    [batch] File uploaded: {upload.id}")

            # Create the batch job
            batch = self.client.batches.create(
                input_file_id=upload.id,
                endpoint="/v1/chat/completions",
                completion_window="1h",
            )
            print(f"    [batch] Batch job created: {batch.id} (status: {batch.status})")

            # Poll until done
            terminal = {"completed", "failed", "cancelled", "expired"}
            while batch.status not in terminal:
                time.sleep(poll_interval_s)
                batch = self.client.batches.retrieve(batch.id)
                counts = batch.request_counts
                print(
                    f"    [batch] {batch.id} — {batch.status} "
                    f"(done={counts.completed} failed={counts.failed} total={counts.total})"
                )

            elapsed = time.time() - start
            print(f"    [batch] Finished in {elapsed/60:.1f} min, status={batch.status}")

            if batch.status != "completed":
                err = f"Batch ended with status={batch.status}"
                return [
                    ModelResponse(model_name=self.model, final_text="", error=err, batch_mode=True)
                    for _ in questions
                ]

            # Fetch and parse output
            raw = self.client.files.content(batch.output_file_id).content
            results_by_id: Dict[str, dict] = {}
            for line in raw.decode().splitlines():
                if line.strip():
                    item = json.loads(line)
                    results_by_id[item["custom_id"]] = item

            responses = []
            for q in questions:
                item = results_by_id.get(q["id"], {})
                resp_body = item.get("response", {})
                if resp_body.get("status_code") == 200:
                    body = resp_body["body"]
                    choice = body["choices"][0]
                    usage = body.get("usage", {})
                    responses.append(
                        ModelResponse(
                            model_name=self.model,
                            final_text=choice["message"]["content"] or "",
                            input_tokens=usage.get("prompt_tokens", 0),
                            output_tokens=usage.get("completion_tokens", 0),
                            latency_s=elapsed / len(questions),
                            batch_mode=True,
                        )
                    )
                else:
                    responses.append(
                        ModelResponse(
                            model_name=self.model,
                            final_text="",
                            latency_s=elapsed / len(questions),
                            batch_mode=True,
                            error=str(item.get("error") or resp_body),
                        )
                    )
            return responses

        except Exception as exc:
            # Batch API not supported by this endpoint — fall back to sequential sync calls
            print(f"    [batch] Batch API unavailable ({exc}), falling back to sequential sync...")
            responses = []
            for q in questions:
                responses.append(self.chat(q["prompt"], system=system, max_tokens=max_tokens))
            return responses

    # ------------------------------------------------------------------
    # Tool-calling (synchronous multi-turn loop)
    # ------------------------------------------------------------------

    def chat_with_tools(
        self,
        prompt: str,
        tool_names: List[str],
        system: str = "You are a knowledgeable medical assistant. Use the provided tools to assess the patient and then give your clinical recommendation.",
        max_tokens: int = 2048,
        max_rounds: int = 10,
    ) -> ModelResponse:
        from tools import get_openai_tools, execute_tool

        openai_tools = get_openai_tools(tool_names)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        tool_calls_log: List[ToolCall] = []
        total_input = total_output = 0
        start = time.time()

        try:
            for _ in range(max_rounds):
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                if resp.usage:
                    total_input += resp.usage.prompt_tokens
                    total_output += resp.usage.completion_tokens

                msg = resp.choices[0].message
                messages.append(msg.model_dump(exclude_unset=True))

                if not msg.tool_calls:
                    return ModelResponse(
                        model_name=self.model,
                        final_text=msg.content or "",
                        tool_calls=tool_calls_log,
                        input_tokens=total_input,
                        output_tokens=total_output,
                        latency_s=time.time() - start,
                    )

                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result = execute_tool(tc.function.name, args)
                    tool_calls_log.append(ToolCall(name=tc.function.name, arguments=args, result=result))
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)}
                    )

            return ModelResponse(
                model_name=self.model,
                final_text="[max tool call rounds exceeded]",
                tool_calls=tool_calls_log,
                input_tokens=total_input,
                output_tokens=total_output,
                latency_s=time.time() - start,
            )

        except Exception as exc:
            return ModelResponse(
                model_name=self.model,
                final_text="",
                tool_calls=tool_calls_log,
                latency_s=time.time() - start,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Gemini (via google-genai SDK)
# ---------------------------------------------------------------------------

class GeminiModel:
    """
    Wraps the Google Gemini API.

    Batch mode:
      - Attempts google-genai `client.batches` API first (inline JSONL, no GCS needed)
      - Falls back to concurrent asyncio requests (parallelised, but not "delayed-cheap")
    """

    def __init__(self, api_key: str, model: str):
        import google.genai as genai
        self.client = genai.Client(api_key=api_key)
        self.model = model

    # ------------------------------------------------------------------
    # Synchronous single-turn chat
    # ------------------------------------------------------------------

    def chat(
        self,
        prompt: str,
        system: str = "You are a knowledgeable medical assistant.",
        max_tokens: int = 2048,
    ) -> ModelResponse:
        from google.genai import types

        start = time.time()
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                    temperature=0.1,
                ),
            )
            elapsed = time.time() - start
            usage = resp.usage_metadata
            return ModelResponse(
                model_name=self.model,
                final_text=resp.text or "",
                input_tokens=usage.prompt_token_count if usage else 0,
                output_tokens=usage.candidates_token_count if usage else 0,
                latency_s=elapsed,
                batch_mode=False,
            )
        except Exception as exc:
            return ModelResponse(
                model_name=self.model,
                final_text="",
                latency_s=time.time() - start,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------

    def batch_chat(
        self,
        questions: List[Dict],   # each dict: {"id": str, "prompt": str}
        system: str = "You are a knowledgeable medical assistant.",
        max_tokens: int = 2048,
        poll_interval_s: int = 30,
    ) -> List[ModelResponse]:
        """
        Try google-genai batch API first (supported on models like gemini-2.5-flash-preview).
        Falls back to concurrent asyncio if batch API is unavailable for this model/key.

        Note: Google's batch API pricing discount (typically ~50%) applies only when the
        batch job is accepted; fallback concurrent mode uses standard pricing.
        """
        # Try native batch first
        try:
            return self._batch_via_genai_api(questions, system, max_tokens, poll_interval_s)
        except Exception as exc:
            print(f"    [batch] Gemini batch API unavailable ({exc}), falling back to concurrent async...")
            return self._batch_via_asyncio(questions, system, max_tokens)

    def _batch_via_genai_api(
        self,
        questions: List[Dict],
        system: str,
        max_tokens: int,
        poll_interval_s: int,
    ) -> List[ModelResponse]:
        from google.genai import types

        print(f"    [batch] Submitting {len(questions)} requests to Gemini batch API...")
        start = time.time()

        # Build inline requests
        inline_requests = [
            types.EmbedContentRequest(  # placeholder — actual type depends on SDK version
                model=self.model,
                content=types.Content(
                    role="user",
                    parts=[types.Part(text=f"[SYSTEM]: {system}\n\n{q['prompt']}")],
                ),
            )
            for q in questions
        ]

        # Use generate_content batch if available in this SDK version
        batch_job = self.client.batches.create(
            model=self.model,
            src=inline_requests,
            config=types.CreateBatchJobConfig(
                display_name="medical-eval-batch",
            ),
        )
        print(f"    [batch] Gemini batch job: {batch_job.name} (state: {batch_job.state})")

        terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}
        while str(batch_job.state) not in terminal and not any(
            t in str(batch_job.state) for t in ["SUCCEEDED", "FAILED", "CANCELLED"]
        ):
            time.sleep(poll_interval_s)
            batch_job = self.client.batches.get(name=batch_job.name)
            print(f"    [batch] {batch_job.name} — {batch_job.state}")

        elapsed = time.time() - start
        if "SUCCEEDED" not in str(batch_job.state):
            raise RuntimeError(f"Gemini batch ended with state={batch_job.state}")

        # Collect results
        responses = []
        results = list(self.client.batches.list_job_results(name=batch_job.name))
        results_by_idx = {i: r for i, r in enumerate(results)}

        for i, q in enumerate(questions):
            r = results_by_idx.get(i)
            if r and hasattr(r, "response") and r.response:
                text = r.response.text or ""
                usage = r.response.usage_metadata
                responses.append(
                    ModelResponse(
                        model_name=self.model,
                        final_text=text,
                        input_tokens=usage.prompt_token_count if usage else 0,
                        output_tokens=usage.candidates_token_count if usage else 0,
                        latency_s=elapsed / len(questions),
                        batch_mode=True,
                    )
                )
            else:
                responses.append(
                    ModelResponse(
                        model_name=self.model,
                        final_text="",
                        latency_s=elapsed / len(questions),
                        batch_mode=True,
                        error="No result in batch output",
                    )
                )
        return responses

    def _batch_via_asyncio(
        self,
        questions: List[Dict],
        system: str,
        max_tokens: int,
    ) -> List[ModelResponse]:
        """Concurrent fallback: fire all requests in parallel via asyncio."""
        from google.genai import types

        start = time.time()

        async def _one(q: Dict) -> ModelResponse:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.chat(q["prompt"], system=system, max_tokens=max_tokens),
            )

        async def _all():
            return await asyncio.gather(*[_one(q) for q in questions])

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already inside an event loop (e.g. Jupyter) — use nest_asyncio or threads
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    futures = [pool.submit(self.chat, q["prompt"], system, max_tokens) for q in questions]
                    responses = [f.result() for f in futures]
            else:
                responses = loop.run_until_complete(_all())
        except Exception:
            # Final fallback: sequential
            responses = [self.chat(q["prompt"], system=system, max_tokens=max_tokens) for q in questions]

        elapsed = time.time() - start
        for r in responses:
            r.latency_s = elapsed / len(questions)
            r.batch_mode = False  # concurrent, not discounted batch
        return responses

    # ------------------------------------------------------------------
    # Tool-calling (synchronous multi-turn loop)
    # ------------------------------------------------------------------

    def chat_with_tools(
        self,
        prompt: str,
        tool_names: List[str],
        system: str = "You are a knowledgeable medical assistant. Use the provided tools to assess the patient and then give your clinical recommendation.",
        max_tokens: int = 2048,
        max_rounds: int = 10,
    ) -> ModelResponse:
        from google.genai import types
        from tools import get_gemini_tools, execute_tool

        gemini_tools = get_gemini_tools(tool_names)
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0.1,
            tools=gemini_tools,
        )

        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        tool_calls_log: List[ToolCall] = []
        total_input = total_output = 0
        start = time.time()
        final_text = ""

        try:
            for _ in range(max_rounds):
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                usage = resp.usage_metadata
                if usage:
                    total_input += usage.prompt_token_count or 0
                    total_output += usage.candidates_token_count or 0

                candidate = resp.candidates[0]
                contents.append(candidate.content)

                fc_parts = [p for p in candidate.content.parts if p.function_call is not None]

                if not fc_parts:
                    final_text = "".join(
                        p.text for p in candidate.content.parts
                        if hasattr(p, "text") and p.text
                    )
                    break

                response_parts = []
                for p in fc_parts:
                    fc = p.function_call
                    args = dict(fc.args)
                    result = execute_tool(fc.name, args)
                    tool_calls_log.append(ToolCall(name=fc.name, arguments=args, result=result))
                    response_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fc.name,
                                response={"result": json.dumps(result)},
                            )
                        )
                    )
                contents.append(types.Content(role="user", parts=response_parts))

            return ModelResponse(
                model_name=self.model,
                final_text=final_text,
                tool_calls=tool_calls_log,
                input_tokens=total_input,
                output_tokens=total_output,
                latency_s=time.time() - start,
            )

        except Exception as exc:
            return ModelResponse(
                model_name=self.model,
                final_text="",
                tool_calls=tool_calls_log,
                latency_s=time.time() - start,
                error=str(exc),
            )
