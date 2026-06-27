#!/usr/bin/env python3
"""
eval/vlmeval_model.py

Custom VLMEvalKit model class that wraps the project's inference engine.
Supports sandbox code execution (evaluate_single_with_cleanup) so models
can write and run Python code during evaluation.

Usage in VLMEvalKit config JSON:
{
    "model": {
        "MyModel": {
            "class": "ImagineModel",
            "api_config_path": "api_config_files/api_config_vlm.json",
            "prompt_template_path": "prompt_template/prompt_template_vis.json",
            "prompt_key": "imagine",
            "exe_code": true,
            "model_name": "qwen",
            "max_tokens": 16000,
            "max_rounds": 8
        }
    }
}

The class is registered into vlmeval.api namespace by run_vlmevalkit_local.py.
"""

import json
import os
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

VLMEVAL_ROOT = os.environ.get(
    "VLMEVAL_ROOT", os.path.join(PROJECT_ROOT, "eval", "VLMEvalKit")
)
if VLMEVAL_ROOT not in sys.path:
    sys.path.insert(0, VLMEVAL_ROOT)

from vlmeval.api.base import BaseAPI


class ImagineModel(BaseAPI):
    """
    VLMEvalKit-compatible model that delegates inference to the project's
    inference engine (inference_engine/vis_inference_demo_gpt.py).

    When exe_code=True, the model gets a PythonExecutor sandbox so it can
    write and execute code during multi-round inference.
    When exe_code=False, it behaves as a plain API call with the project's
    prompt template applied.
    """

    is_api: bool = True
    INTERLEAVE = True

    def __init__(
        self,
        api_config_path: str,
        prompt_template_path: str,
        prompt_key: str = "no_tool",
        model_name: str = "qwen",
        client_type: str = "openai",
        exe_code: bool = False,
        max_tokens: int = 16000,
        max_rounds: int = 8,
        temperature: float = 0.0,
        retry: int = 3,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(retry=retry, verbose=verbose, **kwargs)

        # Resolve relative paths against project root
        if not os.path.isabs(api_config_path):
            api_config_path = os.path.join(PROJECT_ROOT, api_config_path)
        if not os.path.isabs(prompt_template_path):
            prompt_template_path = os.path.join(PROJECT_ROOT, prompt_template_path)

        # Build OpenAI client from api_config
        from openai import OpenAI

        with open(api_config_path) as f:
            api_cfg = json.load(f)
        self.client = OpenAI(
            api_key=api_cfg["api_key"][0],
            base_url=api_cfg.get("base_url"),
        )

        # Eval args dict consumed by evaluate_single_with_cleanup / evaluate_single_data
        self.eval_args = {
            "max_tokens": max_tokens,
            "prompt_template": prompt_template_path,
            "prompt": prompt_key,
            "exe_code": exe_code,
            "temperature": temperature,
            "max_rounds": max_rounds,
            "client_type": client_type,
            "api_name": model_name,
        }

        self._api_config_path = api_config_path
        self._prompt_template_path = prompt_template_path

    # ------------------------------------------------------------------ #
    # VLMEvalKit interface
    # ------------------------------------------------------------------ #
    def _extract_text_only(self, content):
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""

        texts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text = item.get("text", "")
                if text:
                    texts.append(text)
        return "".join(texts).strip()

    def _build_extra_records(self, question, messages, final_response, in_tok, out_tok):
        trace = []
        first_user_replaced = False

        for msg in messages:
            role = msg.get("role")
            if role not in {"user", "assistant", "system", "tool"}:
                continue

            if role == "user" and not first_user_replaced:
                text = question.strip()
                first_user_replaced = True
            else:
                text = self._extract_text_only(msg.get("content"))

            if not text:
                continue

            trace.append({"role": role, "text": text})

        final_text = (final_response or "").strip()
        if final_text and (not trace or trace[-1] != {"role": "assistant", "text": final_text}):
            trace.append({"role": "assistant", "text": final_text})

        return {
            "input_tokens": int(in_tok or 0),
            "output_tokens": int(out_tok or 0),
            "turn_count": len(trace),
            "trace": trace,
        }

    def generate_inner(self, message, dataset=None, **kwargs):
        """
        Called by BaseAPI.generate().

        Args:
            message: list of dicts, e.g.
                [{"type": "image", "value": "/path/to/img.jpg"},
                 {"type": "text",  "value": "Question: ..."}]

        Returns:
            (ret_code, answer, log)
            ret_code 0 = success
        """
        from inference_engine.vis_inference_demo_gpt import evaluate_single_with_cleanup

        # Extract image paths and question text from VLMEvalKit message
        image_paths = []
        question_parts = []
        for msg in message:
            if msg["type"] == "image":
                image_paths.append(msg["value"])
            elif msg["type"] == "text":
                question_parts.append(msg["value"])

        question = "\n".join(question_parts)
        if not image_paths:
            image_paths = [""]

        data_input = {
            "question": question,
            "image_path_list": image_paths,
        }

        try:
            messages, final_response, in_tok, out_tok = evaluate_single_with_cleanup(
                self.eval_args, data_input, self.client
            )
            if final_response is None or final_response == "":
                return -1, self.fail_msg, "Empty response from inference engine"
            extra_records = self._build_extra_records(
                question=question,
                messages=messages,
                final_response=final_response,
                in_tok=in_tok,
                out_tok=out_tok,
            )
            return 0, final_response, extra_records
        except Exception as e:
            tb = traceback.format_exc()
            if self.verbose:
                self.logger.error(f"ImagineModel error: {e}\n{tb}")
            return -1, self.fail_msg, str(e)


# Backward-compatible alias for older local configs.
AdatwiModel = ImagineModel
