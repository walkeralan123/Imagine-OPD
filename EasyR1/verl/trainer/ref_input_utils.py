"""
Utilities for constructing teacher (ref model) inputs for multimodal context distillation (OPD).

The teacher sees the same prompt as the student but with extra images, a
teacher-specific system prompt, and optional ground-truth answer text. We
re-tokenize the teacher's prompt via the processor (so image_pad tokens are
correct), concatenate with the student's response tokens, and compute 3-D RoPE
position IDs for Qwen2/2.5-VL or Qwen3-VL.
"""

from typing import Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto
from ..utils.dataset import process_image


DEFAULT_TEACHER_SYSTEM_PROMPT = (
    "You are given two images. The first image is the diagram for the question. "
    "The second image is an additional reference image that is NOT directly related to the question. "
    "Focus on the first image to solve the problem."
)


def prepare_mm_context_distillation_inputs(
    batch: DataProto,
    processor: ProcessorMixin,
    tokenizer: PreTrainedTokenizer,
    teacher_system_prompt: Optional[str] = None,
    include_ground_truth: bool = True,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> DataProto:
    """Prepare inputs for multimodal context distillation.

    Stores ``ref_input_ids``, ``ref_attention_mask``, ``ref_position_ids`` in
    ``batch.batch`` and ``ref_multi_modal_inputs`` in ``batch.non_tensor_batch``.
    """
    if teacher_system_prompt is None:
        teacher_system_prompt = DEFAULT_TEACHER_SYSTEM_PROMPT

    if "raw_prompt" not in batch.non_tensor_batch:
        raise ValueError(
            "raw_prompt not found in batch.non_tensor_batch. "
            "Please set data.return_raw_chat=True in config."
        )

    if "Qwen3VLProcessor" in processor.__class__.__name__:
        from ..models.transformers.qwen3_vl import get_rope_index
    else:
        from ..models.transformers.qwen2_vl import get_rope_index

    batch_size = len(batch)
    raw_prompts = batch.non_tensor_batch["raw_prompt"]
    ground_truths = batch.non_tensor_batch.get("ground_truth")
    responses = batch.batch["responses"]  # (batch_size, response_length)

    ref_prompt_ids_list = []
    ref_mm_inputs_list = []  # per-sample dicts with pixel_values, image_grid_thw

    for i in range(batch_size):
        # --- 1. Gather student messages and images ---
        messages = raw_prompts[i]
        if isinstance(messages, np.ndarray):
            messages = list(messages)
        ground_truth = None if ground_truths is None else ground_truths[i]
        if isinstance(ground_truth, np.generic):
            ground_truth = ground_truth.item()
        if isinstance(ground_truth, bytes):
            ground_truth = ground_truth.decode("utf-8")

        # Get student's original images from multi_modal_data
        data_item = batch[i]
        original_images = []
        if "multi_modal_data" in data_item.non_tensor_batch:
            mm_data = data_item.non_tensor_batch["multi_modal_data"]
            if isinstance(mm_data, dict):
                raw_images = list(mm_data.get("images", []))
                # Process images (they may be raw bytes/paths from the dataset)
                original_images = [process_image(img, min_pixels, max_pixels) for img in raw_images]

        # Get teacher extra images (already processed in dataset.__getitem__)
        teacher_extra = []
        if "teacher_extra_images" in data_item.non_tensor_batch:
            teacher_extra = list(data_item.non_tensor_batch["teacher_extra_images"])

        # --- 2. Build teacher messages ---
        teacher_messages = []
        for msg in messages:
            msg = dict(msg) if isinstance(msg, dict) else msg
            if msg.get("role") == "system":
                # Skip student system prompt (replaced by teacher's below)
                continue
            elif msg.get("role") == "user":
                content = msg.get("content", "")
                # Build content list: keep original content + append extra image
                if isinstance(content, str):
                    # Convert string content to content list (handles <image> tags)
                    content_list = []
                    parts = content.split("<image>")
                    for j, part in enumerate(parts):
                        if j > 0:
                            content_list.append({"type": "image"})
                        if part:
                            content_list.append({"type": "text", "text": part})
                elif isinstance(content, list):
                    content_list = list(content)
                else:
                    content_list = [{"type": "text", "text": str(content)}]

                # Append extra image slots for teacher
                for _ in teacher_extra:
                    content_list.append({"type": "image"})

                teacher_messages.append({"role": "user", "content": content_list})
            else:
                teacher_messages.append(dict(msg))

        if include_ground_truth and ground_truth is not None:
            answer_text = f"Ground-truth answer:{ground_truth}"
            for teacher_msg in reversed(teacher_messages):
                if teacher_msg.get("role") != "user":
                    continue
                content = teacher_msg.get("content", "")
                if isinstance(content, list):
                    content.append({"type": "text", "text": answer_text})
                else:
                    teacher_msg["content"] = f"{content}\n{answer_text}"
                break
            else:
                teacher_messages.append({"role": "user", "content": answer_text})

        # Prepend teacher system prompt
        teacher_messages.insert(0, {"role": "system", "content": teacher_system_prompt})

        # --- 3. Apply processor to get teacher tokenized input ---
        all_images = original_images + teacher_extra  # order matches <image> slots

        teacher_prompt_str = processor.apply_chat_template(
            teacher_messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        teacher_model_inputs = processor(
            text=[teacher_prompt_str],
            images=all_images if all_images else None,
            return_tensors="pt",
        )

        teacher_prompt_ids = teacher_model_inputs.pop("input_ids")[0]  # (prompt_len,)
        teacher_model_inputs.pop("attention_mask", None)
        teacher_model_inputs.pop("second_per_grid_ts", None)

        # Store per-sample multimodal inputs (pixel_values, image_grid_thw, etc.)
        ref_mm_inputs_list.append(dict(teacher_model_inputs))
        ref_prompt_ids_list.append(teacher_prompt_ids)

    # --- 4. Left-pad teacher prompts ---
    max_prompt_len = max(len(ids) for ids in ref_prompt_ids_list)

    ref_prompt_ids_padded = []
    ref_prompt_attention_mask_padded = []

    for ref_prompt_ids in ref_prompt_ids_list:
        prompt_len = len(ref_prompt_ids)
        pad_len = max_prompt_len - prompt_len

        if pad_len > 0:
            padding = torch.full((pad_len,), tokenizer.pad_token_id, dtype=ref_prompt_ids.dtype)
            padded_ids = torch.cat([padding, ref_prompt_ids], dim=0)
            attn_mask = torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(prompt_len, dtype=torch.long),
            ], dim=0)
        else:
            padded_ids = ref_prompt_ids
            attn_mask = torch.ones(prompt_len, dtype=torch.long)

        ref_prompt_ids_padded.append(padded_ids)
        ref_prompt_attention_mask_padded.append(attn_mask)

    ref_prompt_ids_tensor = torch.stack(ref_prompt_ids_padded, dim=0)
    ref_prompt_attn_tensor = torch.stack(ref_prompt_attention_mask_padded, dim=0)

    # --- 5. Concat with student response tokens ---
    ref_input_ids_tensor = torch.cat([ref_prompt_ids_tensor, responses], dim=1)

    if "response_mask" in batch.batch:
        response_attention_mask = batch.batch["response_mask"]
    else:
        response_attention_mask = torch.ones_like(responses, dtype=torch.long)

    ref_attention_mask_tensor = torch.cat([ref_prompt_attn_tensor, response_attention_mask], dim=1)

    # --- 6. Compute 3-D RoPE position IDs for Qwen2/2.5-VL or Qwen3-VL ---
    ref_position_ids_list = []
    for i in range(batch_size):
        image_grid_thw = ref_mm_inputs_list[i].get("image_grid_thw", None)
        video_grid_thw = ref_mm_inputs_list[i].get("video_grid_thw", None)

        vision_position_ids = get_rope_index(
            processor,
            input_ids=ref_input_ids_tensor[i],
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=ref_attention_mask_tensor[i],
        )  # (3, seqlen)

        # Compute text position IDs (1, seqlen)
        valid_mask = ref_attention_mask_tensor[i].bool()
        text_pos = torch.zeros((1, ref_input_ids_tensor.shape[1]), dtype=torch.long)
        text_pos[0, valid_mask] = torch.arange(valid_mask.sum().item())

        # Combine: (4, seqlen)
        position_ids = torch.cat([text_pos, vision_position_ids], dim=0)
        ref_position_ids_list.append(position_ids)

    ref_position_ids_tensor = torch.stack(ref_position_ids_list, dim=0)  # (B, 4, seqlen)

    # --- 7. Store in batch ---
    batch.batch["ref_input_ids"] = ref_input_ids_tensor
    batch.batch["ref_attention_mask"] = ref_attention_mask_tensor
    batch.batch["ref_position_ids"] = ref_position_ids_tensor
    batch.non_tensor_batch["ref_multi_modal_inputs"] = np.array(ref_mm_inputs_list, dtype=object)

    print(
        f"MM context distillation: ref_input_ids shape={ref_input_ids_tensor.shape}, "
        f"ref_position_ids shape={ref_position_ids_tensor.shape}, "
        f"student input_ids shape={batch.batch['input_ids'].shape}"
    )

    return batch
