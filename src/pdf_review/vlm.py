"""Qwen2.5-VL loading and generation.

Unifies model loading + generate() + generate_json(), which were duplicated
verbatim (generate) or near-verbatim (generate_json) across
run_general_pdf_review.py, rerun_general_pdf_generation.py, and
ask_pdf_question.py. Each caller's own retry-hint sentence and
run_general_pdf_review.py's validator-fallback quirk are preserved via
parameters so this refactor changes no observable behavior.
"""
import gc

from .io import atomic_json


class VLMRunner:
    """Holds a loaded Qwen2.5-VL model + processor and runs generation calls."""

    def __init__(self, model_path, max_pixels):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.float16, device_map="auto", local_files_only=True,
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, max_pixels=max_pixels,
        )

    def generate(self, messages, max_new_tokens):
        """Run one generation call. Ported unchanged (previously duplicated
        identically in all three scripts)."""
        import torch
        from qwen_vl_utils import process_vision_info

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        answer = self.processor.batch_decode(
            generated[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
        counts = {
            "input_tokens": int(inputs.input_ids.shape[1]),
            "generated_tokens": int(generated.shape[1] - inputs.input_ids.shape[1]),
        }
        del inputs, generated
        gc.collect()
        torch.cuda.empty_cache()
        return answer, counts

    def generate_json(
        self, messages, max_new_tokens, validator, *,
        retry_hint, max_retries, output_dir,
        keep_parsed_if_validator_returns_none=False,
    ):
        """Generate, parse, and validate JSON, retrying with a validation-error
        hint on failure. On exhausting max_retries, writes
        output_dir/generation_failure.json and raises.

        retry_hint is the caller-specific sentence(s) inserted into the fixed
        retry-message template (see each CLI for its exact original wording).
        keep_parsed_if_validator_returns_none reproduces
        run_general_pdf_review.py's legacy behavior of falling back to the
        raw parsed JSON when the validator returns None instead of a value;
        the other two callers' validators never return None, so this has no
        effect for them.
        """
        from .io import parse_json_response

        attempts = []
        current_messages = messages
        for attempt in range(max_retries + 1):
            raw, counts = self.generate(current_messages, max_new_tokens)
            try:
                parsed = parse_json_response(raw)
                validated = validator(parsed)
                if keep_parsed_if_validator_returns_none and validated is None:
                    validated = parsed
                return validated, raw, counts, attempts
            except (KeyError, TypeError, ValueError) as error:
                attempts.append({"attempt": attempt + 1, "raw": raw, "error": str(error)})
                print(
                    f"JSON validation failed on attempt {attempt + 1}: {error}\n"
                    f"Raw response: {raw}",
                    flush=True,
                )
                if attempt == max_retries:
                    atomic_json(output_dir / "generation_failure.json", {
                        "error": str(error),
                        "attempts": attempts,
                    })
                    raise ValueError(
                        f"JSON generation failed after {attempt + 1} attempts: {error}"
                    ) from error
                current_messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": (
                        f"The previous response failed validation. Return corrected JSON only. "
                        f"{retry_hint} Validation error: {error}"
                    )},
                ]
        raise RuntimeError("Unreachable JSON generation state")
