import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}


@dataclass
class GeneratorConfig:
    model_name: str = "/RadOnc-MRI1/Student_Folder/whruiray/SA/local_models/Qwen2.5-3B-Instruct"
    use_4bit: bool = True
    max_new_tokens: int = 2048
    temperature: float = 0.0
    do_sample: bool = False
    prompt_template_path: str = "prompts/extract_survival_v1.txt"
    repair_prompt_path: str = "prompts/repair_extract_survival_v1.txt"
    max_retries: int = 1
    cache_dir: str = "cache/extract_survival"
    controlled_findings: List[str] = field(default_factory=list)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _render_template(template: str, replacements: Dict[str, str]) -> str:
    out = template
    for k, v in replacements.items():
        out = out.replace(f"{{{{{k}}}}}", v)
    return out


def _extract_json_block(text: str) -> str:
    text = text.strip()

    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

    return text[start:]


def _cache_key(prompt: str, model_name: str) -> str:
    s = f"{model_name}\n{prompt}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _get_model_and_tokenizer(cfg: GeneratorConfig):
    key = f"{cfg.model_name}__4bit={cfg.use_4bit}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    model_path = Path(cfg.model_name)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Local model path does not exist: {cfg.model_name}. "
            "This pipeline is configured for local-only models."
        )

    quant_config = None
    if cfg.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    _MODEL_CACHE[key] = (tokenizer, model)
    return tokenizer, model


def _llm_generate(prompt: str, cfg: GeneratorConfig) -> str:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    cache_path = Path(cfg.cache_dir) / f"{_cache_key(prompt, cfg.model_name)}.txt"

    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    tokenizer, model = _get_model_and_tokenizer(cfg)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=min(getattr(tokenizer, "model_max_length", 32768), 32768),
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            do_sample=cfg.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if text.startswith(prompt):
        text = text[len(prompt):].strip()

    cache_path.write_text(text, encoding="utf-8")

    import time
    t0 = time.time()
    outputs = model.generate(...)
    print(f"Generation took {time.time() - t0:.1f} sec")
    return text


def _build_meta(canonical) -> Dict[str, Any]:
    meta = {
        "doc_id": canonical.doc_id,
        "patient_id": canonical.patient_id,
    }
    meta.update(canonical.raw_metadata or {})
    return meta


def generate(canonical, cfg: GeneratorConfig) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    raw_text = ""
    try:
        template = _read_text(cfg.prompt_template_path)
        meta = _build_meta(canonical)

        # temporary truncation for stability
        source_text = canonical.canonical_text[:12000]

        prompt = _render_template(
            template,
            {
                "META_JSON": json.dumps(meta, ensure_ascii=False, indent=2),
                "SOURCE_TEXT": source_text,
            },
        )

        raw_text = _llm_generate(prompt, cfg)

        os.makedirs("logs/raw_generations", exist_ok=True)
        raw_path = os.path.join("logs/raw_generations", f"{canonical.doc_id}.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

        json_text = _extract_json_block(raw_text)
        obj = json.loads(json_text)
        return obj, None

    except Exception as e:
        os.makedirs("logs/raw_generations", exist_ok=True)
        err_path = os.path.join("logs/raw_generations", f"{canonical.doc_id}_error.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(f"ERROR: {str(e)}\n\n")
            if raw_text:
                f.write(raw_text)
        return None, str(e)


def repair_validation_errors(
    extraction: Dict[str, Any],
    validation_errors: List[str],
    canonical,
    cfg: GeneratorConfig,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    raw_text = ""
    try:
        template = _read_text(cfg.repair_prompt_path)

        source_text = canonical.canonical_text[:12000]

        broken_json = {
            "current_extraction": extraction,
            "validation_errors": validation_errors,
            "meta": _build_meta(canonical),
            "source_text": source_text,
        }

        prompt = _render_template(
            template,
            {
                "BROKEN_JSON": json.dumps(broken_json, ensure_ascii=False, indent=2),
                "SOURCE_TEXT": source_text,
            },
        )

        raw_text = _llm_generate(prompt, cfg)

        os.makedirs("logs/raw_generations", exist_ok=True)
        raw_path = os.path.join("logs/raw_generations", f"{canonical.doc_id}_repair_validation.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

        json_text = _extract_json_block(raw_text)
        obj = json.loads(json_text)
        return obj, None

    except Exception as e:
        os.makedirs("logs/raw_generations", exist_ok=True)
        err_path = os.path.join("logs/raw_generations", f"{canonical.doc_id}_repair_validation_error.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(f"ERROR: {str(e)}\n\n")
            if raw_text:
                f.write(raw_text)
        return None, str(e)


def repair_with_evaluator_feedback(
    extraction: Dict[str, Any],
    evaluator_feedback: Dict[str, Any],
    canonical,
    cfg: GeneratorConfig,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    raw_text = ""
    try:
        template = _read_text(cfg.repair_prompt_path)

        source_text = canonical.canonical_text[:12000]

        broken_json = {
            "current_extraction": extraction,
            "evaluator_feedback": evaluator_feedback,
            "meta": _build_meta(canonical),
            "source_text": source_text,
        }

        prompt = _render_template(
            template,
            {
                "BROKEN_JSON": json.dumps(broken_json, ensure_ascii=False, indent=2),
                "SOURCE_TEXT": source_text,
            },
        )

        raw_text = _llm_generate(prompt, cfg)

        os.makedirs("logs/raw_generations", exist_ok=True)
        raw_path = os.path.join("logs/raw_generations", f"{canonical.doc_id}_repair_eval.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

        json_text = _extract_json_block(raw_text)
        obj = json.loads(json_text)
        return obj, None

    except Exception as e:
        os.makedirs("logs/raw_generations", exist_ok=True)
        err_path = os.path.join("logs/raw_generations", f"{canonical.doc_id}_repair_eval_error.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(f"ERROR: {str(e)}\n\n")
            if raw_text:
                f.write(raw_text)
        return None, str(e)