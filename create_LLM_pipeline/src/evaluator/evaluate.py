from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

from src.generator.generate import GeneratorConfig, _extract_json_block, _llm_generate


@dataclass
class EvaluatorConfig:
    model_name: str = "/RadOnc-MRI1/Student_Folder/whruiray/SA/local_models/Qwen2.5-7B-Instruct"
    use_4bit: bool = True
    max_new_tokens: int = 2048
    temperature: float = 0.0
    prompt_template_path: str = "prompts/evaluate_survival_v1.txt"
    skip: bool = False


@dataclass
class Scorecard:
    overall_score: float
    confidence: float
    issues: List[str]
    verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _render_template(template: str, replacements: Dict[str, str]) -> str:
    out = template
    for k, v in replacements.items():
        out = out.replace(f"{{{{{k}}}}}", v)
    return out


def evaluate(canonical, extraction, rule_qc, eval_cfg: EvaluatorConfig) -> Scorecard:
    if eval_cfg.skip:
        return Scorecard(
            overall_score=1.0 if rule_qc and rule_qc.passed else 0.4,
            confidence=0.5,
            issues=[] if rule_qc and rule_qc.passed else (rule_qc.errors if rule_qc else ["validation unavailable"]),
            verdict="pass" if rule_qc and rule_qc.passed else "borderline",
        )

    try:
        template = _read_text(eval_cfg.prompt_template_path)

        prompt = _render_template(
            template,
            {
                "SOURCE_TEXT": canonical.canonical_text,
                "EXTRACTION_JSON": json.dumps(extraction, ensure_ascii=False, indent=2),
            },
        )

        gen_cfg = GeneratorConfig(
            model_name=eval_cfg.model_name,
            use_4bit=eval_cfg.use_4bit,
            max_new_tokens=eval_cfg.max_new_tokens,
            temperature=eval_cfg.temperature,
            do_sample=False,
            prompt_template_path="",
            repair_prompt_path="",
            cache_dir="cache/evaluate_survival",
        )

        raw_text = _llm_generate(prompt, gen_cfg)
        json_text = _extract_json_block(raw_text)
        obj = json.loads(json_text)

        score = float(obj.get("score", 0.0))
        verdict = str(obj.get("verdict", "fail"))
        issues = obj.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)]

        confidence = 0.8 if verdict == "pass" else 0.6 if verdict == "borderline" else 0.4

        return Scorecard(
            overall_score=max(0.0, min(1.0, score)),
            confidence=confidence,
            issues=[str(x) for x in issues],
            verdict=verdict,
        )

    except Exception as e:
        return Scorecard(
            overall_score=0.0,
            confidence=0.2,
            issues=[f"evaluator_error: {str(e)}"],
            verdict="fail",
        )