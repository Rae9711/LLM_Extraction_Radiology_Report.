from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from src.canonicalizer.canonicalize import canonicalize
from src.generator.generate import (
    GeneratorConfig,
    generate,
    repair_validation_errors,
    repair_with_evaluator_feedback,
)
from src.validators.validate import RuleQC, validate
from src.evaluator.evaluate import EvaluatorConfig, Scorecard, evaluate
from src.gate.gate import GateConfig, gate

LOG = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    doc_id: str
    decision: str
    extraction_final: Optional[Dict[str, Any]] = None
    quality_metadata: Optional[Dict[str, Any]] = None
    scorecard: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    elapsed_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "doc_id": self.doc_id,
            "decision": self.decision,
            "elapsed_sec": self.elapsed_sec,
        }
        if self.extraction_final is not None:
            d["extraction_final"] = self.extraction_final
        if self.quality_metadata is not None:
            d["quality_metadata"] = self.quality_metadata
        if self.scorecard is not None:
            d["scorecard"] = self.scorecard
        if self.error is not None:
            d["error"] = self.error
        return d


@dataclass
class PipelineConfig:
    section_order: List[str] = field(default_factory=lambda: [
        "comparison",
        "interpretation_date",
        "outside_institution",
        "history",
        "procedure",
        "iv_contrast",
        "oral_contrast",
        "field_strength",
        "reconstructions",
        "findings",
        "impression",
        "summary",
        "other",
    ])
    preserve_case: bool = True
    preserve_punctuation: bool = True

    generator: GeneratorConfig = field(default_factory=GeneratorConfig)

    schema_path: str = "schema/extraction_survival_v1.json"
    require_nonempty_summary: bool = True

    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)

    gate: GateConfig = field(default_factory=GateConfig)

    input_json_dir: str = "input_patients_json"
    output_jsonl: str = "data_processed/pipeline_extractions_survival.jsonl"
    failures_jsonl: str = "data_processed/pipeline_failures_survival.jsonl"
    scorecard_jsonl: str = "data_processed/pipeline_scorecards_survival.jsonl"

    pipeline_version: str = "survival_1.0.0"

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        cfg = cls()
        cfg.pipeline_version = raw.get("pipeline_version", cfg.pipeline_version)

        canon = raw.get("canonicalizer", {})
        cfg.section_order = canon.get("section_order", cfg.section_order)
        cfg.preserve_case = canon.get("preserve_case", cfg.preserve_case)
        cfg.preserve_punctuation = canon.get("preserve_punctuation", cfg.preserve_punctuation)

        gen = raw.get("generator", {})
        cfg.generator = GeneratorConfig(
            model_name=gen.get("model_name", cfg.generator.model_name),
            use_4bit=gen.get("use_4bit", cfg.generator.use_4bit),
            max_new_tokens=gen.get("max_new_tokens", cfg.generator.max_new_tokens),
            temperature=gen.get("temperature", cfg.generator.temperature),
            do_sample=gen.get("do_sample", cfg.generator.do_sample),
            prompt_template_path=gen.get("prompt_template", cfg.generator.prompt_template_path),
            repair_prompt_path=gen.get("repair_prompt_template", cfg.generator.repair_prompt_path),
            max_retries=gen.get("max_retries", cfg.generator.max_retries),
            cache_dir=gen.get("cache_dir", cfg.generator.cache_dir),
        )

        val = raw.get("validators", {})
        cfg.schema_path = val.get("schema_path", cfg.schema_path)
        cfg.require_nonempty_summary = val.get(
            "require_nonempty_summary", cfg.require_nonempty_summary
        )

        ev = raw.get("evaluator", {})
        cfg.evaluator = EvaluatorConfig(
            model_name=ev.get("model_name", cfg.evaluator.model_name),
            use_4bit=ev.get("use_4bit", cfg.evaluator.use_4bit),
            max_new_tokens=ev.get("max_new_tokens", cfg.evaluator.max_new_tokens),
            temperature=ev.get("temperature", cfg.evaluator.temperature),
            prompt_template_path=ev.get("prompt_template", cfg.evaluator.prompt_template_path),
            skip=ev.get("skip", cfg.evaluator.skip),
        )

        gt = raw.get("gate", {})
        cfg.gate = GateConfig(
            pass_threshold=gt.get("pass_threshold", cfg.gate.pass_threshold),
            fail_threshold=gt.get("fail_threshold", cfg.gate.fail_threshold),
            max_repair_iterations=gt.get("max_repair_iterations", cfg.gate.max_repair_iterations),
            re_evaluate_after_repair=gt.get(
                "re_evaluate_after_repair", cfg.gate.re_evaluate_after_repair
            ),
        )

        io = raw.get("io", {})
        cfg.input_json_dir = io.get("input_json_dir", cfg.input_json_dir)
        cfg.output_jsonl = io.get("output_jsonl", cfg.output_jsonl)
        cfg.failures_jsonl = io.get("failures_jsonl", cfg.failures_jsonl)
        cfg.scorecard_jsonl = io.get("scorecard_jsonl", cfg.scorecard_jsonl)

        return cfg


class Pipeline:
    def __init__(self, cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        LOG.info("Pipeline initialized (version=%s)", self.cfg.pipeline_version)

    def process_one(self, raw_report: Dict[str, Any]) -> PipelineResult:
        t0 = time.time()
        doc_id = str(raw_report.get("doc_id", "unknown"))

        try:
            canonical = canonicalize(
                raw_report,
                section_order=self.cfg.section_order,
                preserve_case=self.cfg.preserve_case,
            )
            LOG.info("[A] Canonicalized %s (%d chars)", doc_id, len(canonical.canonical_text))

            if not canonical.canonical_text:
                return PipelineResult(
                    doc_id=doc_id,
                    decision="FAIL",
                    error="Empty canonical text after canonicalization",
                    elapsed_sec=time.time() - t0,
                )

            max_outer_iterations = 3
            outer_iteration = 0
            extraction = None
            rule_qc = None
            scorecard = None

            while outer_iteration < max_outer_iterations:
                outer_iteration += 1

                extraction, gen_error = generate(canonical, self.cfg.generator)
                if extraction is None:
                    LOG.warning("[B] Generator failed for %s: %s", doc_id, gen_error)
                    if outer_iteration >= max_outer_iterations:
                        return PipelineResult(
                            doc_id=doc_id,
                            decision="FAIL",
                            error=f"Generator failed after retries: {gen_error}",
                            elapsed_sec=time.time() - t0,
                        )
                    continue

                max_validation_attempts = 2
                validation_attempt = 0
                validation_passed = False

                while validation_attempt < max_validation_attempts:
                    validation_attempt += 1

                    rule_qc, extraction = validate(
                        extraction=extraction,
                        source_text=canonical.canonical_text,
                        schema_path=self.cfg.schema_path,
                        require_nonempty_summary=self.cfg.require_nonempty_summary,
                        auto_repair=True,
                    )

                    if rule_qc.passed:
                        validation_passed = True
                        LOG.info("[C] Validation passed for %s", doc_id)
                        break

                    LOG.warning("[C] Validation failed for %s: %s", doc_id, rule_qc.errors)

                    if validation_attempt < max_validation_attempts:
                        extraction, repair_error = repair_validation_errors(
                            extraction, rule_qc.errors, canonical, self.cfg.generator
                        )
                        if extraction is None:
                            LOG.warning("[C] Validation repair failed for %s: %s", doc_id, repair_error)
                            break

                scorecard = evaluate(
                    canonical=canonical,
                    extraction=extraction,
                    rule_qc=rule_qc,
                    eval_cfg=self.cfg.evaluator,
                )

                LOG.info(
                    "[D] Evaluated %s: score=%.3f confidence=%.3f",
                    doc_id,
                    scorecard.overall_score,
                    scorecard.confidence,
                )

                if validation_passed and scorecard.overall_score >= self.cfg.gate.pass_threshold:
                    break

                if outer_iteration >= max_outer_iterations:
                    break

                extraction, repair_error = repair_with_evaluator_feedback(
                    extraction, scorecard.to_dict(), canonical, self.cfg.generator
                )
                if extraction is None:
                    LOG.warning("[D] Evaluator-guided repair failed for %s: %s", doc_id, repair_error)

            decision, extraction_final, quality_metadata = gate(
                canonical=canonical,
                extraction=extraction,
                rule_qc=rule_qc,
                scorecard=scorecard,
                gate_cfg=self.cfg.gate,
                eval_cfg=self.cfg.evaluator,
                gen_cfg=self.cfg.generator,
            )

            LOG.info("[E] Gate decision for %s: %s", doc_id, decision)

            return PipelineResult(
                doc_id=doc_id,
                decision=decision,
                extraction_final=extraction_final,
                quality_metadata=quality_metadata,
                scorecard=scorecard.to_dict() if scorecard else None,
                elapsed_sec=time.time() - t0,
            )

        except Exception as e:
            LOG.exception("Pipeline error for %s", doc_id)
            return PipelineResult(
                doc_id=doc_id,
                decision="ERROR",
                error=str(e),
                elapsed_sec=time.time() - t0,
            )

    def process_batch(self, raw_reports: List[Dict[str, Any]], save: bool = True) -> List[PipelineResult]:
        results: List[PipelineResult] = []
        total = len(raw_reports)

        for i, report in enumerate(raw_reports):
            doc_id = report.get("doc_id", f"report_{i}")
            LOG.info("Processing %d/%d: %s", i + 1, total, doc_id)

            result = self.process_one(report)
            results.append(result)

            if save:
                self._save_result(result)

        decisions = [r.decision for r in results]
        LOG.info(
            "Batch complete: %d total | PASS=%d REPAIR=%d FAIL=%d ERROR=%d",
            total,
            decisions.count("PASS"),
            decisions.count("REPAIR"),
            decisions.count("FAIL"),
            decisions.count("ERROR"),
        )
        return results

    def _save_result(self, result: PipelineResult) -> None:
        if result.decision in ("PASS", "REPAIR") and result.extraction_final:
            _jsonl_append(self.cfg.output_jsonl, result.extraction_final)
        elif result.decision in ("FAIL", "ERROR"):
            _jsonl_append(self.cfg.failures_jsonl, {
                "doc_id": result.doc_id,
                "decision": result.decision,
                "error": result.error,
                "quality_metadata": result.quality_metadata,
            })

        if result.scorecard:
            _jsonl_append(self.cfg.scorecard_jsonl, result.scorecard)


def _jsonl_append(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")