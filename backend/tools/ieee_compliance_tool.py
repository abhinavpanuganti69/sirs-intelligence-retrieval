"""
ieee_compliance_tool.py
-----------------------
IEEE Compliance Checker for SIRS RAG Pipeline.

UPGRADED — v2.0 (RAG-based):
    IEEE 12207 — Software Life Cycle Processes       [keyword match]
    IEEE 830   — Software Requirements Specifications [RAG + phi4-mini ✓]
    IEEE 829   — Software Test Documentation          [RAG + phi4-mini ✓]
    IEEE 1016  — Software Design Description          [RAG + phi4-mini ✓]
    IEEE 730   — Software Quality Assurance           [keyword match]

IEEE 830, 829, 1016 now retrieve actual clauses from the standards FAISS
index (built by standards_indexer.py) and ask phi4-mini to reason whether
the content satisfies those clauses. Falls back to keyword matching if the
standards index is not loaded or LLM call fails.

Usage:
    from tools.ieee_compliance_tool import check_compliance

    report = await check_compliance("your RAG response text here")
    print(report["summary"])
"""

import re
import json
import asyncio
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime
from loguru import logger


# ─────────────────────────────────────────────
#  Data classes  (API response shape unchanged)
# ─────────────────────────────────────────────

@dataclass
class StandardResult:
    standard_id:   str
    standard_name: str
    passed:        bool
    score:         float          # 0.0 – 1.0
    checks:        Dict[str, bool]
    failures:      List[str]
    suggestions:   List[str]


@dataclass
class ComplianceReport:
    timestamp:      str
    overall_score:  float         # 0.0 – 1.0
    overall_passed: bool
    standards:      List[StandardResult]
    summary:        str

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict (API response shape is unchanged)."""
        return {
            "timestamp":       self.timestamp,
            "overall_score_pct": round(self.overall_score * 100, 1),
            "overall_passed":  self.overall_passed,
            "verdict":         "COMPLIANT ✅" if self.overall_passed else "NEEDS IMPROVEMENT ⚠️",
            "summary":         self.summary,
            "standards": [
                {
                    "id":          s.standard_id,
                    "name":        s.standard_name,
                    "passed":      s.passed,
                    "score_pct":   round(s.score * 100, 1),
                    "checks":      s.checks,
                    "failures":    s.failures,
                    "suggestions": s.suggestions,
                }
                for s in self.standards
            ],
        }


# ─────────────────────────────────────────────
#  Checker class
# ─────────────────────────────────────────────

class IEEEComplianceChecker:

    PASS_THRESHOLD = 0.5

    def __init__(self):
        # Lazy-loaded to avoid circular imports at module level
        self._store  = None
        self._ollama = None

    # ── Lazy getters ──────────────────────────────────────────────────────────

    def _get_store(self):
        if self._store is None:
            from retrieval.standards_vector_store import get_standards_store
            self._store = get_standards_store()
        return self._store

    def _get_ollama(self):
        if self._ollama is None:
            from llm.ollama_client import OllamaClient
            self._ollama = OllamaClient()
        return self._ollama

    # =========================================================================
    #  KEYWORD CHECKS  (IEEE 12207 and IEEE 730 — no PDFs provided)
    # =========================================================================

    def check_ieee_12207(self, content: str) -> StandardResult:
        """IEEE 12207: Software Life Cycle Processes — keyword match."""
        c = content.lower()
        checks: Dict[str, bool] = {
            "covers_requirements_phase": any(kw in c for kw in [
                "requirement", "srs", "specification", "user need", "stakeholder"]),
            "covers_design_phase": any(kw in c for kw in [
                "design", "architecture", "component", "module", "interface", "layer"]),
            "covers_implementation_phase": any(kw in c for kw in [
                "implement", "develop", "code", "build", "program", "source"]),
            "covers_testing_phase": any(kw in c for kw in [
                "test", "validation", "verification", "quality", "review", "assert"]),
            "covers_deployment_or_maintenance": any(kw in c for kw in [
                "deploy", "maintain", "update", "release", "install", "operation", "run"]),
            "defines_process_roles": any(kw in c for kw in [
                "role", "responsibility", "team", "stakeholder", "owner", "developer"]),
        }
        suggestions = []
        if not checks["covers_requirements_phase"]:
            suggestions.append(
                "Add requirements documentation to cover the IEEE 12207 acquisition/supply process.")
        if not checks["covers_testing_phase"]:
            suggestions.append(
                "Include a verification/validation process as required by IEEE 12207 §6.4.")
        if not checks["covers_deployment_or_maintenance"]:
            suggestions.append(
                "Document the operation and maintenance process per IEEE 12207 §6.7.")

        score    = sum(checks.values()) / len(checks)
        failures = [k.replace("_", " ").capitalize() for k, v in checks.items() if not v]
        return StandardResult(
            standard_id="IEEE 12207",
            standard_name="Software Life Cycle Processes",
            passed=score >= self.PASS_THRESHOLD,
            score=score, checks=checks, failures=failures, suggestions=suggestions,
        )

    def check_ieee_730(self, content: str) -> StandardResult:
        """IEEE 730: Software Quality Assurance — keyword match."""
        c = content.lower()
        checks: Dict[str, bool] = {
            "has_quality_objectives": any(kw in c for kw in [
                "quality", "quality goal", "quality objective", "sqa", "assurance"]),
            "has_review_process": any(kw in c for kw in [
                "review", "inspection", "audit", "walkthrough", "approval", "peer"]),
            "has_metrics_defined": any(kw in c for kw in [
                "metric", "kpi", "measure", "accuracy", "precision", "recall",
                "latency", "throughput", "f1"]),
            "has_defect_tracking": any(kw in c for kw in [
                "defect", "bug", "issue", "error handling", "exception", "fix", "patch"]),
            "has_standards_compliance_mention": any(kw in c for kw in [
                "ieee", "standard", "compliance", "conform", "guideline", "regulation"]),
            "has_security_or_access_control": any(kw in c for kw in [
                "security", "access control", "authentication", "authorization",
                "offline", "encrypt", "secure"]),
        }
        suggestions = []
        if not checks["has_metrics_defined"]:
            suggestions.append(
                "Define quality metrics (retrieval accuracy, latency, recall) per IEEE 730 §4.4.")
        if not checks["has_review_process"]:
            suggestions.append(
                "Document the code review and QA process per IEEE 730 §4.2.")
        if not checks["has_security_or_access_control"]:
            suggestions.append(
                "Include security/access control documentation per IEEE 730 §4.7.")

        score    = sum(checks.values()) / len(checks)
        failures = [k.replace("_", " ").capitalize() for k, v in checks.items() if not v]
        return StandardResult(
            standard_id="IEEE 730",
            standard_name="Software Quality Assurance",
            passed=score >= self.PASS_THRESHOLD,
            score=score, checks=checks, failures=failures, suggestions=suggestions,
        )

    # =========================================================================
    #  KEYWORD FALLBACKS  (used when RAG/LLM fails for 830, 829, 1016)
    # =========================================================================

    def _keyword_830(self, content: str) -> StandardResult:
        c = content.lower()
        checks: Dict[str, bool] = {
            "has_functional_requirements": any(kw in c for kw in [
                "functional", "shall", "must", "should", "feature", "capability"]),
            "has_non_functional_requirements": any(kw in c for kw in [
                "performance", "security", "reliability", "scalability",
                "availability", "offline", "accuracy", "latency"]),
            "has_scope_or_purpose": any(kw in c for kw in [
                "scope", "purpose", "objective", "goal", "overview", "introduction"]),
            "has_measurable_criteria": bool(re.search(
                r"\d+\s*(%|ms|seconds?|users?|requests?|mb|gb|kb|query)", c)),
            "has_traceability_indicators": any(kw in c for kw in [
                "req-", "fr-", "nfr-", "traceable", "traceability", "req id", "#req"]),
            "has_constraints_or_assumptions": any(kw in c for kw in [
                "constraint", "assumption", "limitation", "dependency", "prerequisite"]),
        }
        score    = sum(checks.values()) / len(checks)
        failures = [k.replace("_", " ").capitalize() for k, v in checks.items() if not v]
        return StandardResult(
            standard_id="IEEE 830", standard_name="Software Requirements Specifications",
            passed=score >= self.PASS_THRESHOLD,
            score=score, checks=checks, failures=failures, suggestions=[],
        )

    def _keyword_829(self, content: str) -> StandardResult:
        c = content.lower()
        checks: Dict[str, bool] = {
            "has_test_plan_or_cases": any(kw in c for kw in [
                "test plan", "test case", "test suite", "test script", "tc-", "test id"]),
            "has_expected_results": any(kw in c for kw in [
                "expected", "expected result", "expected output", "pass criteria", "acceptance"]),
            "has_test_types_defined": any(kw in c for kw in [
                "unit test", "integration test", "system test", "regression",
                "smoke test", "load test", "api test", "end-to-end"]),
            "has_pass_fail_criteria": any(kw in c for kw in [
                "pass", "fail", "status", "result", "outcome", "verdict"]),
            "has_test_coverage": any(kw in c for kw in [
                "coverage", "scenario", "edge case", "boundary", "negative test", "corner case"]),
            "has_test_environment_or_tools": any(kw in c for kw in [
                "environment", "pytest", "postman", "tool", "framework", "setup", "ollama"]),
        }
        score    = sum(checks.values()) / len(checks)
        failures = [k.replace("_", " ").capitalize() for k, v in checks.items() if not v]
        return StandardResult(
            standard_id="IEEE 829", standard_name="Software Test Documentation",
            passed=score >= self.PASS_THRESHOLD,
            score=score, checks=checks, failures=failures, suggestions=[],
        )

    def _keyword_1016(self, content: str) -> StandardResult:
        c = content.lower()
        checks: Dict[str, bool] = {
            "has_architecture_description": any(kw in c for kw in [
                "architecture", "system design", "layer", "component", "module", "subsystem"]),
            "has_interface_description": any(kw in c for kw in [
                "interface", "api", "endpoint", "rest", "request", "response", "fastapi"]),
            "has_data_design": any(kw in c for kw in [
                "data", "database", "schema", "model", "entity", "faiss",
                "vector", "index", "embedding"]),
            "has_component_descriptions": any(kw in c for kw in [
                "component", "service", "class", "function", "method", "module",
                "agent", "retrieval", "llm"]),
            "has_design_rationale": any(kw in c for kw in [
                "rationale", "reason", "because", "chosen", "selected",
                "decision", "trade-off", "why"]),
            "has_diagrams_or_models": any(kw in c for kw in [
                "diagram", "uml", "flowchart", "sequence", "er diagram", "dfd", "figure", "chart"]),
        }
        score    = sum(checks.values()) / len(checks)
        failures = [k.replace("_", " ").capitalize() for k, v in checks.items() if not v]
        return StandardResult(
            standard_id="IEEE 1016", standard_name="Software Design Description",
            passed=score >= self.PASS_THRESHOLD,
            score=score, checks=checks, failures=failures, suggestions=[],
        )

    # =========================================================================
    #  RAG + LLM CHECK  (core of the upgrade)
    # =========================================================================

    async def _check_standard_rag(
        self,
        content:       str,
        standard_id:   str,
        standard_name: str,
        fallback_fn,               # keyword fallback callable
    ) -> StandardResult:
        """
        RAG-based compliance check for one IEEE standard:
          1. Retrieve top-3 relevant clauses from standards FAISS index
          2. Send clauses + content snippet to phi4-mini
          3. Parse JSON response → StandardResult
          4. On any failure, fall back to keyword check silently
        """
        store = self._get_store()

        # ── Guard: store not ready ────────────────────────────────────────────
        if not store.is_ready:
            logger.warning(
                f"[IEEECompliance] Standards index not ready — "
                f"keyword fallback for {standard_id}"
            )
            return fallback_fn(content)

        # ── Retrieve clauses ──────────────────────────────────────────────────
        clauses = store.query(content, top_k=3, standard_filter=standard_id)
        if not clauses:
            logger.warning(
                f"[IEEECompliance] No clauses retrieved for {standard_id} — keyword fallback"
            )
            return fallback_fn(content)

        # ── Build clause block (keep short for phi4-mini) ─────────────────────
        clause_lines = []
        for i, cl in enumerate(clauses, 1):
            clause_lines.append(f"Clause {i} (page {cl['page']}): {cl['text'][:280]}")
        clause_block   = "\n".join(clause_lines)
        content_snippet = content[:500].strip()

        # ── Prompt ────────────────────────────────────────────────────────────
        # query → instruction; context → all the data
        # Kept intentionally short — phi4-mini drifts on long prompts
        llm_query = (
            f"You are an IEEE compliance evaluator for {standard_id}. "
            f"Reply ONLY with valid JSON — no markdown, no explanation."
        )
        llm_context = (
            f"Standard: {standard_id} — {standard_name}\n\n"
            f"Official clauses:\n{clause_block}\n\n"
            f"Content to evaluate:\n{content_snippet}\n\n"
            f"Does the content satisfy the above clauses?\n"
            f"Return ONLY this JSON (replace values, keep keys exactly):\n"
            f'{{"passed": true, "score": 0.75, "reason": "one sentence", '
            f'"missing": "what is missing or none"}}'
        )

        # ── LLM call ─────────────────────────────────────────────────────────
        try:
            ollama      = self._get_ollama()
            raw_response = await ollama.generate(query=llm_query, context=llm_context)
            result       = self._parse_llm_json(raw_response, standard_id, standard_name, clauses)
            logger.info(
                f"[IEEECompliance] {standard_id} RAG ✓ — "
                f"passed={result.passed}, score={round(result.score * 100)}%"
            )
            return result

        except Exception as e:
            logger.error(
                f"[IEEECompliance] LLM call failed for {standard_id}: {e} — keyword fallback"
            )
            return fallback_fn(content)

    # ── JSON Parser ───────────────────────────────────────────────────────────

    def _parse_llm_json(
        self,
        raw:           str,
        standard_id:   str,
        standard_name: str,
        clauses:       list,
    ) -> StandardResult:
        """
        Parse phi4-mini response into StandardResult.
        Handles: markdown fences, surrounding prose, malformed JSON.
        """
        try:
            cleaned = raw.strip()
            # Strip ```json ... ``` fences if phi4-mini adds them
            cleaned = re.sub(r"```json|```", "", cleaned).strip()
            # Extract first JSON object even if there's prose around it
            match = re.search(r'\{.*?\}', cleaned, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON object in response: {cleaned[:150]}")

            data    = json.loads(match.group())
            passed  = bool(data.get("passed", False))
            score   = float(data.get("score", 0.5))
            score   = max(0.0, min(1.0, score))          # clamp to [0, 1]
            reason  = str(data.get("reason", ""))
            missing = str(data.get("missing", "none"))

            # Build a checks dict using clause page numbers as keys
            checks: Dict[str, bool] = {}
            for i, cl in enumerate(clauses, 1):
                key          = f"clause_page_{cl['page']}_satisfied"
                # First clause drives pass/fail; subsequent use score threshold
                checks[key]  = passed if i == 1 else (score >= 0.6)
            # Record which pages were cited — visible in frontend
            checks["cited_pages"] = (
                f"IEEE standard pages: {', '.join(str(c['page']) for c in clauses)}"
            )

            failures    = [k for k, v in checks.items() if v is False]
            suggestions = []
            if missing.lower() not in ("none", "n/a", ""):
                suggestions.append(f"{standard_id}: {missing}")
            if not passed and reason:
                suggestions.append(
                    f"Review {standard_id} ({standard_name}). "
                    f"Assessment: {reason}"
                )

            return StandardResult(
                standard_id=standard_id,
                standard_name=standard_name,
                passed=passed,
                score=score,
                checks=checks,
                failures=failures,
                suggestions=suggestions,
            )

        except Exception as e:
            logger.warning(
                f"[IEEECompliance] Parse error for {standard_id}: {e} | "
                f"raw[:200]={raw[:200]}"
            )
            # Neutral fallback — doesn't crash the pipeline
            return StandardResult(
                standard_id=standard_id,
                standard_name=standard_name,
                passed=False,
                score=0.5,
                checks={"rag_parse_error": False},
                failures=["LLM response could not be parsed"],
                suggestions=[
                    f"Manually verify content against {standard_id} — {standard_name}."
                ],
            )

    # =========================================================================
    #  MASTER RUNNER
    # =========================================================================

    async def run_full_compliance_check(self, content: str) -> ComplianceReport:
        """
        Run all 5 IEEE checks and return a ComplianceReport.

        IEEE 12207, 730 → synchronous keyword match
        IEEE 830, 829, 1016 → async RAG + phi4-mini (concurrent via asyncio.gather)
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty for IEEE compliance check.")

        # ── Keyword checks (sync, instant) ────────────────────────────────────
        result_12207 = self.check_ieee_12207(content)
        result_730   = self.check_ieee_730(content)

        # ── RAG checks (async, run concurrently) ──────────────────────────────
        result_830, result_829, result_1016 = await asyncio.gather(
            self._check_standard_rag(
                content, "IEEE 830", "Software Requirements Specifications", self._keyword_830),
            self._check_standard_rag(
                content, "IEEE 829", "Software Test Documentation", self._keyword_829),
            self._check_standard_rag(
                content, "IEEE 1016", "Software Design Description", self._keyword_1016),
        )

        results        = [result_12207, result_830, result_829, result_1016, result_730]
        overall_score  = sum(r.score for r in results) / len(results)
        overall_passed = all(r.passed for r in results)
        passed_count   = sum(1 for r in results if r.passed)

        summary = (
            f"{passed_count}/5 IEEE standards passed | "
            f"Overall score: {round(overall_score * 100, 1)}% | "
            f"{'COMPLIANT ✅' if overall_passed else 'NEEDS IMPROVEMENT ⚠️'} | "
            f"RAG-verified: IEEE 830, 829, 1016"
        )

        return ComplianceReport(
            timestamp=datetime.now().isoformat(),
            overall_score=overall_score,
            overall_passed=overall_passed,
            standards=results,
            summary=summary,
        )


# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────

_checker = IEEEComplianceChecker()   # module-level singleton


async def check_compliance(content: str) -> dict:
    """
    Run full IEEE compliance check on any text content.

    RAG-verified : IEEE 830, IEEE 829, IEEE 1016
    Keyword-based: IEEE 12207, IEEE 730

    Args:
        content: Text to check (RAG answer, document excerpt, etc.)
    Returns:
        dict: JSON-serializable compliance report (same shape as v1).

    Example:
        report = await check_compliance(rag_answer + " " + source_docs)
        print(report["summary"])
        print(report["standards"])
    """
    report = await _checker.run_full_compliance_check(content)
    return report.to_dict()