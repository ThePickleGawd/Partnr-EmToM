"""
LLM-based surprise detection for EMTOM benchmark.

Uses LLM self-report to identify when agents encounter unexpected behaviors
that might indicate interesting mechanics worth investigating.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SurpriseAssessment:
    """Result of surprise detection."""

    is_surprised: bool
    level: int  # 1-5 scale
    explanation: str
    hypothesis: str  # Agent's guess about why this happened
    raw_response: str = ""


class SurpriseDetector:
    """
    LLM-based surprise detection through self-report.

    After each action, asks the LLM to evaluate whether the outcome
    was surprising and why. This identifies moments where the agent
    has discovered unexpected behaviors in the world.
    """

    PROMPT_TEMPLATE = """You just performed an action and observed the result.

Action: {action}
Target: {target}
Expected outcome: {expected}
Actual outcome: {actual}
{trigger_note}

Were you surprised by this outcome? Consider:
- Did the result match your expectations?
- Did anything unusual or unexpected happen?
- Does this reveal something about how this world works?

Respond in JSON format:
{{
    "is_surprised": true/false,
    "level": 1-5 (1=slightly unexpected, 5=completely unexpected),
    "explanation": "<why you were or weren't surprised>",
    "hypothesis": "<if surprised, what might explain this behavior?>"
}}"""

    def __init__(
        self,
        llm_client: Any,
        surprise_threshold: float = 0.5,
        require_hypothesis: bool = True,
    ):
        """
        Initialize the surprise detector.

        Args:
            llm_client: LLM client with generate(prompt) method
            surprise_threshold: Minimum surprise level to report (0-1)
            require_hypothesis: Whether to require hypothesis in response
        """
        self.llm = llm_client
        self.surprise_threshold = surprise_threshold
        self.require_hypothesis = require_hypothesis

    def assess_surprise(
        self,
        agent_id: str,
        action: str,
        target: Optional[str],
        expected: str,
        actual: str,
        trigger: Optional[str] = None,
    ) -> SurpriseAssessment:
        """
        Assess whether an action outcome was surprising.

        Args:
            agent_id: ID of the agent
            action: Action that was performed
            target: Target of the action
            expected: What the agent expected to happen
            actual: What actually happened
            trigger: Optional system hint about surprise

        Returns:
            SurpriseAssessment with surprise level and explanation
        """
        trigger_note = ""
        if trigger:
            trigger_note = f"\nSystem note: {trigger}"

        prompt = self.PROMPT_TEMPLATE.format(
            action=action,
            target=target or "none",
            expected=expected or "default behavior",
            actual=actual,
            trigger_note=trigger_note,
        )

        response = self.llm.generate(prompt)
        return self._parse_response(response)

    def _parse_response(self, response: str) -> SurpriseAssessment:
        """Parse LLM response into SurpriseAssessment."""
        try:
            # Find JSON in response
            json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return SurpriseAssessment(
                    is_surprised=bool(data.get("is_surprised", False)),
                    level=int(data.get("level", 1)),
                    explanation=data.get("explanation", ""),
                    hypothesis=data.get("hypothesis", ""),
                    raw_response=response,
                )
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: try to infer from text
        is_surprised = any(
            word in response.lower()
            for word in ["surprised", "unexpected", "strange", "unusual"]
        )

        return SurpriseAssessment(
            is_surprised=is_surprised,
            level=3 if is_surprised else 1,
            explanation="Could not parse structured response",
            hypothesis="",
            raw_response=response,
        )

    def is_significant_surprise(self, assessment: SurpriseAssessment) -> bool:
        """Check if a surprise meets the significance threshold."""
        if not assessment.is_surprised:
            return False
        # Normalize level to 0-1 scale
        normalized_level = (assessment.level - 1) / 4.0
        return normalized_level >= self.surprise_threshold


class RuleBasedSurpriseDetector:
    """
    Rule-based surprise detection for comparison/fallback.

    Uses predefined rules to identify surprising outcomes without LLM.
    """

    def __init__(self):
        self.rules = [
            # Inverse effects
            self._check_inverse_effect,
            # No effect when expected
            self._check_no_effect,
            # Effect on wrong target
            self._check_wrong_target,
        ]

    def assess_surprise(
        self,
        agent_id: str,
        action: str,
        target: Optional[str],
        expected: str,
        actual: str,
        trigger: Optional[str] = None,
    ) -> SurpriseAssessment:
        """Assess surprise using rules."""
        for rule in self.rules:
            assessment = rule(action, target, expected, actual)
            if assessment.is_surprised:
                return assessment

        # No surprise detected
        return SurpriseAssessment(
            is_surprised=False,
            level=1,
            explanation="Outcome matches expectations",
            hypothesis="",
        )

    def _check_inverse_effect(
        self,
        action: str,
        target: Optional[str],
        expected: str,
        actual: str,
    ) -> SurpriseAssessment:
        """Check if effect is inverse of expected."""
        inverse_pairs = {
            ("open", "closed"): "inverse",
            ("close", "open"): "inverse",
            ("on", "off"): "inverse",
            ("off", "on"): "inverse",
        }

        for (exp_word, act_word), surprise_type in inverse_pairs.items():
            if exp_word in expected.lower() and act_word in actual.lower():
                return SurpriseAssessment(
                    is_surprised=True,
                    level=4,
                    explanation=f"Action had inverse effect: expected {exp_word}, got {act_word}",
                    hypothesis="This object may have inverted controls",
                )

        return SurpriseAssessment(is_surprised=False, level=1, explanation="", hypothesis="")

    def _check_no_effect(
        self,
        action: str,
        target: Optional[str],
        expected: str,
        actual: str,
    ) -> SurpriseAssessment:
        """Check if action had no effect when one was expected."""
        no_effect_phrases = ["nothing happens", "no effect", "nothing changed"]

        if any(phrase in actual.lower() for phrase in no_effect_phrases):
            if "nothing" not in expected.lower():
                return SurpriseAssessment(
                    is_surprised=True,
                    level=3,
                    explanation="Expected an effect, but nothing happened",
                    hypothesis="There may be a hidden condition required",
                )

        return SurpriseAssessment(is_surprised=False, level=1, explanation="", hypothesis="")

    def _check_wrong_target(
        self,
        action: str,
        target: Optional[str],
        expected: str,
        actual: str,
    ) -> SurpriseAssessment:
        """Check if effect occurred on a different target."""
        if target and target not in actual and "affected" in actual.lower():
            return SurpriseAssessment(
                is_surprised=True,
                level=4,
                explanation=f"Action on {target} affected something else",
                hypothesis="This control may be connected to something remote",
            )

        return SurpriseAssessment(is_surprised=False, level=1, explanation="", hypothesis="")


class HybridSurpriseDetector:
    """
    Combines rule-based and LLM-based surprise detection.

    Uses rules as a fast first pass, then LLM for uncertain cases.
    """

    def __init__(
        self,
        llm_client: Any,
        use_llm_threshold: float = 0.5,
    ):
        self.rule_detector = RuleBasedSurpriseDetector()
        self.llm_detector = SurpriseDetector(llm_client)
        self.use_llm_threshold = use_llm_threshold

    def assess_surprise(
        self,
        agent_id: str,
        action: str,
        target: Optional[str],
        expected: str,
        actual: str,
        trigger: Optional[str] = None,
    ) -> SurpriseAssessment:
        """Assess surprise using hybrid approach."""
        # First try rules
        rule_assessment = self.rule_detector.assess_surprise(
            agent_id, action, target, expected, actual, trigger
        )

        # If rules found clear surprise, use that
        if rule_assessment.is_surprised and rule_assessment.level >= 4:
            return rule_assessment

        # If trigger provided, more likely to need LLM
        if trigger:
            return self.llm_detector.assess_surprise(
                agent_id, action, target, expected, actual, trigger
            )

        # Otherwise use rule result
        return rule_assessment


def create_surprise_detector(
    detector_type: str = "llm",
    llm_client: Any = None,
    **kwargs,
) -> SurpriseDetector:
    """
    Factory function to create surprise detectors.

    Args:
        detector_type: "llm", "rules", or "hybrid"
        llm_client: LLM client (required for "llm" and "hybrid")
        **kwargs: Additional arguments

    Returns:
        Surprise detector instance
    """
    if detector_type == "llm":
        if llm_client is None:
            raise ValueError("llm_client required for LLM surprise detector")
        return SurpriseDetector(llm_client, **kwargs)
    elif detector_type == "rules":
        return RuleBasedSurpriseDetector()
    elif detector_type == "hybrid":
        if llm_client is None:
            raise ValueError("llm_client required for hybrid surprise detector")
        return HybridSurpriseDetector(llm_client, **kwargs)
    else:
        raise ValueError(f"Unknown surprise detector type: {detector_type}")
