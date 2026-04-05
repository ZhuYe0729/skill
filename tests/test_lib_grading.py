from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

lib_grading = importlib.import_module("lib_grading")
_combine_grades = lib_grading._combine_grades
_normalize_judge_response = lib_grading._normalize_judge_response
GradeResult = lib_grading.GradeResult
grade_task = lib_grading.grade_task


class JudgeNormalizationTests(unittest.TestCase):
    def test_normalize_judge_response_averages_summed_total_when_breakdown_is_unit_scale(
        self,
    ) -> None:
        parsed = {
            "scores": {
                "coverage": 0.75,
                "synthesis": 0.75,
                "structure": 0.75,
                "tone": 0.8,
                "conciseness": 0.8,
            },
            "total": 3.85,
            "notes": "Summed by mistake",
        }

        normalized = _normalize_judge_response(parsed)

        self.assertAlmostEqual(normalized["total"], 0.77)

    def test_hybrid_score_uses_normalized_judge_total(self) -> None:
        auto = GradeResult(
            task_id="task_16_email_triage",
            score=0.7062937062937062,
            max_score=1.0,
            grading_type="automated",
            breakdown={},
            notes="",
        )
        judge = GradeResult(
            task_id="task_16_email_triage",
            score=0.87,
            max_score=1.0,
            grading_type="llm_judge",
            breakdown={},
            notes="",
        )

        class _Task:
            task_id = "task_16_email_triage"
            grading_weights = {"automated": 0.4, "llm_judge": 0.6}

        combined = _combine_grades(_Task(), auto, judge)

        self.assertAlmostEqual(combined.score, 0.8045174825174824)

    def test_grade_task_uses_runtime_for_llm_judge(self) -> None:
        class _Task:
            task_id = "task_runtime_judge"
            grading_type = "llm_judge"
            llm_judge_rubric = None
            grading_criteria = ["Correctness"]
            prompt = "Do the thing"
            expected_behavior = "Answer correctly"
            grading_weights = None

        class _Runtime:
            called = False

            def run_judge_prompt(self, **kwargs):
                self.called = True
                return {
                    "status": "success",
                    "transcript": [
                        {
                            "type": "message",
                            "message": {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": '{"scores": {"correctness": 0.9}, "total": 0.9, "notes": "ok"}',
                                    }
                                ],
                            },
                        }
                    ],
                }

        runtime = _Runtime()
        execution_result = {
            "status": "success",
            "transcript": [
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "done"}],
                    },
                }
            ],
            "workspace": "",
        }

        with TemporaryDirectory() as tmpdir:
            result = grade_task(
                task=_Task(),
                execution_result=execution_result,
                skill_dir=Path(tmpdir),
                runtime=runtime,
            )

        self.assertTrue(runtime.called)
        self.assertEqual(result.score, 0.9)
        self.assertEqual(result.breakdown, {"correctness": 0.9})


if __name__ == "__main__":
    unittest.main()
