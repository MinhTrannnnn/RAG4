from __future__ import annotations

import re
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from app.config import Settings


ANSWER_PATTERNS = [
    re.compile(r"\bANSWER\s*[:\-]?\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"\bDAP\s*AN\s*[:\-]?\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"\b([ABCD])\b", re.IGNORECASE),
]


class TeacherProxyTimeoutError(RuntimeError):
    """Teacher proxy timed out."""


class TeacherProxyRequestError(RuntimeError):
    """Teacher proxy request failed."""


class LlmAnswerParseError(RuntimeError):
    """Teacher proxy returned text that did not contain A/B/C/D."""


@dataclass(frozen=True)
class LlmAnswer:
    answer: str
    raw_answer: str


class LlmService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(
            base_url=settings.teacher_proxy_base_url,
            api_key=settings.student_id,
        )

    def answer_question(self, question: str, context: str) -> LlmAnswer:
        prompt = build_prompt(question, context)
        try:
            response = self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Return exactly one character only: A, B, C, or D.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                timeout=self._settings.llm_timeout_seconds,
            )
        except APITimeoutError as error:
            raise TeacherProxyTimeoutError("Teacher proxy request timed out") from error
        except (APIConnectionError, APIStatusError) as error:
            raise TeacherProxyRequestError("Teacher proxy request failed") from error

        raw_answer = response.choices[0].message.content or ""
        answer = normalize_answer(raw_answer)
        if answer is None:
            raise LlmAnswerParseError(f"Could not parse answer from model output: {raw_answer!r}")
        return LlmAnswer(answer=answer, raw_answer=raw_answer)


def build_prompt(question: str, context: str) -> str:
    return f"""
You are answering a Vietnamese multiple-choice question using only the provided document context.

Required rules:
- Return exactly one character: A, B, C, or D.
- Do not explain.
- Compare the meaning of each option with the context, not just exact words.
- Prefer the option that is directly supported by the context.
- The question already contains the full answer options.
- If the context is not enough, choose the most likely option from the question and context.

Relevant document context:
{context}

Question:
{question}

Final answer:
""".strip()


def normalize_answer(raw_text: str) -> str | None:
    text = raw_text.strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text

    for pattern in ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()

    for char in text:
        if char in "ABCD":
            return char

    return None
