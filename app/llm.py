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
            max_retries=settings.llm_max_retries,
        )

    def answer_question(self, question: str, context: str, max_retries: int = 2) -> LlmAnswer:
        prompt = build_prompt(question, context)
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._settings.llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert Vietnamese exam answering assistant. "
                                "Return exactly one character: A, B, C, or D. "
                                "No explanation, no punctuation."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    timeout=self._settings.llm_timeout_seconds,
                )
                raw_answer = response.choices[0].message.content or ""
                answer = normalize_answer(raw_answer)
                if answer is None:
                    raise LlmAnswerParseError(
                        f"Could not parse answer from model output: {raw_answer!r}"
                    )
                return LlmAnswer(answer=answer, raw_answer=raw_answer)
            except APITimeoutError as error:
                last_error = error
                if attempt < max_retries:
                    continue
            except (APIConnectionError, APIStatusError) as error:
                last_error = error
                if attempt < max_retries:
                    continue

        if isinstance(last_error, APITimeoutError):
            raise TeacherProxyTimeoutError("Teacher proxy request timed out") from last_error
        raise TeacherProxyRequestError("Teacher proxy request failed") from last_error


def build_prompt(question: str, context: str) -> str:
    return f"""
You are answering a Vietnamese multiple-choice question using ONLY the provided document context.

CRITICAL RULES:
1. Read ALL the context carefully before answering.
2. Identify whether the question asks for the correct, incorrect, or exception option.
3. Match the MEANING of each option against the context, not just exact words.
4. If the question asks about a definition, find the exact definition in context.
5. If the question asks about a list/steps/process, verify each option against the context.
6. Choose the option with the STRONGEST and most DIRECT evidence from the context.
7. If no option is perfectly supported, choose the CLOSEST match.
8. Return ONLY one character: A, B, C, or D.

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
