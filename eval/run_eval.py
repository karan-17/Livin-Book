"""Evaluate the agent on a LangSmith dataset with an LLM judge + simple heuristics.

    python eval/run_eval.py                      # all questions
    python eval/run_eval.py --limit 4            # quick smoke run
    python eval/run_eval.py --recreate           # re-upload questions.jsonl to LangSmith

Requires LANGSMITH_API_KEY (and LANGSMITH_TRACING=true to see traces). Each question costs
~3-6 Gemini calls plus ~1 judge call, so on the free tier keep --concurrency at 1.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langsmith import Client  # noqa: E402

from novel_agent import prompts  # noqa: E402
from novel_agent.config import get_llm, get_settings  # noqa: E402
from novel_agent.graph import build_graph  # noqa: E402
from novel_agent.memory import in_memory  # noqa: E402
from novel_agent.resources import load_resources  # noqa: E402
from novel_agent.runner import stream_turn  # noqa: E402

QUESTIONS = Path(__file__).with_name("questions.jsonl")


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_dataset(client: Client, name: str, recreate: bool) -> None:
    if client.has_dataset(dataset_name=name):
        if not recreate:
            return
        client.delete_dataset(dataset_name=name)
    dataset = client.create_dataset(name, description="Novel agent reference Q&A (characters, events, "
                                                      "chapters, relationships, themes, ending, multi-turn)")
    client.create_examples(dataset_id=dataset.id, examples=[
        {"inputs": q["inputs"], "outputs": q["outputs"], "metadata": {"category": q["category"], "id": q["id"]}}
        for q in load_questions()
    ])
    print(f"Uploaded {len(load_questions())} examples to dataset {name!r}")


# ------------------------------------------------------------------------------ target


def make_target(book_id: str):
    resources = load_resources(book_id)
    checkpointer, store = in_memory()
    graph = build_graph(resources, checkpointer=checkpointer, store=store)

    def target(inputs: dict) -> dict:
        """Runs one example on a fresh thread. Multi-turn examples replay all turns in order."""
        thread_id = f"eval-{uuid.uuid4().hex[:8]}"
        turns = inputs.get("turns") or [inputs["question"]]
        answer, tools_used = "", []
        for question in turns:
            for event in stream_turn(graph, question, thread_id=thread_id, user_id="eval-user", book_id=book_id):
                if event["type"] == "tool_call":
                    tools_used.append(event["name"])
                elif event["type"] == "final":
                    answer = event["text"]
        return {"answer": answer, "tools_used": tools_used}

    return target, resources


# ------------------------------------------------------------------------------ evaluators


class Grade(BaseModel):
    reasoning: str = Field(description="Brief comparison of the answer with the reference")
    score: float = Field(description="1.0 correct, 0.5 partially correct, 0.0 wrong or a refusal")


def make_correctness_judge(title: str):
    judge = get_llm("judge").with_structured_output(Grade)

    def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        question = inputs.get("question") or " → ".join(inputs.get("turns", []))
        grade = judge.invoke([
            ("system", prompts.JUDGE_SYSTEM.format(title=title)),
            ("human", f"Question: {question}\n\nReference answer: {reference_outputs['answer']}\n\n"
                      f"Assistant's answer: {outputs.get('answer', '')}"),
        ])
        return {"key": "correctness", "score": grade.score, "comment": grade.reasoning}

    return correctness


def cites_chapters(outputs: dict) -> dict:
    """Does the answer cite at least one chapter, as the system prompt requires?"""
    cited = bool(re.search(r"\b(?:Ch\.?|Chapters?)\s*\d+", outputs.get("answer", "")))
    return {"key": "cites_chapters", "score": int(cited)}


def used_retrieval(outputs: dict) -> dict:
    """Did the agent ground itself with at least one tool call?"""
    return {"key": "used_tools", "score": int(bool(outputs.get("tools_used")))}


# ------------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--book", default="pride_and_prejudice")
    parser.add_argument("--dataset", default=None, help="LangSmith dataset name")
    parser.add_argument("--limit", type=int, default=0, help="only evaluate the first N examples")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--recreate", action="store_true", help="re-upload questions.jsonl")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    client = Client()
    dataset = args.dataset or f"novel-agent-{args.book}"
    ensure_dataset(client, dataset, args.recreate)
    data = client.list_examples(dataset_name=dataset, limit=args.limit) if args.limit else dataset

    target, resources = make_target(args.book)
    settings = get_settings()
    results = client.evaluate(
        target,
        data=data,
        evaluators=[make_correctness_judge(resources.title), cites_chapters, used_retrieval],
        experiment_prefix=f"{args.book}-{settings.agent_model}",
        metadata={"agent_model": settings.agent_model, "embeddings": settings.embed_model,
                  "rerank": settings.rerank_enabled},
        max_concurrency=args.concurrency,
    )

    totals: dict[str, list[float]] = {}
    for row in results:
        category = (row["example"].metadata or {}).get("category", "?")
        for res in row["evaluation_results"]["results"]:
            if res.score is not None:
                totals.setdefault(res.key, []).append(float(res.score))
                if res.key == "correctness":
                    print(f"  [{res.score:.1f}] {category:<12} {row['example'].metadata.get('id')}")
    print("\nAverages:")
    for key, scores in totals.items():
        print(f"  {key:<15} {sum(scores) / len(scores):.2f}  (n={len(scores)})")
    print(f"\nExperiment: {results.experiment_name}")


if __name__ == "__main__":
    main()
