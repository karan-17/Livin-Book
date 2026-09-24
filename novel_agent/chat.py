"""Terminal chat with the novel agent (multi-turn, persistent memory).

    python -m novel_agent.chat --book pride_and_prejudice [--user alice] [--thread t1] [--quiet]

Commands: /new  /thread <id>  /progress <n|0>  /prefs  /history  /help  /quit
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import uuid

from . import memory
from .graph import build_graph
from .resources import load_resources
from .runner import conversation, stream_turn

HELP = """Commands:
  /new             start a new conversation thread
  /thread <id>     switch to (or resume) a thread
  /progress <n>    set how far you've read (spoiler guard); /progress 0 turns it off
  /prefs           show saved preferences   (/prefs clear to reset)
  /history         show this thread's conversation
  /quit            exit"""

DIM, BOLD, CYAN, YELLOW, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[0m"


def _short(value, width: int = 110) -> str:
    return textwrap.shorten(" ".join(str(value).split()), width=width, placeholder=" …")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chat with an ingested book.")
    parser.add_argument("--book", default="pride_and_prejudice")
    parser.add_argument("--user", default="workshop-user", help="reader id for long-term memory")
    parser.add_argument("--thread", default=None, help="conversation thread id to resume")
    parser.add_argument("--quiet", action="store_true", help="hide the plan and tool calls")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252; the novel uses curly quotes
        sys.stdout.reconfigure(encoding="utf-8")
    if os.name == "nt":
        os.system("")  # enables ANSI colour codes in the classic Windows console
    resources = load_resources(args.book)
    checkpointer, store = memory.sqlite_memory()
    graph = build_graph(resources, checkpointer=checkpointer, store=store)
    thread_id = args.thread or uuid.uuid4().hex[:8]

    print(f"{BOLD}Novel Agent — {resources.title} by {resources.author}{RESET}")
    print(f"{DIM}user={args.user} thread={thread_id}   /help for commands{RESET}\n")

    while True:
        try:
            question = input(f"{BOLD}you ▸ {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.startswith("/"):
            cmd, _, arg = question.partition(" ")
            if cmd in {"/quit", "/exit"}:
                break
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/new":
                thread_id = uuid.uuid4().hex[:8]
                print(f"{DIM}new thread {thread_id}{RESET}")
            elif cmd == "/thread" and arg:
                thread_id = arg.strip()
                print(f"{DIM}switched to thread {thread_id}{RESET}")
            elif cmd == "/progress" and arg.strip().isdigit():
                chapter = int(arg)
                memory.set_reading_progress(store, args.user, resources.book_id, chapter or None)
                print(f"{DIM}spoiler guard {'ON up to chapter ' + str(chapter) if chapter else 'OFF'}{RESET}")
            elif cmd == "/prefs":
                if arg.strip() == "clear":
                    memory.clear_preferences(store, args.user)
                print(f"{DIM}{memory.get_reader_profile(store, args.user)}{RESET}")
            elif cmd == "/history":
                for role, text in conversation(graph, thread_id):
                    print(f"{BOLD if role == 'user' else CYAN}{role}:{RESET} {text}\n")
            else:
                print(HELP)
            continue

        print(f"{CYAN}agent ▸ {RESET}", end="", flush=True)
        streamed = False
        try:
            for event in stream_turn(graph, question, thread_id=thread_id, user_id=args.user,
                                     book_id=resources.book_id):
                kind = event["type"]
                if kind == "token":
                    print(event["text"], end="", flush=True)
                    streamed = True
                elif args.quiet:
                    continue
                elif kind == "plan":
                    plan = event["analysis"]
                    print(f"\n{DIM}  plan [{plan.get('question_type')}]: {_short(plan.get('standalone_question'))}")
                    for step in plan.get("sub_questions", []):
                        print(f"   • {_short(step)}")
                    print(RESET, end="")
                elif kind == "tool_call":
                    args_text = ", ".join(f"{k}={v!r}" for k, v in event["args"].items())
                    print(f"\n{YELLOW}  ↳ {event['name']}({_short(args_text, 90)}){RESET}", end="")
                    streamed = False
                elif kind == "memory":
                    print(f"\n{DIM}  (older turns folded into the conversation summary){RESET}", end="")
                elif kind == "final" and not streamed:
                    print(f"\n{event['text']}", end="")
            print("\n")
        except KeyboardInterrupt:
            print(f"\n{DIM}(interrupted){RESET}\n")
        except Exception as exc:  # keep the REPL alive on API errors (quota, network, ...)
            print(f"\n{YELLOW}Error: {exc}{RESET}\n")


if __name__ == "__main__":
    main()
