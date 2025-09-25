"""Command-line entrypoint for the CheerSearch agent.

Usage examples (after installing requirements):
  set GROQ_API_KEY=<your_key_here> or API_KEY=<your_key_here>
  set GROQ_MODEL=<model_name>
  set AGENT_MAX_STEPS=<max_steps>
  python main.py

This minimal CLI is provided instead of a web UI to keep the
core agent architecture clear and easily testable. A Streamlit or
FastAPI interface can be layered on top later.
"""
from __future__ import annotations

import os
import shlex
import sys
from typing import Optional

from agents.cheersearch_agent import build_agent
from langchain.agents import AgentExecutor


def ensure_api_key() -> str:
	key = os.getenv("GROQ_API_KEY") or os.getenv("API_KEY")
	if not key:
		print("ERROR: GROQ_API_KEY (or API_KEY) environment variable not set.")
		print("Set it with: set GROQ_API_KEY=your_key_here or API_KEY=your_key_here")
		sys.exit(1)
	return key


def print_banner():
	print("CheerSearch Agent - Multimodal CLI")


def handle_input(agent: AgentExecutor, args: list[str]):
	user_prompt = " ".join(args).strip()
	print("[Agent] Thinking...")
	try:
		result = agent.invoke({"input": user_prompt})
		response = result.get("output")
	except Exception as e:
		msg = str(e)
		if "insufficient_quota" in msg or "429" in msg:
			print("Rate/quota issue. Try a smaller model or different provider.")
		else:
			print(f"Error: {e}")
		return
	print(response)


def repl(agent: AgentExecutor):
	print_banner()
	while True:
		try:
			line = input("> ")
		except (EOFError, KeyboardInterrupt):
			print("\nExiting.")
			break
		if not line:
			continue
		if line.lower() in {"exit", "quit"}:
			print("Bye.")
			break
		if line.lower() == "help":
			print(__doc__)
			continue
		# Chat with the agent
		try:
			handle_input(agent, shlex.split(line))
		except Exception as e:
			msg = str(e)
			if "insufficient_quota" in msg or "429" in msg:
				print("Rate/quota exceeded. Switch MODEL or provider.")
			else:
				print(f"Error: {e}")
			continue


def main():
	api_key = ensure_api_key()
	model_name = os.getenv("GROQ_MODEL") or os.getenv("MODEL")
	max_steps_env = os.getenv("AGENT_MAX_STEPS")
	try:
		max_steps = int(max_steps_env) if max_steps_env else None
	except ValueError:
		print("Invalid AGENT_MAX_STEPS value")
		max_steps = None
	print(f"Using Groq model: {model_name} (max_steps={max_steps})\n")
	agent = build_agent(groq_api_key=api_key, model_name=model_name, max_steps=max_steps)
	repl(agent)


if __name__ == "__main__":
	main()

