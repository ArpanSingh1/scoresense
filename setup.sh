#!/usr/bin/env sh
# One-time setup on macOS or Linux: Python packages and the AI model.
# Needs Python 3.11+ and Ollama installed first (see README.md). Then start with ./start.sh
set -e
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "Install Python 3.11+ first."; exit 1; }
command -v ollama >/dev/null || { echo "Install Ollama first: https://ollama.com/download"; exit 1; }
[ -x venv/bin/python ] || python3 -m venv venv
venv/bin/python -m pip install --upgrade pip >/dev/null
venv/bin/python -m pip install -r eval/requirements.txt
ollama pull qwen2.5vl:7b
echo "Setup finished. Start ScoreSense with ./start.sh"
