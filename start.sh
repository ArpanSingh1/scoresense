#!/usr/bin/env sh
# Start ScoreSense on macOS or Linux; it opens http://localhost:8000. Ctrl+C stops it.
cd "$(dirname "$0")/eval"
exec ../venv/bin/python app.py
