#!/bin/bash
cd ~/Desktop/shorts || exit
source venv/bin/activate
if ! curl -s http://localhost:11434 >/dev/null; then
  echo "Starting Ollama..."
  open -a Ollama
  sleep 8
fi
ollama list | grep -q "qwen2.5:7b" || ollama pull qwen2.5:7b
python shorts.py
open shorts_output
echo "Done. Press Enter to close."
read
