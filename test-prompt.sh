#!/usr/bin/env bash
# Quick smoke test for a running SGLang server.
# Usage: ./test-prompt.sh [host:port]

ENDPOINT="${1:-127.0.0.1:${SGLANG_PORT:-30000}}"

curl -s "http://${ENDPOINT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/Phi-4-mini-instruct-FP8-TORCHAO",
    "messages": [{"role": "user", "content": "Explain what a tokenizer does in three sentences."}],
    "max_tokens": 256
  }' | python3 -m json.tool
