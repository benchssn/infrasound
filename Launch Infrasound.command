#!/bin/zsh
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Set up the Python environment first; see README.md.'
  read -r '?Press Enter to close.'
  exit 1
fi
.venv/bin/python infrasound.py "$@"
