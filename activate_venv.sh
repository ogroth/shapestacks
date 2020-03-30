#!/bin/bash
SCRIPT_PATH=$(readlink -f "$0")
SHAPESTACKS_CODE_HOME=$(dirname ${SCRIPT_PATH})
export SHAPESTACKS_CODE_HOME
echo "Set environment variable SHAPESTACKS_CODE_HOME=${SHAPESTACKS_CODE_HOME}"
source venv/bin/activate
echo "Activated virtual environment 'venv'."
