#!/bin/bash

export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH

python -m uvicorn chat:app --host 0.0.0.0 --port 8000