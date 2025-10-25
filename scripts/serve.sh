#!/usr/bin/env bash
uvicorn src.api.fastapi_app:app --reload --port 8000
