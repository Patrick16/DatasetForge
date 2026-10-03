"""Dev-server entrypoint for the Claude Code preview pane.

Identical to `uvicorn app.main:app --reload`, except the port comes from the
$PORT environment variable (falling back to 8000) instead of being
hardcoded -- lets the preview pane's autoPort assignment pick a free port
when 8000 is already taken by something else on the machine.
"""
from __future__ import annotations

import os

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="127.0.0.1", port=port, reload=True)
