"""Convenience launcher for the AgriCast API.

Usage:
    .venv\\Scripts\\python.exe run.py
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
