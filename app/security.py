# app/security.py
import hmac

from fastapi import Header, HTTPException, Request


async def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """Reject callers without the shared key. A public URL that spends money needs a lock."""
    expected = request.app.state.settings.app_api_key
    if not expected:
        return  # no key configured: local development
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")
