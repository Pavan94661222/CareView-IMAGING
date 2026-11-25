"""Password hashing, signed browser sessions, CSRF, and origin checks."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Request, Response, status

from . import clinical_store

SESSION_COOKIE = "clinical_session"
SESSION_TTL_SECONDS = 12 * 60 * 60


def _secret_key() -> bytes:
    configured = os.getenv("APP_SECRET_KEY")
    if configured:
        if len(configured) < 32:
            raise RuntimeError("APP_SECRET_KEY must contain at least 32 characters.")
        return configured.encode("utf-8")

    secret_path = clinical_store.data_dir() / ".session_secret"
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with secret_path.open("x", encoding="ascii") as handle:
            handle.write(secrets.token_urlsafe(48))
    except FileExistsError:
        pass
    secret = secret_path.read_text(encoding="ascii").strip()
    if len(secret) < 32:
        raise RuntimeError(f"Invalid application signing key at {secret_path}.")
    return secret.encode("ascii")


def hash_password(password: str) -> str:
    if not 12 <= len(password) <= 256:
        raise ValueError("Password must be between 12 and 256 characters.")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt$16384$8$1${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, n, r, p, salt_hex, digest_hex = stored_hash.split("$")
        if algorithm != "scrypt":
            return False
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(salt_hex),
            n=int(n),
            r=int(r),
            p=int(p),
        )
        return hmac.compare_digest(derived.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def create_session_token(user: Dict[str, Any]) -> str:
    expires = int(time.time()) + SESSION_TTL_SECONDS
    payload = f"{user['id']}|{int(user['session_version'])}|{expires}|{secrets.token_urlsafe(12)}"
    encoded = _b64encode(payload.encode("utf-8"))
    signature = hmac.new(_secret_key(), encoded.encode("ascii"), hashlib.sha256).digest()
    return f"{encoded}.{_b64encode(signature)}"


def decode_session_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        encoded, supplied_signature = token.split(".", 1)
        expected = hmac.new(_secret_key(), encoded.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64decode(supplied_signature)):
            return None
        user_id, version, expires, _nonce = _b64decode(encoded).decode("utf-8").split("|", 3)
        if int(expires) <= int(time.time()):
            return None
        return {"user_id": user_id, "session_version": int(version)}
    except (ValueError, UnicodeDecodeError):
        return None


def csrf_for_session(token: str) -> str:
    return hmac.new(
        _secret_key(),
        f"csrf:{token}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def set_session(response: Response, user: Dict[str, Any]) -> str:
    token = create_session_token(user)
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=os.getenv("APP_COOKIE_SECURE", "false").strip().lower() in {"1", "true", "yes"},
        samesite="strict",
        path="/",
    )
    return csrf_for_session(token)


def clear_session(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="strict")


def public_user(user: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": user["id"],
        "email": user["email"],
        "display_name": user["display_name"],
        "created_at": user["created_at"],
    }


def get_current_user(request: Request) -> Dict[str, Any]:
    token = request.cookies.get(SESSION_COOKIE)
    session = decode_session_token(token) if token else None
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Please sign in.")
    user = clinical_store.user_by_id(session["user_id"])
    if not user or int(user["session_version"]) != session["session_version"]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Please sign in again.")
    return user


def get_current_user_csrf(
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    token = request.cookies.get(SESSION_COOKIE, "")
    supplied = request.headers.get("X-CSRF-Token", "")
    if not supplied or not hmac.compare_digest(supplied, csrf_for_session(token)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF validation failed.")
    return user


def require_same_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if not origin:
        return
    allowed = {
        item.strip().rstrip("/")
        for item in os.getenv(
            "PNEUMONIA_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if item.strip()
    }
    configured_origin = os.getenv("APP_ORIGIN")
    if configured_origin:
        allowed.add(configured_origin.rstrip("/"))
    if origin.rstrip("/") not in allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid request origin.")
