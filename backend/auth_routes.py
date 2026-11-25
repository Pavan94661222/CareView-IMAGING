"""Local account registration, login, and password recovery endpoints."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
import smtplib
import sqlite3
import ssl
import uuid
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from . import clinical_store
from .auth_security import (
    clear_session,
    get_current_user,
    get_current_user_csrf,
    hash_password,
    public_user,
    require_same_origin,
    set_session,
    verify_password,
)

logger = logging.getLogger("pneumonia.auth")
router = APIRouter(prefix="/api/auth", tags=["authentication"])
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class RegisterPayload(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(min_length=12, max_length=256)
    display_name: str = Field(min_length=1, max_length=100)


class LoginPayload(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(min_length=1, max_length=256)


class ProfilePayload(BaseModel):
    display_name: str = Field(min_length=1, max_length=100)


class PasswordPayload(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=12, max_length=256)


class PasswordResetRequest(BaseModel):
    email: str = Field(max_length=254)


class PasswordResetPayload(BaseModel):
    token: str = Field(min_length=20, max_length=200)
    new_password: str = Field(min_length=12, max_length=256)


def _email(value: str) -> str:
    normalized = value.strip().lower()
    if not EMAIL_PATTERN.fullmatch(normalized):
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    return normalized


def _display_name(value: str) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        raise HTTPException(status_code=422, detail="Name cannot be empty.")
    return normalized


def _smtp_settings() -> Dict[str, Any]:
    host = os.getenv("SMTP_HOST", "").strip()
    sender = os.getenv("SMTP_FROM", "").strip()
    if not host or not sender:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Password reset email is not configured. Configure SMTP_HOST and SMTP_FROM.",
        )
    try:
        port = int(os.getenv("SMTP_PORT", "587"))
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="SMTP_PORT must be a valid port number.") from exc
    return {
        "host": host,
        "port": port,
        "username": os.getenv("SMTP_USERNAME", ""),
        "password": os.getenv("SMTP_PASSWORD", ""),
        "sender": sender,
        "use_tls": os.getenv("SMTP_USE_TLS", "true").strip().lower() in {"1", "true", "yes"},
    }


def _send_reset_email(email: str, token: str, settings: Dict[str, Any]) -> None:
    origin = os.getenv("APP_ORIGIN", "http://127.0.0.1:5173").rstrip("/")
    message = EmailMessage()
    message["Subject"] = "Reset your clinical app password"
    message["From"] = settings["sender"]
    message["To"] = email
    message.set_content(
        "A password reset was requested for your account.\n\n"
        f"Use this one-time code in the app to reset your password: {token}\n"
        f"The code expires in one hour. Open {origin} to continue.\n\n"
        "If you did not request this, you can ignore this email."
    )
    try:
        if settings["port"] == 465:
            with smtplib.SMTP_SSL(
                settings["host"],
                settings["port"],
                timeout=15,
                context=ssl.create_default_context(),
            ) as smtp:
                if settings["username"]:
                    smtp.login(settings["username"], settings["password"])
                smtp.send_message(message)
            return
        with smtplib.SMTP(settings["host"], settings["port"], timeout=15) as smtp:
            smtp.ehlo()
            if settings["use_tls"]:
                smtp.starttls(context=ssl.create_default_context())
                smtp.ehlo()
            if settings["username"]:
                smtp.login(settings["username"], settings["password"])
            smtp.send_message(message)
    except (OSError, smtplib.SMTPException) as exc:
        logger.exception("Password reset email delivery failed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Password reset email could not be sent. Check the email service configuration.",
        ) from exc


@router.post("/register", status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_same_origin)])
def register(payload: RegisterPayload, response: Response) -> Dict[str, Any]:
    email = _email(payload.email)
    display_name = _display_name(payload.display_name)
    try:
        user = clinical_store.create_user(
            uuid.uuid4().hex,
            email,
            display_name,
            hash_password(payload.password),
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="An account with this email already exists.") from exc
    return {"user": public_user(user), "csrf_token": set_session(response, user)}


@router.post("/login", dependencies=[Depends(require_same_origin)])
def login(payload: LoginPayload, response: Response) -> Dict[str, Any]:
    user = clinical_store.user_by_email(_email(payload.email))
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Email or password is incorrect.")
    return {"user": public_user(user), "csrf_token": set_session(response, user)}


@router.get("/me")
def me(request: Request, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    from .auth_security import csrf_for_session, SESSION_COOKIE

    token = request.cookies.get(SESSION_COOKIE, "")
    return {"user": public_user(user), "csrf_token": csrf_for_session(token)}


@router.post("/logout", dependencies=[Depends(require_same_origin)])
def logout(
    response: Response,
    _user: Dict[str, Any] = Depends(get_current_user_csrf),
) -> Dict[str, bool]:
    clear_session(response)
    return {"ok": True}


@router.patch("/profile", dependencies=[Depends(require_same_origin)])
def update_profile(
    payload: ProfilePayload,
    user: Dict[str, Any] = Depends(get_current_user_csrf),
) -> Dict[str, Any]:
    clinical_store.update_profile(user["id"], _display_name(payload.display_name))
    updated = clinical_store.user_by_id(user["id"])
    if not updated:
        raise HTTPException(status_code=404, detail="Account not found.")
    return {"user": public_user(updated)}


@router.post("/password", dependencies=[Depends(require_same_origin)])
def change_password(
    payload: PasswordPayload,
    response: Response,
    user: Dict[str, Any] = Depends(get_current_user_csrf),
) -> Dict[str, Any]:
    if not verify_password(payload.current_password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect.")
    password_hash = hash_password(payload.new_password)
    clinical_store.update_password(user["id"], password_hash)
    updated = clinical_store.user_by_id(user["id"])
    if not updated:
        raise HTTPException(status_code=404, detail="Account not found.")
    csrf_token = set_session(response, updated)
    return {"user": public_user(updated), "csrf_token": csrf_token}


@router.post("/password-reset/request", dependencies=[Depends(require_same_origin)])
def request_password_reset(payload: PasswordResetRequest) -> Dict[str, str]:
    settings = _smtp_settings()
    user = clinical_store.user_by_email(_email(payload.email))
    if user:
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(timespec="seconds")
        clinical_store.create_password_reset(user["id"], token_hash, expires_at)
        _send_reset_email(user["email"], token, settings)
    return {"message": "If an account exists for that email, a reset code has been sent."}


@router.post("/password-reset/confirm", dependencies=[Depends(require_same_origin)])
def reset_password(payload: PasswordResetPayload) -> Dict[str, bool]:
    token_hash = hashlib.sha256(payload.token.encode("utf-8")).hexdigest()
    if not clinical_store.consume_password_reset(token_hash, hash_password(payload.new_password)):
        raise HTTPException(status_code=400, detail="Reset code is invalid or expired.")
    return {"ok": True}
