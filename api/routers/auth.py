"""
AEROTWIN — Authentication & User Management Router
Provides local sign-in, login, registration, and session verification.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, EmailStr

from storage.storage import get_sqlite

logger = logging.getLogger("aerotwin.api.auth")

router = APIRouter()

SALT = "aerotwin_salt_2026"
ACTIVE_SESSIONS: Dict[str, Dict[str, Any]] = {}


def hash_password(password: str) -> str:
    """Hash password with application salt using SHA-256."""
    return hashlib.sha256((SALT + password).encode("utf-8")).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """Constant-time password hash comparison."""
    return hmac.compare_digest(hash_password(password), hashed)


class LoginRequest(BaseModel):
    username_or_email: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = "Flight Ops Engineer"
    role: Optional[str] = "Flight Engineer"


@router.post("/auth/login", summary="User Login")
async def login(req: LoginRequest) -> Dict[str, Any]:
    db = get_sqlite()
    login_id = req.username_or_email.strip()

    # Find user by email or username
    user = db.get_user_by_email(login_id)
    if not user:
        user = db.get_user_by_username(login_id)

    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email/username or password")

    # Generate session token
    token = secrets.token_hex(24)
    user_info = {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "full_name": user["full_name"],
        "role": user["role"],
    }
    ACTIVE_SESSIONS[token] = {
        "user": user_info,
        "created_at": time.time(),
    }

    logger.info("User %s (%s) logged in successfully", user["username"], user["email"])
    return {
        "status": "success",
        "token": token,
        "user": user_info,
        "message": f"Welcome back, {user['full_name']}!",
    }


@router.post("/auth/register", summary="User Registration")
async def register(req: RegisterRequest) -> Dict[str, Any]:
    db = get_sqlite()
    username = req.username.strip()
    email = req.email.strip().lower()

    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    if db.get_user_by_username(username):
        raise HTTPException(status_code=400, detail="Username is already taken")

    if db.get_user_by_email(email):
        raise HTTPException(status_code=400, detail="Email is already registered")

    pwd_hash = hash_password(req.password)
    user_id = db.create_user(
        username=username,
        email=email,
        password_hash=pwd_hash,
        full_name=req.full_name or "Flight Ops Engineer",
        role=req.role or "Flight Engineer",
    )

    token = secrets.token_hex(24)
    user_info = {
        "id": user_id,
        "username": username,
        "email": email,
        "full_name": req.full_name,
        "role": req.role,
    }
    ACTIVE_SESSIONS[token] = {
        "user": user_info,
        "created_at": time.time(),
    }

    logger.info("New user registered: %s (%s)", username, email)
    return {
        "status": "success",
        "token": token,
        "user": user_info,
        "message": "Account created successfully!",
    }


@router.get("/auth/me", summary="Get Current Logged-in User")
async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    if not authorization:
        # Default fallback for demo / guest ops
        return {
            "authenticated": False,
            "user": {
                "id": 1,
                "username": "admin",
                "email": "admin@aerotwin.ai",
                "full_name": "Capt. Alex Chen",
                "role": "Mission Commander",
            }
        }

    token = authorization.replace("Bearer ", "").strip()
    session = ACTIVE_SESSIONS.get(token)
    if not session:
        return {
            "authenticated": False,
            "user": {
                "id": 1,
                "username": "admin",
                "email": "admin@aerotwin.ai",
                "full_name": "Capt. Alex Chen",
                "role": "Mission Commander",
            }
        }

    return {
        "authenticated": True,
        "user": session["user"],
    }


@router.post("/auth/logout", summary="User Logout")
async def logout(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    if authorization:
        token = authorization.replace("Bearer ", "").strip()
        ACTIVE_SESSIONS.pop(token, None)
    return {"status": "logged_out"}
