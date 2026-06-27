"""
Authentication endpoints:
  POST /api/v1/auth/register  → create account
  POST /api/v1/auth/login     → get JWT token
  GET  /api/v1/auth/me        → current user info
  POST /api/v1/auth/logout    → (client-side token drop, endpoint for audit log)
"""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, field_validator

from app.db.database import get_db
from app.auth.models import User
from app.auth.security import hash_password, verify_password, create_access_token
from app.auth.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


# ── Schemas ──────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    username: str
    password: str
    full_name: str = ""

    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

    @field_validator("username")
    @classmethod
    def username_format(cls, v):
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters")
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username can only contain letters, numbers, - and _")
        return v.lower()


class LoginRequest(BaseModel):
    username: str   # username or email
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str
    full_name: str
    role: str


class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    full_name: str
    role: str
    created_at: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Create a new account and return a JWT token."""
    # Check uniqueness
    if db.query(User).filter(User.email == request.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    if db.query(User).filter(User.username == request.username).first():
        raise HTTPException(status_code=409, detail="Username already taken")

    user = User(
        email=request.email,
        username=request.username,
        hashed_password=hash_password(request.password),
        full_name=request.full_name,
        role="engineer",  # first user becomes admin
    )

    # First-ever user gets admin role
    if db.query(User).count() == 0:
        user.role = "admin"

    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": str(user.id), "role": user.role})
    return TokenResponse(
        access_token=token,
        user_id=str(user.id),
        username=user.username,
        full_name=user.full_name or user.username,
        role=user.role,
    )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate and return a JWT token."""
    # Accept either username or email
    user = (
        db.query(User).filter(User.username == request.username.lower()).first()
        or db.query(User).filter(User.email == request.username).first()
    )

    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")

    user.last_login = datetime.utcnow()
    db.commit()

    token = create_access_token({"sub": str(user.id), "role": user.role})
    return TokenResponse(
        access_token=token,
        user_id=str(user.id),
        username=user.username,
        full_name=user.full_name or user.username,
        role=user.role,
    )


@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    return UserResponse(
        user_id=str(current_user.id),
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name or current_user.username,
        role=current_user.role,
        created_at=current_user.created_at.isoformat(),
    )


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """Client should drop the token. Server-side audit log only."""
    return {"status": "logged out", "username": current_user.username}
