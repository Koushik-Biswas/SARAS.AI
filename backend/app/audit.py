from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from .models import AuditLog


def audit(db: Session, *, user_id: str | None, entity: str, entity_id: str, action: str,
          old: dict[str, Any] | None = None, new: dict[str, Any] | None = None, reason: str | None = None) -> None:
    """Append one audit entry in the caller's transaction, so it commits together
    with the change it describes."""
    db.add(AuditLog(user_id=user_id, entity=entity, entity_id=entity_id, action=action,
                    old_value=old, new_value=new, reason=reason))
