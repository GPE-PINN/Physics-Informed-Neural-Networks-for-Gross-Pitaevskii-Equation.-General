# app/forward/router.py
from pathlib import Path
import json
import time
import traceback
from typing import Dict, Any

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from sqlalchemy.orm import Session
from sqlalchemy import select


from .model import GrossPitaevskiiInference
from .schemas import ForwardRequest
from app.db import RequestHistory, get_db

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[2]

model = GrossPitaevskiiInference(
    checkpoint_path=BASE_DIR / "models" / "checkpoint_iter_9500.pt"
)


def save_request_to_db(
    endpoint: str, 
    method: str, 
    body_dict: dict, 
    processing_time: float, 
    input_size: int,
    db: Session
):
    record = RequestHistory(
        endpoint=endpoint,
        method=method,
        body=json.dumps(body_dict, ensure_ascii=False),
        processing_time=processing_time,
        input_size=input_size,
    )
    db.add(record)
    db.commit()


@router.post(
    "/forward",
    responses={
        400: {
            "description": "Bad Request",
            "content": {"text/plain": {"example": "bad request"}},
        },
        403: {
            "description": "Model failed",
            "content": {
                "application/json": {
                    "example": {"detail": "модель не смогла обработать данные"}
                }
            },
        },
    },
)
async def forward(data: ForwardRequest, request: Request, db: Session = Depends(get_db)):
    start_time = time.time()
    if not data.x or not data.y or not data.t:
        raise RequestValidationError(["Empty x, y or t"])

    if not (len(data.x) == len(data.y) == len(data.t)):
        raise RequestValidationError(["x, y, t must have same length"])

    try:
        x = torch.tensor(data.x).unsqueeze(1)
        y = torch.tensor(data.y).unsqueeze(1)
        t = torch.tensor(data.t).unsqueeze(1)
        print(f"Tensor shapes: x={x.shape}, y={y.shape}, t={t.shape}")  # DEBUG
        n = x.shape[0]
        input_size = len(data.x)
        ox = torch.full((n, 1), data.omega_x)
        oy = torch.full((n, 1), data.omega_y)
        g  = torch.full((n, 1), data.g_param)

        result = model.predict(x, y, t, ox, oy, g)
          
        processing_time = time.time() - start_time
    
    
        save_request_to_db(
                endpoint=str(request.url.path),
                method=request.method,
                body_dict=data.model_dump(),
                processing_time=processing_time,
                input_size=input_size,
                db=db)
        return {"trajectory": result.tolist()}

    except Exception as e:
        print(f"ERROR in forward: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=403,
            detail="модель не смогла обработать данные"
        )

@router.get("/history")
def get_history(db: Session = Depends(get_db)):
    records = db.query(RequestHistory).order_by(RequestHistory.id.desc()).all()
    return [
        {
            "id": r.id,
            "endpoint": r.endpoint,
            "method": r.method,
            "body": r.body,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records
    ]




@router.get("/stats")
def get_stats(db: Session = Depends(get_db)) -> Dict[str, Any]:
    records = db.scalars(select(RequestHistory)).all()
    
    if not records:
        return {"message": "No requests yet"}
    
    times = np.array([r.processing_time for r in records])
    sizes = np.array([r.input_size for r in records])
    
    return {
        "total_requests": len(records),
        "processing_time": {
            "mean": float(np.mean(times)),
            "p50": float(np.percentile(times, 50)),
            "p95": float(np.percentile(times, 95)),
            "p99": float(np.percentile(times, 99)),
            "min": float(np.min(times)),
            "max": float(np.max(times))
        },
        "input_sizes": {
            "mean": float(np.mean(sizes)),
            "min": int(np.min(sizes)),
            "max": int(np.max(sizes)),
            "p95": int(np.percentile(sizes, 95))
        }
    }



