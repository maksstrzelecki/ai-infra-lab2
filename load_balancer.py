
from contextlib import asynccontextmanager
from itertools import cycle

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response


WORKERS = [
    "http://localhost:8001",
    "http://localhost:8002",
]

worker_iterator = cycle(WORKERS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=60.0)
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, path: str):

    # Select worker using Round Robin
    worker = next(worker_iterator)

    body = await request.body()
    url = f"{worker}/{path}"

    try:
        response = await request.app.state.client.request(
            method=request.method,
            url=url,
            content=body,
            headers={
                "content-type": request.headers.get("content-type", "")
            }
        )

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Worker unavailable: {worker}"
        ) from e

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers={
            **{
                k: v for k, v in response.headers.items()
                if k.lower() not in {
                    "content-length",
                    "transfer-encoding",
                    "connection",
                    "content-encoding",
                    "x-worker",
                }
            },
            "X-Worker": worker,
        }
    )
