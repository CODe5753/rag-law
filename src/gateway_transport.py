"""One isolated HTTP request; the parent owns the wall-clock deadline."""
import json
import math
from pathlib import Path
import re
import subprocess
import sys

MAX_RESPONSE_BYTES = 1024 * 1024


def validate_timeout(value):
    try:
        timeout = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("AI gateway timeout must be between 1 and 300 seconds") from None
    if isinstance(value, bool) or not math.isfinite(timeout) or not 1 <= timeout <= 300:
        raise ValueError("AI gateway timeout must be between 1 and 300 seconds")
    return timeout


def post_json(url, *, token, payload, timeout):
    timeout = validate_timeout(timeout)
    # Credentials and prompts go only through stdin, never argv or diagnostics.
    request = json.dumps(dict(url=url, token=token, payload=payload, timeout=timeout))
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            input=request.encode("utf-8"), stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        # run() kills and waits for the child before raising, including on Python 3.10.
        raise TimeoutError("AI gateway transport deadline exceeded") from None
    except OSError:
        raise RuntimeError("AI gateway transport failed") from None
    if result.returncode != 0:
        raise RuntimeError("AI gateway transport failed")
    try:
        response = json.loads(result.stdout)
    except (ValueError, UnicodeError):
        raise RuntimeError("AI gateway transport failed") from None
    if response.get("error") == "invalid_response":
        raise ValueError("AI gateway returned invalid response")
    if response.get("error"):
        raise RuntimeError("AI gateway transport failed")
    return response


def _request(request):
    import requests

    with requests.post(
        request["url"], headers={"Authorization": f"Bearer {request['token']}"},
        json=request["payload"], timeout=request["timeout"], stream=True,
    ) as response:
        status = response.status_code
        if not response.ok:
            return {"status_code": status, "data": None}
        body = bytearray()
        for chunk in response.iter_content(chunk_size=8192):
            body.extend(chunk)
            if len(body) > MAX_RESPONSE_BYTES:
                return {"error": "invalid_response"}
        try:
            data = json.loads(body)
        except (ValueError, UnicodeError):
            return {"error": "invalid_response"}
        # Do not send upstream metadata/error bodies back to the parent.
        safe = {}
        if isinstance(data, dict):
            request_id = data.get("id")
            if isinstance(request_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
                safe["id"] = request_id
            if isinstance(data.get("text"), str):
                safe["text"] = data["text"]
        return {"status_code": status, "data": safe}


def _main():
    try:
        result = _request(json.load(sys.stdin))
    except Exception:
        result = {"error": "transport"}
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    _main()
