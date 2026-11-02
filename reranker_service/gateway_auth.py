"""Gateway-only backend authentication. No credential contents in errors or logs."""
import hmac
import os
import stat
from pathlib import Path


def load_gateway_token(default_path):
    path = os.environ.get('MODEL_GATEWAY_TOKEN_FILE', default_path)
    descriptor = None
    try:
        if not path or not Path(path).is_absolute():
            raise ValueError()
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError()
        raw = os.read(descriptor, 4097)
        token = raw.decode('ascii').strip()
        if len(raw) > 4096 or not 32 <= len(token) <= 4096 or any(not 33 <= ord(c) <= 126 for c in token):
            raise ValueError()
        return token.encode('ascii')
    except (OSError, ValueError, UnicodeError):
        raise RuntimeError('Gateway backend token configuration invalid') from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def authorized(headers, expected):
    if expected is None:
        return False
    # Reject duplicate Authorization headers; no proxy-dependent ambiguous parsing.
    values = headers.get_all('Authorization', []) if hasattr(headers, 'get_all') else headers.getlist('authorization')
    if len(values) != 1 or not values[0].startswith('Bearer '):
        return False
    try:
        supplied = values[0][7:].encode('ascii')
    except UnicodeError:
        return False
    if len(supplied) > 4096:
        return False
    return hmac.compare_digest(supplied, expected)
