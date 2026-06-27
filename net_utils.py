"""net_utils.py — Safe, non-destructive network port helpers.

These helpers NEVER kill, signal, or otherwise touch other processes. They
only *probe* whether a port is available by attempting a short-lived bind and
immediately releasing it. If another application already owns a port, the bind
simply fails and we report it as unavailable — the other app keeps running,
untouched. When a requested port is busy, callers can step aside onto the next
free port instead of stomping on whatever is already there.
"""

from __future__ import annotations

import socket
from contextlib import closing

# How many consecutive ports to probe when searching for a free one.
DEFAULT_MAX_TRIES = 100

# Highest valid TCP port number.
MAX_PORT = 65535


def _family_for_host(host: str) -> int:
    """Pick the socket address family that matches *host*.

    A colon in the host string indicates an IPv6 literal (e.g. ``::1`` or
    ``::``); everything else (``0.0.0.0``, ``127.0.0.1``, ``localhost``) is
    treated as IPv4.
    """
    return socket.AF_INET6 if ":" in host else socket.AF_INET


def is_port_available(host: str, port: int) -> bool:
    """Return ``True`` if *port* can be bound on *host* right now.

    This works by creating a fresh socket, attempting to ``bind()`` it, and
    immediately closing it. Crucially we do **not** set ``SO_REUSEADDR`` /
    ``SO_REUSEPORT``: we *want* the bind to fail when another listener already
    owns the port, so we can detect the conflict and leave that process alone
    rather than fighting it for the port.

    The probe is read-only with respect to other processes — it never kills or
    signals whatever may be using the port.
    """
    if not (0 < port <= MAX_PORT):
        return False

    family = _family_for_host(host)
    try:
        with closing(socket.socket(family, socket.SOCK_STREAM)) as sock:
            sock.bind((host, port))
            return True
    except OSError:
        # Port is in use (EADDRINUSE), needs privileges, or host is invalid —
        # in every case it is not safely bindable, so report unavailable.
        return False


def find_available_port(
    host: str,
    port: int,
    max_tries: int = DEFAULT_MAX_TRIES,
) -> int | None:
    """Return the first available port at or above *port*.

    Scans up to *max_tries* consecutive ports starting at *port*. Returns the
    first one that is free, or ``None`` if none of them are. This only ever
    probes for a free port to *step aside* onto — it never terminates anything.
    """
    if max_tries < 1:
        max_tries = 1
    last = min(port + max_tries, MAX_PORT + 1)
    for candidate in range(max(port, 1), last):
        if is_port_available(host, candidate):
            return candidate
    return None
