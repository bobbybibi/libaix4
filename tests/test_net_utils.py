"""Tests for net_utils — safe, non-destructive port probing.

These verify that libaix detects busy ports and steps aside without ever
touching the process already using a port.
"""

from __future__ import annotations

import socket
from contextlib import closing

import pytest

import net_utils


def _bind_holder(host: str = "127.0.0.1") -> tuple[socket.socket, int]:
    """Bind a socket to an ephemeral port and keep holding it.

    Returns the live socket (caller must close it) and the port it holds.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    return sock, port


def test_free_port_is_available():
    # An ephemeral port we bind then release should be reported available.
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert net_utils.is_port_available("127.0.0.1", port) is True


def test_busy_port_is_unavailable_and_holder_survives():
    holder, port = _bind_holder()
    try:
        # The port is occupied → reported unavailable...
        assert net_utils.is_port_available("127.0.0.1", port) is False
        # ...and the probe must NOT have killed/closed the holder: it is still
        # bound and usable. fileno() != -1 means the socket is still open.
        assert holder.fileno() != -1
        # Probing again still reports busy — proof we didn't steal the port.
        assert net_utils.is_port_available("127.0.0.1", port) is False
    finally:
        holder.close()


def test_out_of_range_ports_are_unavailable():
    assert net_utils.is_port_available("127.0.0.1", 0) is False
    assert net_utils.is_port_available("127.0.0.1", 70000) is False
    assert net_utils.is_port_available("127.0.0.1", -1) is False


def test_find_available_port_steps_past_busy_port():
    holder, port = _bind_holder()
    try:
        found = net_utils.find_available_port("127.0.0.1", port)
        assert found is not None
        # Must not hand back the occupied port.
        assert found != port
        assert found > port
        # And the returned port must actually be free.
        assert net_utils.is_port_available("127.0.0.1", found) is True
    finally:
        holder.close()


def test_find_available_port_returns_requested_when_free():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert net_utils.find_available_port("127.0.0.1", port) == port


def test_find_available_port_none_when_exhausted():
    holder, port = _bind_holder()
    try:
        # Only probe the single busy port → nothing free in range.
        assert net_utils.find_available_port("127.0.0.1", port, max_tries=1) is None
    finally:
        holder.close()


def test_resolve_port_returns_free_port():
    import start

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert start._resolve_port("127.0.0.1", port) == port


def test_resolve_port_strict_exits_when_busy():
    import start

    holder, port = _bind_holder()
    try:
        with pytest.raises(SystemExit):
            start._resolve_port("127.0.0.1", port, strict=True)
    finally:
        holder.close()


def test_resolve_port_steps_aside_when_busy():
    import start

    holder, port = _bind_holder()
    try:
        resolved = start._resolve_port("127.0.0.1", port, strict=False)
        assert resolved != port
        assert net_utils.is_port_available("127.0.0.1", resolved) is True
    finally:
        holder.close()
