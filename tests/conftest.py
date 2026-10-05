"""Keep offline tests independent of the operator's running browser daemon."""

import pytest


@pytest.fixture(autouse=True)
def isolated_browser_socket(monkeypatch, tmp_path):
    # Integration tests opt into their own short, temporary Unix socket instead.
    monkeypatch.setenv("GADDI_SOCKET", str(tmp_path / "absent.sock"))
    monkeypatch.delenv("OMNIREAD_RENDERER", raising=False)
