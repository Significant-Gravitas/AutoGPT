import logging
import os

import pytest

from swap_proxy.__main__ import CA_FILE, DHPARAM_FILE, MissingCA, require_ca


def test_without_a_provisioned_ca_the_service_does_not_start(tmp_path):
    """mitmproxy would generate one, different on every replica and restart."""
    with pytest.raises(MissingCA, match="SWAP_PROXY_CONFDIR"):
        require_ca(str(tmp_path), generate=False)


def test_a_mounted_ca_is_accepted(tmp_path):
    (tmp_path / CA_FILE).write_text("-----BEGIN PRIVATE KEY-----")
    require_ca(str(tmp_path), generate=False)


def test_a_read_only_mount_needs_the_dhparam_file_beside_the_ca(tmp_path):
    """mitmproxy writes it when it is missing, and would crash on the write."""
    (tmp_path / CA_FILE).write_text("-----BEGIN PRIVATE KEY-----")
    tmp_path.chmod(0o555)
    if os.access(tmp_path, os.W_OK):  # root ignores the mode
        tmp_path.chmod(0o755)
        return
    try:
        with pytest.raises(MissingCA, match=DHPARAM_FILE):
            require_ca(str(tmp_path), generate=False)
        tmp_path.chmod(0o755)
        (tmp_path / DHPARAM_FILE).write_text("-----BEGIN DH PARAMETERS-----")
        tmp_path.chmod(0o555)
        require_ca(str(tmp_path), generate=False)
    finally:
        tmp_path.chmod(0o755)


def test_a_local_run_may_ask_for_a_generated_one(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="swap_proxy.__main__"):
        require_ca(str(tmp_path), generate=True)
    assert "Local use only" in caplog.text
