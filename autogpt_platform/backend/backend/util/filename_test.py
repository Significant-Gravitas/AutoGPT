import pytest

from backend.util.filename import MAX_FILENAME_BYTES, sanitize_filename


class TestSanitizeFilenameByteLimit:
    """Filesystems cap a name component at 255 bytes, not characters (#15651)."""

    @staticmethod
    def _assert_fits(result: str) -> None:
        encoded = result.encode("utf-8")
        assert len(encoded) <= MAX_FILENAME_BYTES < 255
        # Round-trips, i.e. no code point was split at the cut.
        assert encoded.decode("utf-8") == result

    @pytest.mark.parametrize(
        "filename",
        [
            "夜" * 84 + ".txt",  # 88 chars / 256 bytes: under the old char cap
            "夜" * 300 + ".txt",
            "🌙" * 64 + ".png",  # 4-byte code points
            "Nachtschicht " + "🌙" * 190 + ".png",
            "a" + "夜" * 100 + ".txt",  # cut lands mid-character without care
            "ab" + "🌙" * 80 + ".md",
        ],
        ids=[
            "cjk-256-bytes",
            "cjk-long",
            "emoji",
            "mixed-emoji",
            "cjk-odd-boundary",
            "emoji-odd-boundary",
        ],
    )
    def test_multibyte_names_fit_and_keep_extension(self, filename: str):
        result = sanitize_filename(filename)

        self._assert_fits(result)
        assert result.endswith(filename.rsplit(".", 1)[1])
        assert result.rsplit(".", 1)[1] == filename.rsplit(".", 1)[1]
        assert filename.startswith(result.rsplit(".", 1)[0])

    def test_long_ascii_name_keeps_extension(self):
        result = sanitize_filename("a" * 300 + ".pdf")

        self._assert_fits(result)
        assert result.endswith(".pdf")
        assert len(result) == 200  # the character cap still applies

    def test_long_multibyte_name_without_extension(self):
        result = sanitize_filename("夜" * 300)

        self._assert_fits(result)
        assert result == "夜" * 80

    def test_long_text_after_dot_is_not_treated_as_extension(self):
        result = sanitize_filename("report." + "夜" * 300)

        self._assert_fits(result)
        assert result.startswith("report.")

    @pytest.mark.parametrize(
        "filename",
        ["夜" * 20 + ".txt", "🌙 moon.png", "a.b"],
        ids=["cjk", "emoji", "ascii"],
    )
    def test_short_names_are_unchanged(self, filename: str):
        assert sanitize_filename(filename) == filename
