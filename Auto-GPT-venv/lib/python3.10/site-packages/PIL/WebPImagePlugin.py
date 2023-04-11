from io import BytesIO

from . import Image, ImageFile

try:
    from . import _webp

    SUPPORTED = True
except ImportError:
    SUPPORTED = False


_VALID_WEBP_MODES = {"RGBX": True, "RGBA": True, "RGB": True}

_VALID_WEBP_LEGACY_MODES = {"RGB": True, "RGBA": True}

_VP8_MODES_BY_IDENTIFIER = {
    b"VP8 ": "RGB",
    b"VP8X": "RGBA",
    b"VP8L": "RGBA",  # lossless
}


def _accept(prefix):
    is_riff_file_format = prefix[:4] == b"RIFF"
    is_webp_file = prefix[8:12] == b"WEBP"
    is_valid_vp8_mode = prefix[12:16] in _VP8_MODES_BY_IDENTIFIER

    if is_riff_file_format and is_webp_file and is_valid_vp8_mode:
        if not SUPPORTED:
            return (
                "image file could not be identified because WEBP support not installed"
            )
        return True


class WebPImageFile(ImageFile.ImageFile):
    format = "WEBP"
    format_description = "WebP image"
    __loaded = 0
    __logical_frame = 0

    def _open(self):
        if not _webp.HAVE_WEBPANIM:
            # Legacy mode
            data, width, height, self.mode, icc_profile, exif = _webp.WebPDecode(
                self.fp.read()
            )
            if icc_profile:
                self.info["icc_profile"] = icc_profile
            if exif:
                self.info["exif"] = exif
            self._size = width, height
            self.fp = BytesIO(data)
            self.tile = [("raw", (0, 0) + self.size, 0, self.mode)]
            self.n_frames = 1
            self.is_animated = False
            return

        # Use the newer AnimDecoder API to parse the (possibly) animated file,
        # and access muxed chunks like ICC/EXIF/XMP.
        self._decoder = _webp.WebPAnimDecoder(self.fp.read())

        # Get info from decoder
        width, height, loop_count, bgcolor, frame_count, mode = self._decoder.get_info()
        self._size = width, height
        self.info["loop"] = loop_count
        bg_a, bg_r, bg_g, bg_b = (
            (bgcolor >> 24) & 0xFF,
            (bgcolor >> 16) & 0xFF,
            (bgcolor >> 8) & 0xFF,
            bgcolor & 0xFF,
        )
        self.info["background"] = (bg_r, bg_g, bg_b, bg_a)
        self.n_frames = frame_count
        self.is_animated = self.n_frames > 1
        self.mode = "RGB" if mode == "RGBX" else mode
        self.rawmode = mode
        self.tile = []

        # Attempt to read ICC / EXIF / XMP chunks from file
        icc_profile = self._decoder.get_chunk("ICCP")
        exif = self._decoder.get_chunk("EXIF")
        xmp = self._decoder.get_chunk("XMP ")
        if icc_profile:
            self.info["icc_profile"] = icc_profile
        if exif:
            self.info["exif"] = exif
        if xmp:
            self.info["xmp"] = xmp

        # Initialize seek state
        self._reset(reset=False)

    def _getexif(self):
        if "exif" not in self.info:
            return None
        return self.getexif()._get_merged_dict()

    def getxmp(self):
        """
        Returns a dictionary containing the XMP tags.
        Requires defusedxml to be installed.

        :returns: XMP tags in a dictionary.
        """
        return self._getxmp(self.info["xmp"]) if "xmp" in self.info else {}

    def seek(self, frame):
        if not self._seek_check(frame):
            return

        # Set logical frame to requested position
        self.__logical_frame = frame

    def _reset(self, reset=True):
        if reset:
            self._decoder.reset()
        self.__physical_frame = 0
        self.__loaded = -1
        self.__timestamp = 0

    def _get_next(self):
        # Get next frame
        ret = self._decoder.get_next()
        self.__physical_frame += 1

        # Check if an error occurred
        if ret is None:
            self._reset()  # Reset just to be safe
            self.seek(0)
            msg = "failed to decode next frame in WebP file"
            raise EOFError(msg)

        # Compute duration
        data, timestamp = ret
        duration = timestamp - self.__timestamp
        self.__timestamp = timestamp

        # libwebp gives frame end, adjust to start of frame
        timestamp -= duration
        return data, timestamp, duration

    def _seek(self, frame):
        if self.__physical_frame == frame:
            return  # Nothing to do
        if frame < self.__physical_frame:
            self._reset()  # Rewind to beginning
        while self.__physical_frame < frame:
            self._get_next()  # Advance to the requested frame

    def load(self):
        if _webp.HAVE_WEBPANIM:
            if self.__loaded != self.__logical_frame:
                self._seek(self.__logical_frame)

                # We need to load the image data for this frame
                data, timestamp, duration = self._get_next()
                self.info["timestamp"] = timestamp
                self.info["duration"] = duration
                self.__loaded = self.__logical_frame

                # Set tile
                if self.fp and self._exclusive_fp:
                    self.fp.close()
                self.fp = BytesIO(data)
                self.tile = [("raw", (0, 0) + self.size, 0, self.rawmode)]

        return super().load()

    def tell(self):
        if not _webp.HAVE_WEBPANIM:
            return super().tell()

        return self.__logical_frame


def _save_all(im, fp, filename):
    encoderinfo = im.encoderinfo.copy()
    append_images = list(encoderinfo.get("append_images", []))

    # If total frame count is 1, then save using the legacy API, which
    # will preserve non-alpha modes
    total = 0
    for ims in [im] + append_images:
        total += getattr(ims, "n_frames", 1)
    if total == 1:
        _save(im, fp, filename)
        return

    background = (0, 0, 0, 0)
    if "background" in encoderinfo:
        background = encoderinfo["background"]
    elif "background" in im.info:
        background = im.info["background"]
        if isinstance(background, int):
            # GifImagePlugin stores a global color table index in
            # info["background"]. So it must be converted to an RGBA value
            palette = im.getpalette()
            if palette:
                r, g, b = palette[background * 3 : (background + 1) * 3]
                background = (r, g, b, 255)
            else:
                background = (background, background, background, 255)

    duration = im.encoderinfo.get("duration", im.info.get("duration", 0))
    loop = im.encoderinfo.get("loop", 0)
    minimize_size = im.encoderinfo.get("minimize_size", False)
    kmin = im.encoderinfo.get("kmin", None)
    kmax = im.encoderinfo.get("kmax", None)
    allow_mixed = im.encoderinfo.get("allow_mixed", False)
    verbose = False
    lossless = im.encoderinfo.get("lossless", False)
    quality = im.encoderinfo.get("quality", 80)
    method = im.encoderinfo.get("method", 0)
    icc_profile = im.encoderinfo.get("icc_profile") or ""
    exif = im.encoderinfo.get("exif", "")
    if isinstance(exif, Image.Exif):
        exif = exif.tobytes()
    xmp = im.encoderinfo.get("xmp", "")
    if allow_mixed:
        lossless = False

    # Sensible keyframe defaults are from gif2webp.c script
    if kmin is None:
        kmin = 9 if lossless else 3
    if kmax is None:
        kmax = 17 if lossless else 5

    # Validate background color
    if (
        not isinstance(background, (list, tuple))
        or len(background) != 4
        or not all(0 <= v < 256 for v in background)
    ):
        msg = f"Background color is not an RGBA tuple clamped to (0-255): {background}"
        raise OSError(msg)

    # Convert to packed uint
    bg_r, bg_g, bg_b, bg_a = background
    background = (bg_a << 24) | (bg_r << 16) | (bg_g << 8) | (bg_b << 0)

    # Setup the WebP animation encoder
    enc = _webp.WebPAnimEncoder(
        im.size[0],
        im.size[1],
        background,
        loop,
        minimize_size,
        kmin,
        kmax,
        allow_mixed,
        verbose,
    )

    # Add each frame
    frame_idx = 0
    timestamp = 0
    cur_idx = im.tell()
    try:
        for ims in [im] + append_images:
            # Get # of frames in this image
            nfr = getattr(ims, "n_frames", 1)

            for idx in range(nfr):
                ims.seek(idx)
                ims.load()

                # Make sure image mode is supported
                frame = ims
                rawmode = ims.mode
                if ims.mode not in _VALID_WEBP_MODES:
                    alpha = (
                        "A" in ims.mode
                        or "a" in ims.mode
                        or (ims.mode == "P" and "A" in ims.im.getpalettemode())
                    )
                    rawmode = "RGBA" if alpha else "RGB"
                    frame = ims.convert(rawmode)

                if rawmode == "RGB":
                    # For faster conversion, use RGBX
                    rawmode = "RGBX"

                # Append the frame to the animation encoder
                enc.add(
                    frame.tobytes("raw", rawmode),
                    round(timestamp),
                    frame.size[0],
                    frame.size[1],
                    rawmode,
                    lossless,
                    quality,
                    method,
                )

                # Update timestamp and frame index
                if isinstance(duration, (list, tuple)):
                    timestamp += duration[frame_idx]
                else:
                    timestamp += duration
                frame_idx += 1

    finally:
        im.seek(cur_idx)

    # Force encoder to flush frames
    enc.add(None, round(timestamp), 0, 0, "", lossless, quality, 0)

    # Get the final output from the encoder
    data = enc.assemble(icc_profile, exif, xmp)
    if data is None:
        msg = "cannot write file as WebP (encoder returned None)"
        raise OSError(msg)

    fp.write(data)


def _save(im, fp, filename):
    lossless = im.encoderinfo.get("lossless", False)
    quality = im.encoderinfo.get("quality", 80)
    icc_profile = im.encoderinfo.get("icc_profile") or ""
    exif = im.encoderinfo.get("exif", b"")
    if isinstance(exif, Image.Exif):
        exif = exif.tobytes()
    if exif.startswith(b"Exif\x00\x00"):
        exif = exif[6:]
    xmp = im.encoderinfo.get("xmp", "")
    method = im.encoderinfo.get("method", 4)
    exact = 1 if im.encoderinfo.get("exact") else 0

    if im.mode not in _VALID_WEBP_LEGACY_MODES:
        alpha = (
            "A" in im.mode
            or "a" in im.mode
            or (im.mode == "P" and "transparency" in im.info)
        )
        im = im.convert("RGBA" if alpha else "RGB")

    data = _webp.WebPEncode(
        im.tobytes(),
        im.size[0],
        im.size[1],
        lossless,
        float(quality),
        im.mode,
        icc_profile,
        method,
        exact,
        exif,
        xmp,
    )
    if data is None:
        msg = "cannot write file as WebP (encoder returned None)"
        raise OSError(msg)

    fp.write(data)


Image.register_open(WebPImageFile.format, WebPImageFile, _accept)
if SUPPORTED:
    Image.register_save(WebPImageFile.format, _save)
    if _webp.HAVE_WEBPANIM:
        Image.register_save_all(WebPImageFile.format, _save_all)
    Image.register_extension(WebPImageFile.format, ".webp")
    Image.register_mime(WebPImageFile.format, "image/webp")
