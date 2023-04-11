#
# The Python Imaging Library.
#
# QOI support for PIL
#
# See the README file for information on usage and redistribution.
#

import os

from . import Image, ImageFile
from ._binary import i32be as i32
from ._binary import o8


def _accept(prefix):
    return prefix[:4] == b"qoif"


class QoiImageFile(ImageFile.ImageFile):
    format = "QOI"
    format_description = "Quite OK Image"

    def _open(self):
        if not _accept(self.fp.read(4)):
            msg = "not a QOI file"
            raise SyntaxError(msg)

        self._size = tuple(i32(self.fp.read(4)) for i in range(2))

        channels = self.fp.read(1)[0]
        self.mode = "RGB" if channels == 3 else "RGBA"

        self.fp.seek(1, os.SEEK_CUR)  # colorspace
        self.tile = [("qoi", (0, 0) + self._size, self.fp.tell(), None)]


class QoiDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def _add_to_previous_pixels(self, value):
        self._previous_pixel = value

        r, g, b, a = value
        hash_value = (r * 3 + g * 5 + b * 7 + a * 11) % 64
        self._previously_seen_pixels[hash_value] = value

    def decode(self, buffer):
        self._previously_seen_pixels = {}
        self._previous_pixel = None
        self._add_to_previous_pixels(b"".join(o8(i) for i in (0, 0, 0, 255)))

        data = bytearray()
        bands = Image.getmodebands(self.mode)
        while len(data) < self.state.xsize * self.state.ysize * bands:
            byte = self.fd.read(1)[0]
            if byte == 0b11111110:  # QOI_OP_RGB
                value = self.fd.read(3) + o8(255)
            elif byte == 0b11111111:  # QOI_OP_RGBA
                value = self.fd.read(4)
            else:
                op = byte >> 6
                if op == 0:  # QOI_OP_INDEX
                    op_index = byte & 0b00111111
                    value = self._previously_seen_pixels.get(op_index, (0, 0, 0, 0))
                elif op == 1:  # QOI_OP_DIFF
                    value = (
                        (self._previous_pixel[0] + ((byte & 0b00110000) >> 4) - 2)
                        % 256,
                        (self._previous_pixel[1] + ((byte & 0b00001100) >> 2) - 2)
                        % 256,
                        (self._previous_pixel[2] + (byte & 0b00000011) - 2) % 256,
                    )
                    value += (self._previous_pixel[3],)
                elif op == 2:  # QOI_OP_LUMA
                    second_byte = self.fd.read(1)[0]
                    diff_green = (byte & 0b00111111) - 32
                    diff_red = ((second_byte & 0b11110000) >> 4) - 8
                    diff_blue = (second_byte & 0b00001111) - 8

                    value = tuple(
                        (self._previous_pixel[i] + diff_green + diff) % 256
                        for i, diff in enumerate((diff_red, 0, diff_blue))
                    )
                    value += (self._previous_pixel[3],)
                elif op == 3:  # QOI_OP_RUN
                    run_length = (byte & 0b00111111) + 1
                    value = self._previous_pixel
                    if bands == 3:
                        value = value[:3]
                    data += value * run_length
                    continue
                value = b"".join(o8(i) for i in value)
            self._add_to_previous_pixels(value)

            if bands == 3:
                value = value[:3]
            data += value
        self.set_as_raw(bytes(data))
        return -1, 0


Image.register_open(QoiImageFile.format, QoiImageFile, _accept)
Image.register_decoder("qoi", QoiDecoder)
Image.register_extension(QoiImageFile.format, ".qoi")
