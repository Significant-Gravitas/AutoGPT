#
# The Python Imaging Library.
# $Id$
#
# TIFF tags
#
# This module provides clear-text names for various well-known
# TIFF tags.  the TIFF codec works just fine without it.
#
# Copyright (c) Secret Labs AB 1999.
#
# See the README file for information on usage and redistribution.
#

##
# This module provides constants and clear-text names for various
# well-known TIFF tags.
##

from collections import namedtuple


class TagInfo(namedtuple("_TagInfo", "value name type length enum")):
    __slots__ = []

    def __new__(cls, value=None, name="unknown", type=None, length=None, enum=None):
        return super().__new__(cls, value, name, type, length, enum or {})

    def cvt_enum(self, value):
        # Using get will call hash(value), which can be expensive
        # for some types (e.g. Fraction). Since self.enum is rarely
        # used, it's usually better to test it first.
        return self.enum.get(value, value) if self.enum else value


def lookup(tag, group=None):
    """
    :param tag: Integer tag number
    :param group: Which :py:data:`~PIL.TiffTags.TAGS_V2_GROUPS` to look in

    .. versionadded:: 8.3.0

    :returns: Taginfo namedtuple, From the ``TAGS_V2`` info if possible,
        otherwise just populating the value and name from ``TAGS``.
        If the tag is not recognized, "unknown" is returned for the name

    """

    if group is not None:
        info = TAGS_V2_GROUPS[group].get(tag) if group in TAGS_V2_GROUPS else None
    else:
        info = TAGS_V2.get(tag)
    return info or TagInfo(tag, TAGS.get(tag, "unknown"))


##
# Map tag numbers to tag info.
#
#  id: (Name, Type, Length, enum_values)
#
# The length here differs from the length in the tiff spec.  For
# numbers, the tiff spec is for the number of fields returned. We
# agree here.  For string-like types, the tiff spec uses the length of
# field in bytes.  In Pillow, we are using the number of expected
# fields, in general 1 for string-like types.


BYTE = 1
ASCII = 2
SHORT = 3
LONG = 4
RATIONAL = 5
SIGNED_BYTE = 6
UNDEFINED = 7
SIGNED_SHORT = 8
SIGNED_LONG = 9
SIGNED_RATIONAL = 10
FLOAT = 11
DOUBLE = 12
IFD = 13
LONG8 = 16

TAGS_V2 = {
    254: ("NewSubfileType", LONG, 1),
    255: ("SubfileType", SHORT, 1),
    256: ("ImageWidth", LONG, 1),
    257: ("ImageLength", LONG, 1),
    258: ("BitsPerSample", SHORT, 0),
    259: (
        "Compression",
        SHORT,
        1,
        {
            "Uncompressed": 1,
            "CCITT 1d": 2,
            "Group 3 Fax": 3,
            "Group 4 Fax": 4,
            "LZW": 5,
            "JPEG": 6,
            "PackBits": 32773,
        },
    ),
    262: (
        "PhotometricInterpretation",
        SHORT,
        1,
        {
            "WhiteIsZero": 0,
            "BlackIsZero": 1,
            "RGB": 2,
            "RGB Palette": 3,
            "Transparency Mask": 4,
            "CMYK": 5,
            "YCbCr": 6,
            "CieLAB": 8,
            "CFA": 32803,  # TIFF/EP, Adobe DNG
            "LinearRaw": 32892,  # Adobe DNG
        },
    ),
    263: ("Threshholding", SHORT, 1),
    264: ("CellWidth", SHORT, 1),
    265: ("CellLength", SHORT, 1),
    266: ("FillOrder", SHORT, 1),
    269: ("DocumentName", ASCII, 1),
    270: ("ImageDescription", ASCII, 1),
    271: ("Make", ASCII, 1),
    272: ("Model", ASCII, 1),
    273: ("StripOffsets", LONG, 0),
    274: ("Orientation", SHORT, 1),
    277: ("SamplesPerPixel", SHORT, 1),
    278: ("RowsPerStrip", LONG, 1),
    279: ("StripByteCounts", LONG, 0),
    280: ("MinSampleValue", SHORT, 0),
    281: ("MaxSampleValue", SHORT, 0),
    282: ("XResolution", RATIONAL, 1),
    283: ("YResolution", RATIONAL, 1),
    284: ("PlanarConfiguration", SHORT, 1, {"Contiguous": 1, "Separate": 2}),
    285: ("PageName", ASCII, 1),
    286: ("XPosition", RATIONAL, 1),
    287: ("YPosition", RATIONAL, 1),
    288: ("FreeOffsets", LONG, 1),
    289: ("FreeByteCounts", LONG, 1),
    290: ("GrayResponseUnit", SHORT, 1),
    291: ("GrayResponseCurve", SHORT, 0),
    292: ("T4Options", LONG, 1),
    293: ("T6Options", LONG, 1),
    296: ("ResolutionUnit", SHORT, 1, {"none": 1, "inch": 2, "cm": 3}),
    297: ("PageNumber", SHORT, 2),
    301: ("TransferFunction", SHORT, 0),
    305: ("Software", ASCII, 1),
    306: ("DateTime", ASCII, 1),
    315: ("Artist", ASCII, 1),
    316: ("HostComputer", ASCII, 1),
    317: ("Predictor", SHORT, 1, {"none": 1, "Horizontal Differencing": 2}),
    318: ("WhitePoint", RATIONAL, 2),
    319: ("PrimaryChromaticities", RATIONAL, 6),
    320: ("ColorMap", SHORT, 0),
    321: ("HalftoneHints", SHORT, 2),
    322: ("TileWidth", LONG, 1),
    323: ("TileLength", LONG, 1),
    324: ("TileOffsets", LONG, 0),
    325: ("TileByteCounts", LONG, 0),
    330: ("SubIFDs", LONG, 0),
    332: ("InkSet", SHORT, 1),
    333: ("InkNames", ASCII, 1),
    334: ("NumberOfInks", SHORT, 1),
    336: ("DotRange", SHORT, 0),
    337: ("TargetPrinter", ASCII, 1),
    338: ("ExtraSamples", SHORT, 0),
    339: ("SampleFormat", SHORT, 0),
    340: ("SMinSampleValue", DOUBLE, 0),
    341: ("SMaxSampleValue", DOUBLE, 0),
    342: ("TransferRange", SHORT, 6),
    347: ("JPEGTables", UNDEFINED, 1),
    # obsolete JPEG tags
    512: ("JPEGProc", SHORT, 1),
    513: ("JPEGInterchangeFormat", LONG, 1),
    514: ("JPEGInterchangeFormatLength", LONG, 1),
    515: ("JPEGRestartInterval", SHORT, 1),
    517: ("JPEGLosslessPredictors", SHORT, 0),
    518: ("JPEGPointTransforms", SHORT, 0),
    519: ("JPEGQTables", LONG, 0),
    520: ("JPEGDCTables", LONG, 0),
    521: ("JPEGACTables", LONG, 0),
    529: ("YCbCrCoefficients", RATIONAL, 3),
    530: ("YCbCrSubSampling", SHORT, 2),
    531: ("YCbCrPositioning", SHORT, 1),
    532: ("ReferenceBlackWhite", RATIONAL, 6),
    700: ("XMP", BYTE, 0),
    33432: ("Copyright", ASCII, 1),
    33723: ("IptcNaaInfo", UNDEFINED, 1),
    34377: ("PhotoshopInfo", BYTE, 0),
    # FIXME add more tags here
    34665: ("ExifIFD", LONG, 1),
    34675: ("ICCProfile", UNDEFINED, 1),
    34853: ("GPSInfoIFD", LONG, 1),
    36864: ("ExifVersion", UNDEFINED, 1),
    37724: ("ImageSourceData", UNDEFINED, 1),
    40965: ("InteroperabilityIFD", LONG, 1),
    41730: ("CFAPattern", UNDEFINED, 1),
    # MPInfo
    45056: ("MPFVersion", UNDEFINED, 1),
    45057: ("NumberOfImages", LONG, 1),
    45058: ("MPEntry", UNDEFINED, 1),
    45059: ("ImageUIDList", UNDEFINED, 0),  # UNDONE, check
    45060: ("TotalFrames", LONG, 1),
    45313: ("MPIndividualNum", LONG, 1),
    45569: ("PanOrientation", LONG, 1),
    45570: ("PanOverlap_H", RATIONAL, 1),
    45571: ("PanOverlap_V", RATIONAL, 1),
    45572: ("BaseViewpointNum", LONG, 1),
    45573: ("ConvergenceAngle", SIGNED_RATIONAL, 1),
    45574: ("BaselineLength", RATIONAL, 1),
    45575: ("VerticalDivergence", SIGNED_RATIONAL, 1),
    45576: ("AxisDistance_X", SIGNED_RATIONAL, 1),
    45577: ("AxisDistance_Y", SIGNED_RATIONAL, 1),
    45578: ("AxisDistance_Z", SIGNED_RATIONAL, 1),
    45579: ("YawAngle", SIGNED_RATIONAL, 1),
    45580: ("PitchAngle", SIGNED_RATIONAL, 1),
    45581: ("RollAngle", SIGNED_RATIONAL, 1),
    40960: ("FlashPixVersion", UNDEFINED, 1),
    50741: ("MakerNoteSafety", SHORT, 1, {"Unsafe": 0, "Safe": 1}),
    50780: ("BestQualityScale", RATIONAL, 1),
    50838: ("ImageJMetaDataByteCounts", LONG, 0),  # Can be more than one
    50839: ("ImageJMetaData", UNDEFINED, 1),  # see Issue #2006
}
TAGS_V2_GROUPS = {
    # ExifIFD
    34665: {
        36864: ("ExifVersion", UNDEFINED, 1),
        40960: ("FlashPixVersion", UNDEFINED, 1),
        40965: ("InteroperabilityIFD", LONG, 1),
        41730: ("CFAPattern", UNDEFINED, 1),
    },
    # GPSInfoIFD
    34853: {
        0: ("GPSVersionID", BYTE, 4),
        1: ("GPSLatitudeRef", ASCII, 2),
        2: ("GPSLatitude", RATIONAL, 3),
        3: ("GPSLongitudeRef", ASCII, 2),
        4: ("GPSLongitude", RATIONAL, 3),
        5: ("GPSAltitudeRef", BYTE, 1),
        6: ("GPSAltitude", RATIONAL, 1),
        7: ("GPSTimeStamp", RATIONAL, 3),
        8: ("GPSSatellites", ASCII, 0),
        9: ("GPSStatus", ASCII, 2),
        10: ("GPSMeasureMode", ASCII, 2),
        11: ("GPSDOP", RATIONAL, 1),
        12: ("GPSSpeedRef", ASCII, 2),
        13: ("GPSSpeed", RATIONAL, 1),
        14: ("GPSTrackRef", ASCII, 2),
        15: ("GPSTrack", RATIONAL, 1),
        16: ("GPSImgDirectionRef", ASCII, 2),
        17: ("GPSImgDirection", RATIONAL, 1),
        18: ("GPSMapDatum", ASCII, 0),
        19: ("GPSDestLatitudeRef", ASCII, 2),
        20: ("GPSDestLatitude", RATIONAL, 3),
        21: ("GPSDestLongitudeRef", ASCII, 2),
        22: ("GPSDestLongitude", RATIONAL, 3),
        23: ("GPSDestBearingRef", ASCII, 2),
        24: ("GPSDestBearing", RATIONAL, 1),
        25: ("GPSDestDistanceRef", ASCII, 2),
        26: ("GPSDestDistance", RATIONAL, 1),
        27: ("GPSProcessingMethod", UNDEFINED, 0),
        28: ("GPSAreaInformation", UNDEFINED, 0),
        29: ("GPSDateStamp", ASCII, 11),
        30: ("GPSDifferential", SHORT, 1),
    },
    # InteroperabilityIFD
    40965: {1: ("InteropIndex", ASCII, 1), 2: ("InteropVersion", UNDEFINED, 1)},
}

# Legacy Tags structure
# these tags aren't included above, but were in the previous versions
TAGS = {
    347: "JPEGTables",
    700: "XMP",
    # Additional Exif Info
    32932: "Wang Annotation",
    33434: "ExposureTime",
    33437: "FNumber",
    33445: "MD FileTag",
    33446: "MD ScalePixel",
    33447: "MD ColorTable",
    33448: "MD LabName",
    33449: "MD SampleInfo",
    33450: "MD PrepDate",
    33451: "MD PrepTime",
    33452: "MD FileUnits",
    33550: "ModelPixelScaleTag",
    33723: "IptcNaaInfo",
    33918: "INGR Packet Data Tag",
    33919: "INGR Flag Registers",
    33920: "IrasB Transformation Matrix",
    33922: "ModelTiepointTag",
    34264: "ModelTransformationTag",
    34377: "PhotoshopInfo",
    34735: "GeoKeyDirectoryTag",
    34736: "GeoDoubleParamsTag",
    34737: "GeoAsciiParamsTag",
    34850: "ExposureProgram",
    34852: "SpectralSensitivity",
    34855: "ISOSpeedRatings",
    34856: "OECF",
    34864: "SensitivityType",
    34865: "StandardOutputSensitivity",
    34866: "RecommendedExposureIndex",
    34867: "ISOSpeed",
    34868: "ISOSpeedLatitudeyyy",
    34869: "ISOSpeedLatitudezzz",
    34908: "HylaFAX FaxRecvParams",
    34909: "HylaFAX FaxSubAddress",
    34910: "HylaFAX FaxRecvTime",
    36864: "ExifVersion",
    36867: "DateTimeOriginal",
    36868: "DateTimeDigitized",
    37121: "ComponentsConfiguration",
    37122: "CompressedBitsPerPixel",
    37724: "ImageSourceData",
    37377: "ShutterSpeedValue",
    37378: "ApertureValue",
    37379: "BrightnessValue",
    37380: "ExposureBiasValue",
    37381: "MaxApertureValue",
    37382: "SubjectDistance",
    37383: "MeteringMode",
    37384: "LightSource",
    37385: "Flash",
    37386: "FocalLength",
    37396: "SubjectArea",
    37500: "MakerNote",
    37510: "UserComment",
    37520: "SubSec",
    37521: "SubSecTimeOriginal",
    37522: "SubsecTimeDigitized",
    40960: "FlashPixVersion",
    40961: "ColorSpace",
    40962: "PixelXDimension",
    40963: "PixelYDimension",
    40964: "RelatedSoundFile",
    40965: "InteroperabilityIFD",
    41483: "FlashEnergy",
    41484: "SpatialFrequencyResponse",
    41486: "FocalPlaneXResolution",
    41487: "FocalPlaneYResolution",
    41488: "FocalPlaneResolutionUnit",
    41492: "SubjectLocation",
    41493: "ExposureIndex",
    41495: "SensingMethod",
    41728: "FileSource",
    41729: "SceneType",
    41730: "CFAPattern",
    41985: "CustomRendered",
    41986: "ExposureMode",
    41987: "WhiteBalance",
    41988: "DigitalZoomRatio",
    41989: "FocalLengthIn35mmFilm",
    41990: "SceneCaptureType",
    41991: "GainControl",
    41992: "Contrast",
    41993: "Saturation",
    41994: "Sharpness",
    41995: "DeviceSettingDescription",
    41996: "SubjectDistanceRange",
    42016: "ImageUniqueID",
    42032: "CameraOwnerName",
    42033: "BodySerialNumber",
    42034: "LensSpecification",
    42035: "LensMake",
    42036: "LensModel",
    42037: "LensSerialNumber",
    42112: "GDAL_METADATA",
    42113: "GDAL_NODATA",
    42240: "Gamma",
    50215: "Oce Scanjob Description",
    50216: "Oce Application Selector",
    50217: "Oce Identification Number",
    50218: "Oce ImageLogic Characteristics",
    # Adobe DNG
    50706: "DNGVersion",
    50707: "DNGBackwardVersion",
    50708: "UniqueCameraModel",
    50709: "LocalizedCameraModel",
    50710: "CFAPlaneColor",
    50711: "CFALayout",
    50712: "LinearizationTable",
    50713: "BlackLevelRepeatDim",
    50714: "BlackLevel",
    50715: "BlackLevelDeltaH",
    50716: "BlackLevelDeltaV",
    50717: "WhiteLevel",
    50718: "DefaultScale",
    50719: "DefaultCropOrigin",
    50720: "DefaultCropSize",
    50721: "ColorMatrix1",
    50722: "ColorMatrix2",
    50723: "CameraCalibration1",
    50724: "CameraCalibration2",
    50725: "ReductionMatrix1",
    50726: "ReductionMatrix2",
    50727: "AnalogBalance",
    50728: "AsShotNeutral",
    50729: "AsShotWhiteXY",
    50730: "BaselineExposure",
    50731: "BaselineNoise",
    50732: "BaselineSharpness",
    50733: "BayerGreenSplit",
    50734: "LinearResponseLimit",
    50735: "CameraSerialNumber",
    50736: "LensInfo",
    50737: "ChromaBlurRadius",
    50738: "AntiAliasStrength",
    50740: "DNGPrivateData",
    50778: "CalibrationIlluminant1",
    50779: "CalibrationIlluminant2",
    50784: "Alias Layer Metadata",
}


def _populate():
    for k, v in TAGS_V2.items():
        # Populate legacy structure.
        TAGS[k] = v[0]
        if len(v) == 4:
            for sk, sv in v[3].items():
                TAGS[(k, sv)] = sk

        TAGS_V2[k] = TagInfo(k, *v)

    for group, tags in TAGS_V2_GROUPS.items():
        for k, v in tags.items():
            tags[k] = TagInfo(k, *v)


_populate()
##
# Map type numbers to type names -- defined in ImageFileDirectory.

TYPES = {}

# was:
# TYPES = {
#     1: "byte",
#     2: "ascii",
#     3: "short",
#     4: "long",
#     5: "rational",
#     6: "signed byte",
#     7: "undefined",
#     8: "signed short",
#     9: "signed long",
#     10: "signed rational",
#     11: "float",
#     12: "double",
# }

#
# These tags are handled by default in libtiff, without
# adding to the custom dictionary. From tif_dir.c, searching for
# case TIFFTAG in the _TIFFVSetField function:
# Line: item.
# 148: case TIFFTAG_SUBFILETYPE:
# 151: case TIFFTAG_IMAGEWIDTH:
# 154: case TIFFTAG_IMAGELENGTH:
# 157: case TIFFTAG_BITSPERSAMPLE:
# 181: case TIFFTAG_COMPRESSION:
# 202: case TIFFTAG_PHOTOMETRIC:
# 205: case TIFFTAG_THRESHHOLDING:
# 208: case TIFFTAG_FILLORDER:
# 214: case TIFFTAG_ORIENTATION:
# 221: case TIFFTAG_SAMPLESPERPIXEL:
# 228: case TIFFTAG_ROWSPERSTRIP:
# 238: case TIFFTAG_MINSAMPLEVALUE:
# 241: case TIFFTAG_MAXSAMPLEVALUE:
# 244: case TIFFTAG_SMINSAMPLEVALUE:
# 247: case TIFFTAG_SMAXSAMPLEVALUE:
# 250: case TIFFTAG_XRESOLUTION:
# 256: case TIFFTAG_YRESOLUTION:
# 262: case TIFFTAG_PLANARCONFIG:
# 268: case TIFFTAG_XPOSITION:
# 271: case TIFFTAG_YPOSITION:
# 274: case TIFFTAG_RESOLUTIONUNIT:
# 280: case TIFFTAG_PAGENUMBER:
# 284: case TIFFTAG_HALFTONEHINTS:
# 288: case TIFFTAG_COLORMAP:
# 294: case TIFFTAG_EXTRASAMPLES:
# 298: case TIFFTAG_MATTEING:
# 305: case TIFFTAG_TILEWIDTH:
# 316: case TIFFTAG_TILELENGTH:
# 327: case TIFFTAG_TILEDEPTH:
# 333: case TIFFTAG_DATATYPE:
# 344: case TIFFTAG_SAMPLEFORMAT:
# 361: case TIFFTAG_IMAGEDEPTH:
# 364: case TIFFTAG_SUBIFD:
# 376: case TIFFTAG_YCBCRPOSITIONING:
# 379: case TIFFTAG_YCBCRSUBSAMPLING:
# 383: case TIFFTAG_TRANSFERFUNCTION:
# 389: case TIFFTAG_REFERENCEBLACKWHITE:
# 393: case TIFFTAG_INKNAMES:

# Following pseudo-tags are also handled by default in libtiff:
# TIFFTAG_JPEGQUALITY 65537

# some of these are not in our TAGS_V2 dict and were included from tiff.h

# This list also exists in encode.c
LIBTIFF_CORE = {
    255,
    256,
    257,
    258,
    259,
    262,
    263,
    266,
    274,
    277,
    278,
    280,
    281,
    340,
    341,
    282,
    283,
    284,
    286,
    287,
    296,
    297,
    321,
    320,
    338,
    32995,
    322,
    323,
    32998,
    32996,
    339,
    32997,
    330,
    531,
    530,
    301,
    532,
    333,
    # as above
    269,  # this has been in our tests forever, and works
    65537,
}

LIBTIFF_CORE.remove(255)  # We don't have support for subfiletypes
LIBTIFF_CORE.remove(322)  # We don't have support for writing tiled images with libtiff
LIBTIFF_CORE.remove(323)  # Tiled images
LIBTIFF_CORE.remove(333)  # Ink Names either

# Note to advanced users: There may be combinations of these
# parameters and values that when added properly, will work and
# produce valid tiff images that may work in your application.
# It is safe to add and remove tags from this set from Pillow's point
# of view so long as you test against libtiff.
