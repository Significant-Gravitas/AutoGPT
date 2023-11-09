#
# This file is part of pyasn1-modules software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
import base64
import sys

stSpam, stHam, stDump = 0, 1, 2


# The markers parameters is in form ('start1', 'stop1'), ('start2', 'stop2')...
# Return is (marker-index, substrate)
def readPemBlocksFromFile(fileObj, *markers):
    startMarkers = dict(map(lambda x: (x[1], x[0]),
                            enumerate(map(lambda y: y[0], markers))))
    stopMarkers = dict(map(lambda x: (x[1], x[0]),
                           enumerate(map(lambda y: y[1], markers))))
    idx = -1
    substrate = ''
    certLines = []
    state = stSpam
    while True:
        certLine = fileObj.readline()
        if not certLine:
            break
        certLine = certLine.strip()
        if state == stSpam:
            if certLine in startMarkers:
                certLines = []
                idx = startMarkers[certLine]
                state = stHam
                continue
        if state == stHam:
            if certLine in stopMarkers and stopMarkers[certLine] == idx:
                state = stDump
            else:
                certLines.append(certLine)
        if state == stDump:
            if sys.version_info[0] <= 2:
                substrate = ''.join([base64.b64decode(x) for x in certLines])
            else:
                substrate = ''.encode().join([base64.b64decode(x.encode()) for x in certLines])
            break
    return idx, substrate


# Backward compatibility routine
def readPemFromFile(fileObj,
                    startMarker='-----BEGIN CERTIFICATE-----',
                    endMarker='-----END CERTIFICATE-----'):
    idx, substrate = readPemBlocksFromFile(fileObj, (startMarker, endMarker))
    return substrate


def readBase64fromText(text):
    if sys.version_info[0] <= 2:
        return base64.b64decode(text)
    else:
        return base64.b64decode(text.encode())


def readBase64FromFile(fileObj):
    return readBase64fromText(fileObj.read())
