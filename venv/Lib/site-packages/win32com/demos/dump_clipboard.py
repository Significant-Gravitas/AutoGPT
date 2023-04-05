import pythoncom
import win32con

formats = """CF_TEXT CF_BITMAP CF_METAFILEPICT CF_SYLK CF_DIF CF_TIFF
            CF_OEMTEXT CF_DIB CF_PALETTE CF_PENDATA CF_RIFF CF_WAVE
            CF_UNICODETEXT CF_ENHMETAFILE CF_HDROP CF_LOCALE CF_MAX
            CF_OWNERDISPLAY CF_DSPTEXT CF_DSPBITMAP CF_DSPMETAFILEPICT
            CF_DSPENHMETAFILE""".split()
format_name_map = {}
for f in formats:
    val = getattr(win32con, f)
    format_name_map[val] = f

tymeds = [attr for attr in pythoncom.__dict__.keys() if attr.startswith("TYMED_")]


def DumpClipboard():
    do = pythoncom.OleGetClipboard()
    print("Dumping all clipboard formats...")
    for fe in do.EnumFormatEtc():
        fmt, td, aspect, index, tymed = fe
        tymeds_this = [
            getattr(pythoncom, t) for t in tymeds if tymed & getattr(pythoncom, t)
        ]
        print("Clipboard format", format_name_map.get(fmt, str(fmt)))
        for t_this in tymeds_this:
            # As we are enumerating there should be no need to call
            # QueryGetData, but we do anyway!
            fetc_query = fmt, td, aspect, index, t_this
            try:
                do.QueryGetData(fetc_query)
            except pythoncom.com_error:
                print("Eeek - QGD indicated failure for tymed", t_this)
            # now actually get it.
            try:
                medium = do.GetData(fetc_query)
            except pythoncom.com_error as exc:
                print("Failed to get the clipboard data:", exc)
                continue
            if medium.tymed == pythoncom.TYMED_GDI:
                data = "GDI handle %d" % medium.data
            elif medium.tymed == pythoncom.TYMED_MFPICT:
                data = "METAFILE handle %d" % medium.data
            elif medium.tymed == pythoncom.TYMED_ENHMF:
                data = "ENHMETAFILE handle %d" % medium.data
            elif medium.tymed == pythoncom.TYMED_HGLOBAL:
                data = "%d bytes via HGLOBAL" % len(medium.data)
            elif medium.tymed == pythoncom.TYMED_FILE:
                data = "filename '%s'" % data
            elif medium.tymed == pythoncom.TYMED_ISTREAM:
                stream = medium.data
                stream.Seek(0, 0)
                bytes = 0
                while 1:
                    chunk = stream.Read(4096)
                    if not chunk:
                        break
                    bytes += len(chunk)
                data = "%d bytes via IStream" % bytes
            elif medium.tymed == pythoncom.TYMED_ISTORAGE:
                data = "a IStorage"
            else:
                data = "*** unknown tymed!"
            print(" -> got", data)
    do = None


if __name__ == "__main__":
    DumpClipboard()
    if pythoncom._GetInterfaceCount() + pythoncom._GetGatewayCount():
        print(
            "XXX - Leaving with %d/%d COM objects alive"
            % (pythoncom._GetInterfaceCount(), pythoncom._GetGatewayCount())
        )
