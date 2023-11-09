import win32api
import win32file


def ProgressRoutine(
    TotalFileSize,
    TotalBytesTransferred,
    StreamSize,
    StreamBytesTransferred,
    StreamNumber,
    CallbackReason,
    SourceFile,
    DestinationFile,
    Data,
):
    print(Data)
    print(
        TotalFileSize,
        TotalBytesTransferred,
        StreamSize,
        StreamBytesTransferred,
        StreamNumber,
        CallbackReason,
        SourceFile,
        DestinationFile,
    )
    ##if TotalBytesTransferred > 100000:
    ##    return win32file.PROGRESS_STOP
    return win32file.PROGRESS_CONTINUE


temp_dir = win32api.GetTempPath()
fsrc = win32api.GetTempFileName(temp_dir, "cfe")[0]
fdst = win32api.GetTempFileName(temp_dir, "cfe")[0]
print(fsrc, fdst)

f = open(fsrc, "w")
f.write("xxxxxxxxxxxxxxxx\n" * 32768)
f.close()
## add a couple of extra data streams
f = open(fsrc + ":stream_y", "w")
f.write("yyyyyyyyyyyyyyyy\n" * 32768)
f.close()
f = open(fsrc + ":stream_z", "w")
f.write("zzzzzzzzzzzzzzzz\n" * 32768)
f.close()

operation_desc = "Copying " + fsrc + " to " + fdst
win32file.CopyFileEx(
    fsrc,
    fdst,
    ProgressRoutine,
    Data=operation_desc,
    Cancel=False,
    CopyFlags=win32file.COPY_FILE_RESTARTABLE,
    Transaction=None,
)
