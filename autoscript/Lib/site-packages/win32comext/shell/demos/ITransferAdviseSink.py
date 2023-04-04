# ITransferAdviseSink implementation template

import pythoncom
from win32com.server.policy import DesignatedWrapPolicy
from win32com.shell import shell, shellcon

tsf_flags = list(
    (k, v) for k, v in list(shellcon.__dict__.items()) if k.startswith("TSF_")
)


def decode_flags(flags):
    if flags == 0:
        return "TSF_NORMAL"
    flag_txt = ""
    for k, v in tsf_flags:
        if flags & v:
            if flag_txt:
                flag_txt = flag_txt + "|" + k
            else:
                flag_txt = k
    return flag_txt


TRANSFER_ADVISE_STATES = {}
for k, v in list(shellcon.__dict__.items()):
    if k.startswith("TS_"):
        TRANSFER_ADVISE_STATES[v] = k


def decode_flags(flags):
    if flags == 0:
        return "TSF_NORMAL"
    flag_txt = ""
    for k, v in tsf_flags:
        if flags & v:
            if flag_txt:
                flag_txt = flag_txt + "|" + k
            else:
                flag_txt = k
    return flag_txt


class TransferAdviseSink(DesignatedWrapPolicy):
    _com_interfaces_ = [shell.IID_ITransferAdviseSink]
    _public_methods_ = [
        "UpdateProgress",
        "UpdateTransferState",
        "ConfirmOverwrite",
        "ConfirmEncryptionLoss",
        "FileFailure",
        "SubStreamFailure",
        "PropertyFailure",
    ]

    def __init__(self):
        self._wrap_(self)

    def UpdateProgress(
        self,
        SizeCurrent,
        SizeTotal,
        FilesCurrent,
        FilesTotal,
        FoldersCurrent,
        FoldersTotal,
    ):
        print("UpdateProgress - processed so far:")
        print("\t %s out of %s bytes" % (SizeCurrent, SizeTotal))
        print("\t %s out of %s files" % (FilesCurrent, FilesTotal))
        print("\t %s out of %s folders" % (FoldersCurrent, FoldersTotal))

    def UpdateTransferState(self, State):
        print(
            "Current state: ",
            TRANSFER_ADVISE_STATES.get(State, "??? Unknown state %s ???" % State),
        )

    def ConfirmOverwrite(self, Source, DestParent, Name):
        print(
            "ConfirmOverwrite: ",
            Source.GetDisplayName(shellcon.SHGDN_FORPARSING),
            DestParent.GetDisplayName(shellcon.SHGDN_FORPARSING),
            Name,
        )

    def ConfirmEncryptionLoss(self, Source):
        print(
            "ConfirmEncryptionLoss:", Source.GetDisplayName(shellcon.SHGDN_FORPARSING)
        )

    def FileFailure(self, Item, ItemName, Error):
        print("FileFailure:", Item.GetDisplayName(shellcon.SHGDN_FORPARSING), ItemName)

    def SubStreamFailure(self, Item, StreamName, Error):
        print("SubStreamFailure:\n")

    def PropertyFailure(self, Item, key, Error):
        print("PropertyFailure:\n")


def CreateSink():
    return pythoncom.WrapObject(
        TransferAdviseSink(),
        shell.IID_ITransferAdviseSink,
        shell.IID_ITransferAdviseSink,
    )
