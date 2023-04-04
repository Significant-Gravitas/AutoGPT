# Sample implementation of IFileOperationProgressSink that just prints
# some basic info

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


class FileOperationProgressSink(DesignatedWrapPolicy):
    _com_interfaces_ = [shell.IID_IFileOperationProgressSink]
    _public_methods_ = [
        "StartOperations",
        "FinishOperations",
        "PreRenameItem",
        "PostRenameItem",
        "PreMoveItem",
        "PostMoveItem",
        "PreCopyItem",
        "PostCopyItem",
        "PreDeleteItem",
        "PostDeleteItem",
        "PreNewItem",
        "PostNewItem",
        "UpdateProgress",
        "ResetTimer",
        "PauseTimer",
        "ResumeTimer",
    ]

    def __init__(self):
        self._wrap_(self)

    def StartOperations(self):
        print("StartOperations")

    def FinishOperations(self, Result):
        print("FinishOperations: HRESULT ", Result)

    def PreRenameItem(self, Flags, Item, NewName):
        print(
            "PreRenameItem: Renaming "
            + Item.GetDisplayName(shellcon.SHGDN_FORPARSING)
            + " to "
            + NewName
        )

    def PostRenameItem(self, Flags, Item, NewName, hrRename, NewlyCreated):
        if NewlyCreated is not None:
            newfile = NewlyCreated.GetDisplayName(shellcon.SHGDN_FORPARSING)
        else:
            newfile = "not renamed, HRESULT " + str(hrRename)
        print(
            "PostRenameItem: renamed "
            + Item.GetDisplayName(shellcon.SHGDN_FORPARSING)
            + " to "
            + newfile
        )

    def PreMoveItem(self, Flags, Item, DestinationFolder, NewName):
        print(
            "PreMoveItem: Moving "
            + Item.GetDisplayName(shellcon.SHGDN_FORPARSING)
            + " to "
            + DestinationFolder.GetDisplayName(shellcon.SHGDN_FORPARSING)
            + "\\"
            + str(NewName)
        )

    def PostMoveItem(
        self, Flags, Item, DestinationFolder, NewName, hrMove, NewlyCreated
    ):
        if NewlyCreated is not None:
            newfile = NewlyCreated.GetDisplayName(shellcon.SHGDN_FORPARSING)
        else:
            newfile = "not copied, HRESULT " + str(hrMove)
        print(
            "PostMoveItem: Moved "
            + Item.GetDisplayName(shellcon.SHGDN_FORPARSING)
            + " to "
            + newfile
        )

    def PreCopyItem(self, Flags, Item, DestinationFolder, NewName):
        if not NewName:
            NewName = ""
        print(
            "PreCopyItem: Copying "
            + Item.GetDisplayName(shellcon.SHGDN_FORPARSING)
            + " to "
            + DestinationFolder.GetDisplayName(shellcon.SHGDN_FORPARSING)
            + "\\"
            + NewName
        )
        print("Flags: ", decode_flags(Flags))

    def PostCopyItem(
        self, Flags, Item, DestinationFolder, NewName, hrCopy, NewlyCreated
    ):
        if NewlyCreated is not None:
            newfile = NewlyCreated.GetDisplayName(shellcon.SHGDN_FORPARSING)
        else:
            newfile = "not copied, HRESULT " + str(hrCopy)
        print(
            "PostCopyItem: Copied "
            + Item.GetDisplayName(shellcon.SHGDN_FORPARSING)
            + " to "
            + newfile
        )
        print("Flags: ", decode_flags(Flags))

    def PreDeleteItem(self, Flags, Item):
        print(
            "PreDeleteItem: Deleting " + Item.GetDisplayName(shellcon.SHGDN_FORPARSING)
        )

    def PostDeleteItem(self, Flags, Item, hrDelete, NewlyCreated):
        print(
            "PostDeleteItem: Deleted " + Item.GetDisplayName(shellcon.SHGDN_FORPARSING)
        )
        if NewlyCreated:
            print(
                "	Moved to recycle bin - "
                + NewlyCreated.GetDisplayName(shellcon.SHGDN_FORPARSING)
            )

    def PreNewItem(self, Flags, DestinationFolder, NewName):
        print(
            "PreNewItem: Creating "
            + DestinationFolder.GetDisplayName(shellcon.SHGDN_FORPARSING)
            + "\\"
            + NewName
        )

    def PostNewItem(
        self,
        Flags,
        DestinationFolder,
        NewName,
        TemplateName,
        FileAttributes,
        hrNew,
        NewItem,
    ):
        print(
            "PostNewItem: Created " + NewItem.GetDisplayName(shellcon.SHGDN_FORPARSING)
        )

    def UpdateProgress(self, WorkTotal, WorkSoFar):
        print("UpdateProgress: ", WorkSoFar, WorkTotal)

    def ResetTimer(self):
        print("ResetTimer")

    def PauseTimer(self):
        print("PauseTimer")

    def ResumeTimer(self):
        print("ResumeTimer")


def CreateSink():
    return pythoncom.WrapObject(
        FileOperationProgressSink(), shell.IID_IFileOperationProgressSink
    )
