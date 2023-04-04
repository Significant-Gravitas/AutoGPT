import string
import time
import traceback

import pythoncom
import win32com.client
import win32com.client.gencache
import win32con

constants = win32com.client.constants

win32com.client.gencache.EnsureModule("{783CD4E0-9D54-11CF-B8EE-00608CC9A71F}", 0, 5, 0)

error = "vssutil error"


def GetSS():
    ss = win32com.client.Dispatch("SourceSafe")
    # SS seems a bit weird.  It defaults the arguments as empty strings, but
    # then complains when they are used - so we pass "Missing"
    ss.Open(pythoncom.Missing, pythoncom.Missing, pythoncom.Missing)
    return ss


def test(projectName):
    ss = GetSS()
    project = ss.VSSItem(projectName)

    for item in project.GetVersions(constants.VSSFLAG_RECURSYES):
        print(item.VSSItem.Name, item.VersionNumber, item.Action)


# 	item=i.Versions[0].VSSItem
# 	for h in i.Versions:
# 		print `h.Comment`, h.Action, h.VSSItem.Name


def SubstituteInString(inString, evalEnv):
    substChar = "$"
    fields = string.split(inString, substChar)
    newFields = []
    for i in range(len(fields)):
        didSubst = 0
        strVal = fields[i]
        if i % 2 != 0:
            try:
                strVal = eval(strVal, evalEnv[0], evalEnv[1])
                newFields.append(strVal)
                didSubst = 1
            except:
                traceback.print_exc()
                print("Could not substitute", strVal)
        if not didSubst:
            newFields.append(strVal)
    return string.join(map(str, newFields), "")


def SubstituteInFile(inName, outName, evalEnv):
    inFile = open(inName, "r")
    try:
        outFile = open(outName, "w")
        try:
            while 1:
                line = inFile.read()
                if not line:
                    break
                outFile.write(SubstituteInString(line, evalEnv))
        finally:
            outFile.close()
    finally:
        inFile.close()


def VssLog(project, linePrefix="", noLabels=5, maxItems=150):
    lines = []
    num = 0
    labelNum = 0
    for i in project.GetVersions(constants.VSSFLAG_RECURSYES):
        num = num + 1
        if num > maxItems:
            break
        commentDesc = itemDesc = ""
        if i.Action[:5] == "Added":
            continue
        if len(i.Label):
            labelNum = labelNum + 1
            itemDesc = i.Action
        else:
            itemDesc = i.VSSItem.Name
            if str(itemDesc[-4:]) == ".dsp":
                continue
        if i.Comment:
            commentDesc = "\n%s\t%s" % (linePrefix, i.Comment)
        lines.append(
            "%s%s\t%s%s"
            % (
                linePrefix,
                time.asctime(time.localtime(int(i.Date))),
                itemDesc,
                commentDesc,
            )
        )
        if labelNum > noLabels:
            break
    return string.join(lines, "\n")


def SubstituteVSSInFile(projectName, inName, outName):
    import win32api

    if win32api.GetFullPathName(inName) == win32api.GetFullPathName(outName):
        raise RuntimeError("The input and output filenames can not be the same")
    sourceSafe = GetSS()
    project = sourceSafe.VSSItem(projectName)
    # Find the last label
    label = None
    for version in project.Versions:
        if version.Label:
            break
    else:
        print("Couldnt find a label in the sourcesafe project!")
        return
    # Setup some local helpers for the conversion strings.
    vss_label = version.Label
    vss_date = time.asctime(time.localtime(int(version.Date)))
    now = time.asctime(time.localtime(time.time()))
    SubstituteInFile(inName, outName, (locals(), globals()))


def CountCheckouts(item):
    num = 0
    if item.Type == constants.VSSITEM_PROJECT:
        for sub in item.Items:
            num = num + CountCheckouts(sub)
    else:
        if item.IsCheckedOut:
            num = num + 1
    return num


def GetLastBuildNo(project):
    i = GetSS().VSSItem(project)
    # Find the last label
    lab = None
    for version in i.Versions:
        lab = str(version.Label)
        if lab:
            return lab
    return None


def MakeNewBuildNo(project, buildDesc=None, auto=0, bRebrand=0):
    if buildDesc is None:
        buildDesc = "Created by Python"
    ss = GetSS()
    i = ss.VSSItem(project)
    num = CountCheckouts(i)
    if num > 0:
        msg = (
            "This project has %d items checked out\r\n\r\nDo you still want to continue?"
            % num
        )
        import win32ui

        if win32ui.MessageBox(msg, project, win32con.MB_YESNO) != win32con.IDYES:
            return

    oldBuild = buildNo = GetLastBuildNo(project)
    if buildNo is None:
        buildNo = "1"
        oldBuild = "<None>"
    else:
        try:
            buildNo = string.atoi(buildNo)
            if not bRebrand:
                buildNo = buildNo + 1
            buildNo = str(buildNo)
        except ValueError:
            raise error("The previous label could not be incremented: %s" % (oldBuild))

    if not auto:
        from pywin.mfc import dialog

        buildNo = dialog.GetSimpleInput(
            "Enter new build number", buildNo, "%s - Prev: %s" % (project, oldBuild)
        )
        if buildNo is None:
            return
    i.Label(buildNo, "Build %s: %s" % (buildNo, buildDesc))
    if auto:
        print("Branded project %s with label %s" % (project, buildNo))
    return buildNo


if __name__ == "__main__":
    # 	UpdateWiseExeName("PyWiseTest.wse", "PyWiseTest-10.exe")

    # 	MakeVersion()
    # 	test(tp)
    # 	MakeNewBuildNo(tp)
    tp = "\\Python\\Python Win32 Extensions"
    SubstituteVSSInFile(
        tp, "d:\\src\\pythonex\\win32\\win32.txt", "d:\\temp\\win32.txt"
    )
