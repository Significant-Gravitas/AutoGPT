import getopt
import sys
import traceback

import win32api
import win32net
import win32netcon
import win32security

verbose_level = 0

server = None  # Run on local machine.


def verbose(msg):
    if verbose_level:
        print(msg)


def CreateUser():
    "Creates a new test user, then deletes the user"
    testName = "PyNetTestUser"
    try:
        win32net.NetUserDel(server, testName)
        print("Warning - deleted user before creating it!")
    except win32net.error:
        pass

    d = {}
    d["name"] = testName
    d["password"] = "deleteme"
    d["priv"] = win32netcon.USER_PRIV_USER
    d["comment"] = "Delete me - created by Python test code"
    d["flags"] = win32netcon.UF_NORMAL_ACCOUNT | win32netcon.UF_SCRIPT
    win32net.NetUserAdd(server, 1, d)
    try:
        try:
            win32net.NetUserChangePassword(server, testName, "wrong", "new")
            print("ERROR: NetUserChangePassword worked with a wrong password!")
        except win32net.error:
            pass
        win32net.NetUserChangePassword(server, testName, "deleteme", "new")
    finally:
        win32net.NetUserDel(server, testName)
    print("Created a user, changed their password, and deleted them!")


def UserEnum():
    "Enumerates all the local users"
    resume = 0
    nuser = 0
    while 1:
        data, total, resume = win32net.NetUserEnum(
            server, 3, win32netcon.FILTER_NORMAL_ACCOUNT, resume
        )
        verbose(
            "Call to NetUserEnum obtained %d entries of %d total" % (len(data), total)
        )
        for user in data:
            verbose("Found user %s" % user["name"])
            nuser = nuser + 1
        if not resume:
            break
    assert nuser, "Could not find any users!"
    print("Enumerated all the local users")


def GroupEnum():
    "Enumerates all the domain groups"
    nmembers = 0
    resume = 0
    while 1:
        data, total, resume = win32net.NetGroupEnum(server, 1, resume)
        #               print "Call to NetGroupEnum obtained %d entries of %d total" % (len(data), total)
        for group in data:
            verbose("Found group %(name)s:%(comment)s " % group)
            memberresume = 0
            while 1:
                memberdata, total, memberresume = win32net.NetGroupGetUsers(
                    server, group["name"], 0, resume
                )
                for member in memberdata:
                    verbose(" Member %(name)s" % member)
                    nmembers = nmembers + 1
                if memberresume == 0:
                    break
        if not resume:
            break
    assert nmembers, "Couldnt find a single member in a single group!"
    print("Enumerated all the groups")


def LocalGroupEnum():
    "Enumerates all the local groups"
    resume = 0
    nmembers = 0
    while 1:
        data, total, resume = win32net.NetLocalGroupEnum(server, 1, resume)
        for group in data:
            verbose("Found group %(name)s:%(comment)s " % group)
            memberresume = 0
            while 1:
                memberdata, total, memberresume = win32net.NetLocalGroupGetMembers(
                    server, group["name"], 2, resume
                )
                for member in memberdata:
                    # Just for the sake of it, we convert the SID to a username
                    username, domain, type = win32security.LookupAccountSid(
                        server, member["sid"]
                    )
                    nmembers = nmembers + 1
                    verbose(" Member %s (%s)" % (username, member["domainandname"]))
                if memberresume == 0:
                    break
        if not resume:
            break
    assert nmembers, "Couldnt find a single member in a single group!"
    print("Enumerated all the local groups")


def ServerEnum():
    "Enumerates all servers on the network"
    resume = 0
    while 1:
        data, total, resume = win32net.NetServerEnum(
            server, 100, win32netcon.SV_TYPE_ALL, None, resume
        )
        for s in data:
            verbose("Found server %s" % s["name"])
            # Now loop over the shares.
            shareresume = 0
            while 1:
                sharedata, total, shareresume = win32net.NetShareEnum(
                    server, 2, shareresume
                )
                for share in sharedata:
                    verbose(
                        " %(netname)s (%(path)s):%(remark)s - in use by %(current_uses)d users"
                        % share
                    )
                if not shareresume:
                    break
        if not resume:
            break
    print("Enumerated all the servers on the network")


def LocalGroup(uname=None):
    "Creates a local group, adds some members, deletes them, then removes the group"
    level = 3
    if uname is None:
        uname = win32api.GetUserName()
    if uname.find("\\") < 0:
        uname = win32api.GetDomainName() + "\\" + uname
    group = "python_test_group"
    # delete the group if it already exists
    try:
        win32net.NetLocalGroupDel(server, group)
        print("WARNING: existing local group '%s' has been deleted.")
    except win32net.error:
        pass
    group_data = {"name": group}
    win32net.NetLocalGroupAdd(server, 1, group_data)
    try:
        u = {"domainandname": uname}
        win32net.NetLocalGroupAddMembers(server, group, level, [u])
        mem, tot, res = win32net.NetLocalGroupGetMembers(server, group, level)
        print("members are", mem)
        if mem[0]["domainandname"] != uname:
            print("ERROR: LocalGroup just added %s, but members are %r" % (uname, mem))
        # Convert the list of dicts to a list of strings.
        win32net.NetLocalGroupDelMembers(
            server, group, [m["domainandname"] for m in mem]
        )
    finally:
        win32net.NetLocalGroupDel(server, group)
    print("Created a local group, added and removed members, then deleted the group")


def GetInfo(userName=None):
    "Dumps level 3 information about the current user"
    if userName is None:
        userName = win32api.GetUserName()
    print("Dumping level 3 information about user")
    info = win32net.NetUserGetInfo(server, userName, 3)
    for key, val in list(info.items()):
        verbose("%s=%s" % (key, val))


def SetInfo(userName=None):
    "Attempts to change the current users comment, then set it back"
    if userName is None:
        userName = win32api.GetUserName()
    oldData = win32net.NetUserGetInfo(server, userName, 3)
    try:
        d = oldData.copy()
        d["usr_comment"] = "Test comment"
        win32net.NetUserSetInfo(server, userName, 3, d)
        new = win32net.NetUserGetInfo(server, userName, 3)["usr_comment"]
        if str(new) != "Test comment":
            raise RuntimeError("Could not read the same comment back - got %s" % new)
        print("Changed the data for the user")
    finally:
        win32net.NetUserSetInfo(server, userName, 3, oldData)


def SetComputerInfo():
    "Doesnt actually change anything, just make sure we could ;-)"
    info = win32net.NetWkstaGetInfo(None, 502)
    # *sob* - but we can't!  Why not!!!
    # win32net.NetWkstaSetInfo(None, 502, info)


def usage(tests):
    import os

    print("Usage: %s [-s server ] [-v] [Test ...]" % os.path.basename(sys.argv[0]))
    print("  -v : Verbose - print more information")
    print("  -s : server - execute the tests against the named server")
    print("  -c : include the CreateUser test by default")
    print("where Test is one of:")
    for t in tests:
        print(t.__name__, ":", t.__doc__)
    print()
    print("If not tests are specified, all tests are run")
    sys.exit(1)


def main():
    tests = []
    for ob in list(globals().values()):
        if type(ob) == type(main) and ob.__doc__:
            tests.append(ob)
    opts, args = getopt.getopt(sys.argv[1:], "s:hvc")
    create_user = False
    for opt, val in opts:
        if opt == "-s":
            global server
            server = val
        if opt == "-h":
            usage(tests)
        if opt == "-v":
            global verbose_level
            verbose_level = verbose_level + 1
        if opt == "-c":
            create_user = True

    if len(args) == 0:
        print("Running all tests - use '-h' to see command-line options...")
        dotests = tests
        if not create_user:
            dotests.remove(CreateUser)
    else:
        dotests = []
        for arg in args:
            for t in tests:
                if t.__name__ == arg:
                    dotests.append(t)
                    break
            else:
                print("Test '%s' unknown - skipping" % arg)
    if not len(dotests):
        print("Nothing to do!")
        usage(tests)
    for test in dotests:
        try:
            test()
        except:
            print("Test %s failed" % test.__name__)
            traceback.print_exc()


if __name__ == "__main__":
    main()
