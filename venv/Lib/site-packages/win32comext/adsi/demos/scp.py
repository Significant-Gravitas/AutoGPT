"""A re-implementation of the MS DirectoryService samples related to services.

* Adds and removes an ActiveDirectory "Service Connection Point",
  including managing the security on the object.
* Creates and registers Service Principal Names.
* Changes the username for a domain user.

Some of these functions are likely to become move to a module - but there
is also a little command-line-interface to try these functions out.

For example:

scp.py --account-name=domain\\user --service-class=PythonScpTest \\
       --keyword=foo --keyword=bar --binding-string=bind_info \\
       ScpCreate SpnCreate SpnRegister

would:
* Attempt to delete a Service Connection Point for the service class
  'PythonScpTest'
* Attempt to create a Service Connection Point for that class, with 2
  keywords and a binding string of 'bind_info'
* Create a Service Principal Name for the service and register it

to undo those changes, you could execute:

scp.py --account-name=domain\\user --service-class=PythonScpTest \\
       SpnCreate SpnUnregister ScpDelete

which will:
* Create a SPN
* Unregister that SPN from the Active Directory.
* Delete the Service Connection Point

Executing with --test will create and remove one of everything.
"""

import optparse
import textwrap
import traceback

import ntsecuritycon as dscon
import win32api
import win32con
import win32security
import winerror
from win32com.adsi import adsi
from win32com.adsi.adsicon import *
from win32com.client import Dispatch

verbose = 1
g_createdSCP = None
g_createdSPNs = []
g_createdSPNLast = None

import logging

logger = logging  # use logging module global methods for now.

# still a bit confused about log(n, ...) vs logger.info/debug()


# Returns distinguished name of SCP.
def ScpCreate(
    service_binding_info,
    service_class_name,  # Service class string to store in SCP.
    account_name=None,  # Logon account that needs access to SCP.
    container_name=None,
    keywords=None,
    object_class="serviceConnectionPoint",
    dns_name_type="A",
    dn=None,
    dns_name=None,
):
    container_name = container_name or service_class_name
    if not dns_name:
        # Get the DNS name of the local computer
        dns_name = win32api.GetComputerNameEx(win32con.ComputerNameDnsFullyQualified)
    # Get the distinguished name of the computer object for the local computer
    if dn is None:
        dn = win32api.GetComputerObjectName(win32con.NameFullyQualifiedDN)

    # Compose the ADSpath and bind to the computer object for the local computer
    comp = adsi.ADsGetObject("LDAP://" + dn, adsi.IID_IDirectoryObject)

    # Publish the SCP as a child of the computer object
    keywords = keywords or []
    # Fill in the attribute values to be stored in the SCP.
    attrs = [
        ("cn", ADS_ATTR_UPDATE, ADSTYPE_CASE_IGNORE_STRING, (container_name,)),
        ("objectClass", ADS_ATTR_UPDATE, ADSTYPE_CASE_IGNORE_STRING, (object_class,)),
        ("keywords", ADS_ATTR_UPDATE, ADSTYPE_CASE_IGNORE_STRING, keywords),
        ("serviceDnsName", ADS_ATTR_UPDATE, ADSTYPE_CASE_IGNORE_STRING, (dns_name,)),
        (
            "serviceDnsNameType",
            ADS_ATTR_UPDATE,
            ADSTYPE_CASE_IGNORE_STRING,
            (dns_name_type,),
        ),
        (
            "serviceClassName",
            ADS_ATTR_UPDATE,
            ADSTYPE_CASE_IGNORE_STRING,
            (service_class_name,),
        ),
        (
            "serviceBindingInformation",
            ADS_ATTR_UPDATE,
            ADSTYPE_CASE_IGNORE_STRING,
            (service_binding_info,),
        ),
    ]
    new = comp.CreateDSObject("cn=" + container_name, attrs)
    logger.info("New connection point is at %s", container_name)
    # Wrap in a usable IDispatch object.
    new = Dispatch(new)
    # And allow access to the SCP for the specified account name
    AllowAccessToScpProperties(account_name, new)
    return new


def ScpDelete(container_name, dn=None):
    if dn is None:
        dn = win32api.GetComputerObjectName(win32con.NameFullyQualifiedDN)
    logger.debug("Removing connection point '%s' from %s", container_name, dn)

    # Compose the ADSpath and bind to the computer object for the local computer
    comp = adsi.ADsGetObject("LDAP://" + dn, adsi.IID_IDirectoryObject)
    comp.DeleteDSObject("cn=" + container_name)
    logger.info("Deleted service connection point '%s'", container_name)


# This function is described in detail in the MSDN article titled
# "Enabling Service Account to Access SCP Properties"
# From that article:
# The following sample code sets a pair of ACEs on a service connection point
# (SCP) object. The ACEs grant read/write access to the user or computer account
# under which the service instance will be running. Your service installation
# program calls this code to ensure that the service will be allowed to update
# its properties at run time. If you don't set ACEs like these, your service
# will get access-denied errors if it tries to modify the SCP's properties.
#
# The code uses the IADsSecurityDescriptor, IADsAccessControlList, and
# IADsAccessControlEntry interfaces to do the following:
# * Get the SCP object's security descriptor.
# * Set ACEs in the DACL of the security descriptor.
# * Set the security descriptor back on the SCP object.


def AllowAccessToScpProperties(
    accountSAM,  # Service account to allow access.
    scpObject,  # The IADs SCP object.
    schemaIDGUIDs=(  # Attributes to allow write-access to.
        "{28630eb8-41d5-11d1-a9c1-0000f80367c1}",  # serviceDNSName
        "{b7b1311c-b82e-11d0-afee-0000f80367c1}",  # serviceBindingInformation
    ),
):
    # If no service account is specified, service runs under LocalSystem.
    # So allow access to the computer account of the service's host.
    if accountSAM:
        trustee = accountSAM
    else:
        # Get the SAM account name of the computer object for the server.
        trustee = win32api.GetComputerObjectName(win32con.NameSamCompatible)

    # Get the nTSecurityDescriptor attribute
    attribute = "nTSecurityDescriptor"
    sd = getattr(scpObject, attribute)
    acl = sd.DiscretionaryAcl

    for sguid in schemaIDGUIDs:
        ace = Dispatch(adsi.CLSID_AccessControlEntry)

        # Set the properties of the ACE.
        # Allow read and write access to the property.
        ace.AccessMask = ADS_RIGHT_DS_READ_PROP | ADS_RIGHT_DS_WRITE_PROP

        # Set the trustee, which is either the service account or the
        # host computer account.
        ace.Trustee = trustee

        # Set the ACE type.
        ace.AceType = ADS_ACETYPE_ACCESS_ALLOWED_OBJECT

        # Set AceFlags to zero because ACE is not inheritable.
        ace.AceFlags = 0

        # Set Flags to indicate an ACE that protects a specified object.
        ace.Flags = ADS_FLAG_OBJECT_TYPE_PRESENT

        # Set ObjectType to the schemaIDGUID of the attribute.
        ace.ObjectType = sguid

        # Add the ACEs to the DACL.
        acl.AddAce(ace)

    # Write the modified DACL back to the security descriptor.
    sd.DiscretionaryAcl = acl
    # Write the ntSecurityDescriptor property to the property cache.
    setattr(scpObject, attribute, sd)
    # SetInfo updates the SCP object in the directory.
    scpObject.SetInfo()
    logger.info("Set security on object for account '%s'" % (trustee,))


# Service Principal Names functions from the same sample.
# The example calls the DsWriteAccountSpn function, which stores the SPNs in
# Microsoft Active Directory under the servicePrincipalName attribute of the
# account object specified by the serviceAcctDN parameter. The account object
# corresponds to the logon account specified in the CreateService call for this
# service instance. If the logon account is a domain user account,
# serviceAcctDN must be the distinguished name of the account object in
# Active Directory for that user account. If the service's logon account is the
# LocalSystem account, serviceAcctDN must be the distinguished name of the
# computer account object for the host computer on which the service is
# installed. win32api.TranslateNames and win32security.DsCrackNames can
# be used to convert a domain\account format name to a distinguished name.
def SpnRegister(
    serviceAcctDN,  # DN of the service's logon account
    spns,  # List of SPNs to register
    operation,  # Add, replace, or delete SPNs
):
    assert type(spns) not in [str, str] and hasattr(spns, "__iter__"), (
        "spns must be a sequence of strings (got %r)" % spns
    )
    # Bind to a domain controller.
    # Get the domain for the current user.
    samName = win32api.GetUserNameEx(win32api.NameSamCompatible)
    samName = samName.split("\\", 1)[0]

    if not serviceAcctDN:
        # Get the SAM account name of the computer object for the server.
        serviceAcctDN = win32api.GetComputerObjectName(win32con.NameFullyQualifiedDN)
    logger.debug("SpnRegister using DN '%s'", serviceAcctDN)

    # Get the name of a domain controller in that domain.
    info = win32security.DsGetDcName(
        domainName=samName,
        flags=dscon.DS_IS_FLAT_NAME
        | dscon.DS_RETURN_DNS_NAME
        | dscon.DS_DIRECTORY_SERVICE_REQUIRED,
    )
    # Bind to the domain controller.
    handle = win32security.DsBind(info["DomainControllerName"])

    # Write the SPNs to the service account or computer account.
    logger.debug("DsWriteAccountSpn with spns %s")
    win32security.DsWriteAccountSpn(
        handle,  # handle to the directory
        operation,  # Add or remove SPN from account's existing SPNs
        serviceAcctDN,  # DN of service account or computer account
        spns,
    )  # names

    # Unbind the DS in any case (but Python would do it anyway)
    handle.Close()


def UserChangePassword(username_dn, new_password):
    # set the password on the account.
    # Use the distinguished name to bind to the account object.
    accountPath = "LDAP://" + username_dn
    user = adsi.ADsGetObject(accountPath, adsi.IID_IADsUser)

    # Set the password on the account.
    user.SetPassword(new_password)


# functions related to the command-line interface
def log(level, msg, *args):
    if verbose >= level:
        print(msg % args)


class _NoDefault:
    pass


def _get_option(po, opt_name, default=_NoDefault):
    parser, options = po
    ret = getattr(options, opt_name, default)
    if not ret and default is _NoDefault:
        parser.error("The '%s' option must be specified for this operation" % opt_name)
    if not ret:
        ret = default
    return ret


def _option_error(po, why):
    parser = po[0]
    parser.error(why)


def do_ScpCreate(po):
    """Create a Service Connection Point"""
    global g_createdSCP
    scp = ScpCreate(
        _get_option(po, "binding_string"),
        _get_option(po, "service_class"),
        _get_option(po, "account_name_sam", None),
        keywords=_get_option(po, "keywords", None),
    )
    g_createdSCP = scp
    return scp.distinguishedName


def do_ScpDelete(po):
    """Delete a Service Connection Point"""
    sc = _get_option(po, "service_class")
    try:
        ScpDelete(sc)
    except adsi.error as details:
        if details[0] != winerror.ERROR_DS_OBJ_NOT_FOUND:
            raise
        log(2, "ScpDelete ignoring ERROR_DS_OBJ_NOT_FOUND for service-class '%s'", sc)
    return sc


def do_SpnCreate(po):
    """Create a Service Principal Name"""
    # The 'service name' is the dn of our scp.
    if g_createdSCP is None:
        # Could accept an arg to avoid this?
        _option_error(po, "ScpCreate must have been specified before SpnCreate")
    # Create a Service Principal Name"
    spns = win32security.DsGetSpn(
        dscon.DS_SPN_SERVICE,
        _get_option(po, "service_class"),
        g_createdSCP.distinguishedName,
        _get_option(po, "port", 0),
        None,
        None,
    )
    spn = spns[0]
    log(2, "Created SPN: %s", spn)
    global g_createdSPNLast
    g_createdSPNLast = spn
    g_createdSPNs.append(spn)
    return spn


def do_SpnRegister(po):
    """Register a previously created Service Principal Name"""
    if not g_createdSPNLast:
        _option_error(po, "SpnCreate must appear before SpnRegister")

    SpnRegister(
        _get_option(po, "account_name_dn", None),
        (g_createdSPNLast,),
        dscon.DS_SPN_ADD_SPN_OP,
    )
    return g_createdSPNLast


def do_SpnUnregister(po):
    """Unregister a previously created Service Principal Name"""
    if not g_createdSPNLast:
        _option_error(po, "SpnCreate must appear before SpnUnregister")
    SpnRegister(
        _get_option(po, "account_name_dn", None),
        (g_createdSPNLast,),
        dscon.DS_SPN_DELETE_SPN_OP,
    )
    return g_createdSPNLast


def do_UserChangePassword(po):
    """Change the password for a specified user"""
    UserChangePassword(_get_option(po, "account_name_dn"), _get_option(po, "password"))
    return "Password changed OK"


handlers = (
    ("ScpCreate", do_ScpCreate),
    ("ScpDelete", do_ScpDelete),
    ("SpnCreate", do_SpnCreate),
    ("SpnRegister", do_SpnRegister),
    ("SpnUnregister", do_SpnUnregister),
    ("UserChangePassword", do_UserChangePassword),
)


class HelpFormatter(optparse.IndentedHelpFormatter):
    def format_description(self, description):
        return description


def main():
    global verbose
    _handlers_dict = {}

    arg_descs = []
    for arg, func in handlers:
        this_desc = "\n".join(textwrap.wrap(func.__doc__, subsequent_indent=" " * 8))
        arg_descs.append("  %s: %s" % (arg, this_desc))
        _handlers_dict[arg.lower()] = func

    description = __doc__ + "\ncommands:\n" + "\n".join(arg_descs) + "\n"

    parser = optparse.OptionParser(
        usage="%prog [options] command ...",
        description=description,
        formatter=HelpFormatter(),
    )

    parser.add_option(
        "-v",
        action="count",
        dest="verbose",
        default=1,
        help="increase the verbosity of status messages",
    )

    parser.add_option(
        "-q", "--quiet", action="store_true", help="Don't print any status messages"
    )

    parser.add_option(
        "-t",
        "--test",
        action="store_true",
        help="Execute a mini-test suite, providing defaults for most options and args",
    ),

    parser.add_option(
        "",
        "--show-tracebacks",
        action="store_true",
        help="Show the tracebacks for any exceptions",
    )

    parser.add_option("", "--service-class", help="The service class name to use")

    parser.add_option(
        "", "--port", default=0, help="The port number to associate with the SPN"
    )

    parser.add_option(
        "", "--binding-string", help="The binding string to use for SCP creation"
    )

    parser.add_option(
        "", "--account-name", help="The account name to use (default is LocalSystem)"
    )

    parser.add_option("", "--password", help="The password to set.")

    parser.add_option(
        "",
        "--keyword",
        action="append",
        dest="keywords",
        help="""A keyword to add to the SCP.  May be specified
                              multiple times""",
    )

    parser.add_option(
        "",
        "--log-level",
        help="""The log-level to use - may be a number or a logging
                             module constant""",
        default=str(logging.WARNING),
    )

    options, args = parser.parse_args()
    po = (parser, options)
    # fixup misc
    try:
        options.port = int(options.port)
    except (TypeError, ValueError):
        parser.error("--port must be numeric")
    # fixup log-level
    try:
        log_level = int(options.log_level)
    except (TypeError, ValueError):
        try:
            log_level = int(getattr(logging, options.log_level.upper()))
        except (ValueError, TypeError, AttributeError):
            parser.error("Invalid --log-level value")
    try:
        sl = logger.setLevel
        # logger is a real logger
    except AttributeError:
        # logger is logging module
        sl = logging.getLogger().setLevel
    sl(log_level)
    # Check -q/-v
    if options.quiet and options.verbose:
        parser.error("Can't specify --quiet and --verbose")
    if options.quiet:
        options.verbose -= 1
    verbose = options.verbose
    # --test
    if options.test:
        if args:
            parser.error("Can't specify args with --test")

        args = "ScpDelete ScpCreate SpnCreate SpnRegister SpnUnregister ScpDelete"
        log(1, "--test - pretending args are:\n %s", args)
        args = args.split()
        if not options.service_class:
            options.service_class = "PythonScpTest"
            log(2, "--test: --service-class=%s", options.service_class)
        if not options.keywords:
            options.keywords = "Python Powered".split()
            log(2, "--test: --keyword=%s", options.keywords)
        if not options.binding_string:
            options.binding_string = "test binding string"
            log(2, "--test: --binding-string=%s", options.binding_string)

    # check args
    if not args:
        parser.error("No command specified (use --help for valid commands)")
    for arg in args:
        if arg.lower() not in _handlers_dict:
            parser.error("Invalid command '%s' (use --help for valid commands)" % arg)

    # Patch up account-name.
    if options.account_name:
        log(2, "Translating account name '%s'", options.account_name)
        options.account_name_sam = win32security.TranslateName(
            options.account_name, win32api.NameUnknown, win32api.NameSamCompatible
        )
        log(2, "NameSamCompatible is '%s'", options.account_name_sam)
        options.account_name_dn = win32security.TranslateName(
            options.account_name, win32api.NameUnknown, win32api.NameFullyQualifiedDN
        )
        log(2, "NameFullyQualifiedDNis '%s'", options.account_name_dn)

    # do it.
    for arg in args:
        handler = _handlers_dict[arg.lower()]  # already been validated
        if handler is None:
            parser.error("Invalid command '%s'" % arg)
        err_msg = None
        try:
            try:
                log(2, "Executing '%s'...", arg)
                result = handler(po)
                log(1, "%s: %s", arg, result)
            except:
                if options.show_tracebacks:
                    print("--show-tracebacks specified - dumping exception")
                    traceback.print_exc()
                raise
        except adsi.error as xxx_todo_changeme:
            (hr, desc, exc, argerr) = xxx_todo_changeme.args
            if exc:
                extra_desc = exc[2]
            else:
                extra_desc = ""
            err_msg = desc
            if extra_desc:
                err_msg += "\n\t" + extra_desc
        except win32api.error as xxx_todo_changeme1:
            (hr, func, msg) = xxx_todo_changeme1.args
            err_msg = msg
        if err_msg:
            log(1, "Command '%s' failed: %s", arg, err_msg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("*** Interrupted")
