""" a clumsy attempt at a macro language to let the programmer execute code on the server (ex: determine 64bit)"""
from . import is64bit as is64bit


def macro_call(macro_name, args, kwargs):
    """allow the programmer to perform limited processing on the server by passing macro names and args

    :new_key - the key name the macro will create
    :args[0] - macro name
    :args[1:] - any arguments
    :code - the value of the keyword item
    :kwargs - the connection keyword dictionary. ??key has been removed
    --> the value to put in for kwargs['name'] = value
    """
    if isinstance(args, (str, str)):
        args = [
            args
        ]  # the user forgot to pass a sequence, so make a string into args[0]
    new_key = args[0]
    try:
        if macro_name == "is64bit":
            if is64bit.Python():  # if on 64 bit Python
                return new_key, args[1]  # return first argument
            else:
                try:
                    return new_key, args[2]  # else return second argument (if defined)
                except IndexError:
                    return new_key, ""  # else return blank

        elif (
            macro_name == "getuser"
        ):  # get the name of the user the server is logged in under
            if not new_key in kwargs:
                import getpass

                return new_key, getpass.getuser()

        elif macro_name == "getnode":  # get the name of the computer running the server
            import platform

            try:
                return new_key, args[1] % platform.node()
            except IndexError:
                return new_key, platform.node()

        elif macro_name == "getenv":  # expand the server's environment variable args[1]
            try:
                dflt = args[2]  # if not found, default from args[2]
            except IndexError:  # or blank
                dflt = ""
            return new_key, os.environ.get(args[1], dflt)

        elif macro_name == "auto_security":
            if (
                not "user" in kwargs or not kwargs["user"]
            ):  # missing, blank, or Null username
                return new_key, "Integrated Security=SSPI"
            return new_key, "User ID=%(user)s; Password=%(password)s" % kwargs

        elif (
            macro_name == "find_temp_test_path"
        ):  # helper function for testing ado operation -- undocumented
            import os
            import tempfile

            return new_key, os.path.join(
                tempfile.gettempdir(), "adodbapi_test", args[1]
            )

        raise ValueError("Unknown connect string macro=%s" % macro_name)
    except:
        raise ValueError("Error in macro processing %s %s" % (macro_name, repr(args)))


def process(
    args, kwargs, expand_macros=False
):  # --> connection string with keyword arguments processed.
    """attempts to inject arguments into a connection string using Python "%" operator for strings

    co: adodbapi connection object
    args: positional parameters from the .connect() call
    kvargs: keyword arguments from the .connect() call
    """
    try:
        dsn = args[0]
    except IndexError:
        dsn = None
    if isinstance(
        dsn, dict
    ):  # as a convenience the first argument may be django settings
        kwargs.update(dsn)
    elif (
        dsn
    ):  # the connection string is passed to the connection as part of the keyword dictionary
        kwargs["connection_string"] = dsn
    try:
        a1 = args[1]
    except IndexError:
        a1 = None
    # historically, the second positional argument might be a timeout value
    if isinstance(a1, int):
        kwargs["timeout"] = a1
    # if the second positional argument is a string, then it is user
    elif isinstance(a1, str):
        kwargs["user"] = a1
    # if the second positional argument is a dictionary, use it as keyword arguments, too
    elif isinstance(a1, dict):
        kwargs.update(a1)
    try:
        kwargs["password"] = args[2]  # the third positional argument is password
        kwargs["host"] = args[3]  # the fourth positional argument is host name
        kwargs["database"] = args[4]  # the fifth positional argument is database name
    except IndexError:
        pass

    # make sure connection string is defined somehow
    if not "connection_string" in kwargs:
        try:  # perhaps 'dsn' was defined
            kwargs["connection_string"] = kwargs["dsn"]
        except KeyError:
            try:  # as a last effort, use the "host" keyword
                kwargs["connection_string"] = kwargs["host"]
            except KeyError:
                raise TypeError("Must define 'connection_string' for ado connections")
    if expand_macros:
        for kwarg in list(kwargs.keys()):
            if kwarg.startswith("macro_"):  # If a key defines a macro
                macro_name = kwarg[6:]  # name without the "macro_"
                macro_code = kwargs.pop(
                    kwarg
                )  # we remove the macro_key and get the code to execute
                new_key, rslt = macro_call(
                    macro_name, macro_code, kwargs
                )  # run the code in the local context
                kwargs[new_key] = rslt  # put the result back in the keywords dict
    # special processing for PyRO IPv6 host address
    try:
        s = kwargs["proxy_host"]
        if ":" in s:  # it is an IPv6 address
            if s[0] != "[":  # is not surrounded by brackets
                kwargs["proxy_host"] = s.join(("[", "]"))  # put it in brackets
    except KeyError:
        pass
    return kwargs
