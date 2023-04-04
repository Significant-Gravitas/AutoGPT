remote = False  # automatic testing of remote access has been removed here


def try_connection(verbose, *args, **kwargs):
    import adodbapi

    dbconnect = adodbapi.connect
    try:
        s = dbconnect(*args, **kwargs)  # connect to server
        if verbose:
            print("Connected to:", s.connection_string)
            print("which has tables:", s.get_table_names())
        s.close()  # thanks, it worked, goodbye
    except adodbapi.DatabaseError as inst:
        print(inst.args[0])  # should be the error message
        print("***Failed getting connection using=", repr(args), repr(kwargs))
        return False, (args, kwargs), None

    print("  (successful)")

    return True, (args, kwargs, remote), dbconnect


def try_operation_with_expected_exception(
    expected_exception_list, some_function, *args, **kwargs
):
    try:
        some_function(*args, **kwargs)
    except expected_exception_list as e:
        return True, e
    except:
        raise  # an exception other than the expected occurred
    return False, "The expected exception did not occur"
