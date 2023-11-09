"""A test runner for pywin32"""
import os
import site
import subprocess
import sys

# locate the dirs based on where this script is - it may be either in the
# source tree, or in an installed Python 'Scripts' tree.
this_dir = os.path.dirname(__file__)
site_packages = [
    site.getusersitepackages(),
] + site.getsitepackages()

failures = []


# Run a test using subprocess and wait for the result.
# If we get an returncode != 0, we know that there was an error, but we don't
# abort immediately - we run as many tests as we can.
def run_test(script, cmdline_extras):
    dirname, scriptname = os.path.split(script)
    # some tests prefer to be run from their directory.
    cmd = [sys.executable, "-u", scriptname] + cmdline_extras
    print("--- Running '%s' ---" % script)
    sys.stdout.flush()
    result = subprocess.run(cmd, check=False, cwd=dirname)
    print("*** Test script '%s' exited with %s" % (script, result.returncode))
    sys.stdout.flush()
    if result.returncode:
        failures.append(script)


def find_and_run(possible_locations, extras):
    for maybe in possible_locations:
        if os.path.isfile(maybe):
            run_test(maybe, extras)
            break
    else:
        raise RuntimeError(
            "Failed to locate a test script in one of %s" % possible_locations
        )


def main():
    import argparse

    code_directories = [this_dir] + site_packages

    parser = argparse.ArgumentParser(
        description="A script to trigger tests in all subprojects of PyWin32."
    )
    parser.add_argument(
        "-no-user-interaction",
        default=False,
        action="store_true",
        help="(This is now the default - use `-user-interaction` to include them)",
    )

    parser.add_argument(
        "-user-interaction",
        action="store_true",
        help="Include tests which require user interaction",
    )

    parser.add_argument(
        "-skip-adodbapi",
        default=False,
        action="store_true",
        help="Skip the adodbapi tests; useful for CI where there's no provider",
    )

    args, remains = parser.parse_known_args()

    # win32, win32ui / Pythonwin

    extras = []
    if args.user_interaction:
        extras += ["-user-interaction"]
    extras.extend(remains)
    scripts = [
        "win32/test/testall.py",
        "Pythonwin/pywin/test/all.py",
    ]
    for script in scripts:
        maybes = [os.path.join(directory, script) for directory in code_directories]
        find_and_run(maybes, extras)

    # win32com
    maybes = [
        os.path.join(directory, "win32com", "test", "testall.py")
        for directory in [
            os.path.join(this_dir, "com"),
        ]
        + site_packages
    ]
    extras = remains + ["1"]  # only run "level 1" tests in CI
    find_and_run(maybes, extras)

    # adodbapi
    if not args.skip_adodbapi:
        maybes = [
            os.path.join(directory, "adodbapi", "test", "adodbapitest.py")
            for directory in code_directories
        ]
        find_and_run(maybes, remains)
        # This script has a hard-coded sql server name in it, (and markh typically
        # doesn't have a different server to test on) but there is now supposed to be a server out there on the Internet
        # just to run these tests, so try it...
        maybes = [
            os.path.join(directory, "adodbapi", "test", "test_adodbapi_dbapi20.py")
            for directory in code_directories
        ]
        find_and_run(maybes, remains)

    if failures:
        print("The following scripts failed")
        for failure in failures:
            print(">", failure)
        sys.exit(1)
    print("All tests passed \\o/")


if __name__ == "__main__":
    main()
