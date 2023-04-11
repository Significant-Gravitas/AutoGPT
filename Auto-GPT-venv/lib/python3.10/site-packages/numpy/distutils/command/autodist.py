"""This module implements additional tests ala autoconf which can be useful.

"""
import textwrap

# We put them here since they could be easily reused outside numpy.distutils

def check_inline(cmd):
    """Return the inline identifier (may be empty)."""
    cmd._check_compiler()
    body = textwrap.dedent("""
        #ifndef __cplusplus
        static %(inline)s int static_func (void)
        {
            return 0;
        }
        %(inline)s int nostatic_func (void)
        {
            return 0;
        }
        #endif""")

    for kw in ['inline', '__inline__', '__inline']:
        st = cmd.try_compile(body % {'inline': kw}, None, None)
        if st:
            return kw

    return ''


def check_restrict(cmd):
    """Return the restrict identifier (may be empty)."""
    cmd._check_compiler()
    body = textwrap.dedent("""
        static int static_func (char * %(restrict)s a)
        {
            return 0;
        }
        """)

    for kw in ['restrict', '__restrict__', '__restrict']:
        st = cmd.try_compile(body % {'restrict': kw}, None, None)
        if st:
            return kw

    return ''


def check_compiler_gcc(cmd):
    """Check if the compiler is GCC."""

    cmd._check_compiler()
    body = textwrap.dedent("""
        int
        main()
        {
        #if (! defined __GNUC__)
        #error gcc required
        #endif
            return 0;
        }
        """)
    return cmd.try_compile(body, None, None)


def check_gcc_version_at_least(cmd, major, minor=0, patchlevel=0):
    """
    Check that the gcc version is at least the specified version."""

    cmd._check_compiler()
    version = '.'.join([str(major), str(minor), str(patchlevel)])
    body = textwrap.dedent("""
        int
        main()
        {
        #if (! defined __GNUC__) || (__GNUC__ < %(major)d) || \\
                (__GNUC_MINOR__ < %(minor)d) || \\
                (__GNUC_PATCHLEVEL__ < %(patchlevel)d)
        #error gcc >= %(version)s required
        #endif
            return 0;
        }
        """)
    kw = {'version': version, 'major': major, 'minor': minor,
          'patchlevel': patchlevel}

    return cmd.try_compile(body % kw, None, None)


def check_gcc_function_attribute(cmd, attribute, name):
    """Return True if the given function attribute is supported."""
    cmd._check_compiler()
    body = textwrap.dedent("""
        #pragma GCC diagnostic error "-Wattributes"
        #pragma clang diagnostic error "-Wattributes"

        int %s %s(void* unused)
        {
            return 0;
        }

        int
        main()
        {
            return 0;
        }
        """) % (attribute, name)
    return cmd.try_compile(body, None, None) != 0


def check_gcc_function_attribute_with_intrinsics(cmd, attribute, name, code,
                                                include):
    """Return True if the given function attribute is supported with
    intrinsics."""
    cmd._check_compiler()
    body = textwrap.dedent("""
        #include<%s>
        int %s %s(void)
        {
            %s;
            return 0;
        }

        int
        main()
        {
            return 0;
        }
        """) % (include, attribute, name, code)
    return cmd.try_compile(body, None, None) != 0


def check_gcc_variable_attribute(cmd, attribute):
    """Return True if the given variable attribute is supported."""
    cmd._check_compiler()
    body = textwrap.dedent("""
        #pragma GCC diagnostic error "-Wattributes"
        #pragma clang diagnostic error "-Wattributes"

        int %s foo;

        int
        main()
        {
            return 0;
        }
        """) % (attribute, )
    return cmd.try_compile(body, None, None) != 0
