import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple

import pytest

from . import util
from numpy.f2py.f2py2e import main as f2pycli

#########################
# CLI utils and classes #
#########################

PPaths = namedtuple("PPaths", "finp, f90inp, pyf, wrap77, wrap90, cmodf")


def get_io_paths(fname_inp, mname="untitled"):
    """Takes in a temporary file for testing and returns the expected output and input paths

    Here expected output is essentially one of any of the possible generated
    files.

    ..note::

         Since this does not actually run f2py, none of these are guaranteed to
         exist, and module names are typically incorrect

    Parameters
    ----------
    fname_inp : str
                The input filename
    mname : str, optional
                The name of the module, untitled by default

    Returns
    -------
    genp : NamedTuple PPaths
            The possible paths which are generated, not all of which exist
    """
    bpath = Path(fname_inp)
    return PPaths(
        finp=bpath.with_suffix(".f"),
        f90inp=bpath.with_suffix(".f90"),
        pyf=bpath.with_suffix(".pyf"),
        wrap77=bpath.with_name(f"{mname}-f2pywrappers.f"),
        wrap90=bpath.with_name(f"{mname}-f2pywrappers2.f90"),
        cmodf=bpath.with_name(f"{mname}module.c"),
    )


##############
# CLI Fixtures and Tests #
#############


@pytest.fixture(scope="session")
def hello_world_f90(tmpdir_factory):
    """Generates a single f90 file for testing"""
    fdat = util.getpath("tests", "src", "cli", "hiworld.f90").read_text()
    fn = tmpdir_factory.getbasetemp() / "hello.f90"
    fn.write_text(fdat, encoding="ascii")
    return fn


@pytest.fixture(scope="session")
def hello_world_f77(tmpdir_factory):
    """Generates a single f77 file for testing"""
    fdat = util.getpath("tests", "src", "cli", "hi77.f").read_text()
    fn = tmpdir_factory.getbasetemp() / "hello.f"
    fn.write_text(fdat, encoding="ascii")
    return fn


@pytest.fixture(scope="session")
def retreal_f77(tmpdir_factory):
    """Generates a single f77 file for testing"""
    fdat = util.getpath("tests", "src", "return_real", "foo77.f").read_text()
    fn = tmpdir_factory.getbasetemp() / "foo.f"
    fn.write_text(fdat, encoding="ascii")
    return fn

@pytest.fixture(scope="session")
def f2cmap_f90(tmpdir_factory):
    """Generates a single f90 file for testing"""
    fdat = util.getpath("tests", "src", "f2cmap", "isoFortranEnvMap.f90").read_text()
    f2cmap = util.getpath("tests", "src", "f2cmap", ".f2py_f2cmap").read_text()
    fn = tmpdir_factory.getbasetemp() / "f2cmap.f90"
    fmap = tmpdir_factory.getbasetemp() / "mapfile"
    fn.write_text(fdat, encoding="ascii")
    fmap.write_text(f2cmap, encoding="ascii")
    return fn


def test_gen_pyf(capfd, hello_world_f90, monkeypatch):
    """Ensures that a signature file is generated via the CLI
    CLI :: -h
    """
    ipath = Path(hello_world_f90)
    opath = Path(hello_world_f90).stem + ".pyf"
    monkeypatch.setattr(sys, "argv", f'f2py -h {opath} {ipath}'.split())

    with util.switchdir(ipath.parent):
        f2pycli()  # Generate wrappers
        out, _ = capfd.readouterr()
        assert "Saving signatures to file" in out
        assert Path(f'{opath}').exists()


def test_gen_pyf_stdout(capfd, hello_world_f90, monkeypatch):
    """Ensures that a signature file can be dumped to stdout
    CLI :: -h
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -h stdout {ipath}'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Saving signatures to file" in out
        assert "function hi() ! in " in out


def test_gen_pyf_no_overwrite(capfd, hello_world_f90, monkeypatch):
    """Ensures that the CLI refuses to overwrite signature files
    CLI :: -h without --overwrite-signature
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -h faker.pyf {ipath}'.split())

    with util.switchdir(ipath.parent):
        Path("faker.pyf").write_text("Fake news", encoding="ascii")
        with pytest.raises(SystemExit):
            f2pycli()  # Refuse to overwrite
            _, err = capfd.readouterr()
            assert "Use --overwrite-signature to overwrite" in err


@pytest.mark.xfail
def test_f2py_skip(capfd, retreal_f77, monkeypatch):
    """Tests that functions can be skipped
    CLI :: skip:
    """
    foutl = get_io_paths(retreal_f77, mname="test")
    ipath = foutl.finp
    toskip = "t0 t4 t8 sd s8 s4"
    remaining = "td s0"
    monkeypatch.setattr(
        sys, "argv",
        f'f2py {ipath} -m test skip: {toskip}'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, err = capfd.readouterr()
        for skey in toskip.split():
            assert (
                f'buildmodule: Could not found the body of interfaced routine "{skey}". Skipping.'
                in err)
        for rkey in remaining.split():
            assert f'Constructing wrapper function "{rkey}"' in out


def test_f2py_only(capfd, retreal_f77, monkeypatch):
    """Test that functions can be kept by only:
    CLI :: only:
    """
    foutl = get_io_paths(retreal_f77, mname="test")
    ipath = foutl.finp
    toskip = "t0 t4 t8 sd s8 s4"
    tokeep = "td s0"
    monkeypatch.setattr(
        sys, "argv",
        f'f2py {ipath} -m test only: {tokeep}'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, err = capfd.readouterr()
        for skey in toskip.split():
            assert (
                f'buildmodule: Could not find the body of interfaced routine "{skey}". Skipping.'
                in err)
        for rkey in tokeep.split():
            assert f'Constructing wrapper function "{rkey}"' in out


def test_file_processing_switch(capfd, hello_world_f90, retreal_f77,
                                monkeypatch):
    """Tests that it is possible to return to file processing mode
    CLI :: :
    BUG: numpy-gh #20520
    """
    foutl = get_io_paths(retreal_f77, mname="test")
    ipath = foutl.finp
    toskip = "t0 t4 t8 sd s8 s4"
    ipath2 = Path(hello_world_f90)
    tokeep = "td s0 hi"  # hi is in ipath2
    mname = "blah"
    monkeypatch.setattr(
        sys,
        "argv",
        f'f2py {ipath} -m {mname} only: {tokeep} : {ipath2}'.split(
        ),
    )

    with util.switchdir(ipath.parent):
        f2pycli()
        out, err = capfd.readouterr()
        for skey in toskip.split():
            assert (
                f'buildmodule: Could not find the body of interfaced routine "{skey}". Skipping.'
                in err)
        for rkey in tokeep.split():
            assert f'Constructing wrapper function "{rkey}"' in out


def test_mod_gen_f77(capfd, hello_world_f90, monkeypatch):
    """Checks the generation of files based on a module name
    CLI :: -m
    """
    MNAME = "hi"
    foutl = get_io_paths(hello_world_f90, mname=MNAME)
    ipath = foutl.f90inp
    monkeypatch.setattr(sys, "argv", f'f2py {ipath} -m {MNAME}'.split())
    with util.switchdir(ipath.parent):
        f2pycli()

    # Always generate C module
    assert Path.exists(foutl.cmodf)
    # File contains a function, check for F77 wrappers
    assert Path.exists(foutl.wrap77)


def test_lower_cmod(capfd, hello_world_f77, monkeypatch):
    """Lowers cases by flag or when -h is present

    CLI :: --[no-]lower
    """
    foutl = get_io_paths(hello_world_f77, mname="test")
    ipath = foutl.finp
    capshi = re.compile(r"HI\(\)")
    capslo = re.compile(r"hi\(\)")
    # Case I: --lower is passed
    monkeypatch.setattr(sys, "argv", f'f2py {ipath} -m test --lower'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert capslo.search(out) is not None
        assert capshi.search(out) is None
    # Case II: --no-lower is passed
    monkeypatch.setattr(sys, "argv",
                        f'f2py {ipath} -m test --no-lower'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert capslo.search(out) is None
        assert capshi.search(out) is not None


def test_lower_sig(capfd, hello_world_f77, monkeypatch):
    """Lowers cases in signature files by flag or when -h is present

    CLI :: --[no-]lower -h
    """
    foutl = get_io_paths(hello_world_f77, mname="test")
    ipath = foutl.finp
    # Signature files
    capshi = re.compile(r"Block: HI")
    capslo = re.compile(r"Block: hi")
    # Case I: --lower is implied by -h
    # TODO: Clean up to prevent passing --overwrite-signature
    monkeypatch.setattr(
        sys,
        "argv",
        f'f2py {ipath} -h {foutl.pyf} -m test --overwrite-signature'.split(),
    )

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert capslo.search(out) is not None
        assert capshi.search(out) is None

    # Case II: --no-lower overrides -h
    monkeypatch.setattr(
        sys,
        "argv",
        f'f2py {ipath} -h {foutl.pyf} -m test --overwrite-signature --no-lower'
        .split(),
    )

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert capslo.search(out) is None
        assert capshi.search(out) is not None


def test_build_dir(capfd, hello_world_f90, monkeypatch):
    """Ensures that the build directory can be specified

    CLI :: --build-dir
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    odir = "tttmp"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --build-dir {odir}'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert f"Wrote C/API module \"{mname}\"" in out


def test_overwrite(capfd, hello_world_f90, monkeypatch):
    """Ensures that the build directory can be specified

    CLI :: --overwrite-signature
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(
        sys, "argv",
        f'f2py -h faker.pyf {ipath} --overwrite-signature'.split())

    with util.switchdir(ipath.parent):
        Path("faker.pyf").write_text("Fake news", encoding="ascii")
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Saving signatures to file" in out


def test_latexdoc(capfd, hello_world_f90, monkeypatch):
    """Ensures that TeX documentation is written out

    CLI :: --latex-doc
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --latex-doc'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Documentation is saved to file" in out
        with Path(f"{mname}module.tex").open() as otex:
            assert "\\documentclass" in otex.read()


def test_nolatexdoc(capfd, hello_world_f90, monkeypatch):
    """Ensures that TeX documentation is written out

    CLI :: --no-latex-doc
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --no-latex-doc'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Documentation is saved to file" not in out


def test_shortlatex(capfd, hello_world_f90, monkeypatch):
    """Ensures that truncated documentation is written out

    TODO: Test to ensure this has no effect without --latex-doc
    CLI :: --latex-doc --short-latex
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(
        sys,
        "argv",
        f'f2py -m {mname} {ipath} --latex-doc --short-latex'.split(),
    )

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Documentation is saved to file" in out
        with Path(f"./{mname}module.tex").open() as otex:
            assert "\\documentclass" not in otex.read()


def test_restdoc(capfd, hello_world_f90, monkeypatch):
    """Ensures that RsT documentation is written out

    CLI :: --rest-doc
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --rest-doc'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "ReST Documentation is saved to file" in out
        with Path(f"./{mname}module.rest").open() as orst:
            assert r".. -*- rest -*-" in orst.read()


def test_norestexdoc(capfd, hello_world_f90, monkeypatch):
    """Ensures that TeX documentation is written out

    CLI :: --no-rest-doc
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --no-rest-doc'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "ReST Documentation is saved to file" not in out


def test_debugcapi(capfd, hello_world_f90, monkeypatch):
    """Ensures that debugging wrappers are written

    CLI :: --debug-capi
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --debug-capi'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        with Path(f"./{mname}module.c").open() as ocmod:
            assert r"#define DEBUGCFUNCS" in ocmod.read()


@pytest.mark.xfail(reason="Consistently fails on CI.")
def test_debugcapi_bld(hello_world_f90, monkeypatch):
    """Ensures that debugging wrappers work

    CLI :: --debug-capi -c
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} -c --debug-capi'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        cmd_run = shlex.split("python3 -c \"import blah; blah.hi()\"")
        rout = subprocess.run(cmd_run, capture_output=True, encoding='UTF-8')
        eout = ' Hello World\n'
        eerr = textwrap.dedent("""\
debug-capi:Python C/API function blah.hi()
debug-capi:float hi=:output,hidden,scalar
debug-capi:hi=0
debug-capi:Fortran subroutine `f2pywraphi(&hi)'
debug-capi:hi=0
debug-capi:Building return value.
debug-capi:Python C/API function blah.hi: successful.
debug-capi:Freeing memory.
        """)
        assert rout.stdout == eout
        assert rout.stderr == eerr


def test_wrapfunc_def(capfd, hello_world_f90, monkeypatch):
    """Ensures that fortran subroutine wrappers for F77 are included by default

    CLI :: --[no]-wrap-functions
    """
    # Implied
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv", f'f2py -m {mname} {ipath}'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
    out, _ = capfd.readouterr()
    assert r"Fortran 77 wrappers are saved to" in out

    # Explicit
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --wrap-functions'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert r"Fortran 77 wrappers are saved to" in out


def test_nowrapfunc(capfd, hello_world_f90, monkeypatch):
    """Ensures that fortran subroutine wrappers for F77 can be disabled

    CLI :: --no-wrap-functions
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --no-wrap-functions'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert r"Fortran 77 wrappers are saved to" not in out


def test_inclheader(capfd, hello_world_f90, monkeypatch):
    """Add to the include directories

    CLI :: -include
    TODO: Document this in the help string
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(
        sys,
        "argv",
        f'f2py -m {mname} {ipath} -include<stdbool.h> -include<stdio.h> '.
        split(),
    )

    with util.switchdir(ipath.parent):
        f2pycli()
        with Path(f"./{mname}module.c").open() as ocmod:
            ocmr = ocmod.read()
            assert "#include <stdbool.h>" in ocmr
            assert "#include <stdio.h>" in ocmr


def test_inclpath():
    """Add to the include directories

    CLI :: --include-paths
    """
    # TODO: populate
    pass


def test_hlink():
    """Add to the include directories

    CLI :: --help-link
    """
    # TODO: populate
    pass


def test_f2cmap(capfd, f2cmap_f90, monkeypatch):
    """Check that Fortran-to-Python KIND specs can be passed

    CLI :: --f2cmap
    """
    ipath = Path(f2cmap_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -m blah {ipath} --f2cmap mapfile'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Reading f2cmap from 'mapfile' ..." in out
        assert "Mapping \"real(kind=real32)\" to \"float\"" in out
        assert "Mapping \"real(kind=real64)\" to \"double\"" in out
        assert "Mapping \"integer(kind=int64)\" to \"long_long\"" in out
        assert "Successfully applied user defined f2cmap changes" in out


def test_quiet(capfd, hello_world_f90, monkeypatch):
    """Reduce verbosity

    CLI :: --quiet
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -m blah {ipath} --quiet'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert len(out) == 0


def test_verbose(capfd, hello_world_f90, monkeypatch):
    """Increase verbosity

    CLI :: --verbose
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -m blah {ipath} --verbose'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "analyzeline" in out


def test_version(capfd, monkeypatch):
    """Ensure version

    CLI :: -v
    """
    monkeypatch.setattr(sys, "argv", 'f2py -v'.split())
    # TODO: f2py2e should not call sys.exit() after printing the version
    with pytest.raises(SystemExit):
        f2pycli()
        out, _ = capfd.readouterr()
        import numpy as np
        assert np.__version__ == out.strip()


@pytest.mark.xfail(reason="Consistently fails on CI.")
def test_npdistop(hello_world_f90, monkeypatch):
    """
    CLI :: -c
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -m blah {ipath} -c'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        cmd_run = shlex.split("python -c \"import blah; blah.hi()\"")
        rout = subprocess.run(cmd_run, capture_output=True, encoding='UTF-8')
        eout = ' Hello World\n'
        assert rout.stdout == eout


# Numpy distutils flags
# TODO: These should be tested separately


def test_npd_fcompiler():
    """
    CLI :: -c --fcompiler
    """
    # TODO: populate
    pass


def test_npd_compiler():
    """
    CLI :: -c --compiler
    """
    # TODO: populate
    pass


def test_npd_help_fcompiler():
    """
    CLI :: -c --help-fcompiler
    """
    # TODO: populate
    pass


def test_npd_f77exec():
    """
    CLI :: -c --f77exec
    """
    # TODO: populate
    pass


def test_npd_f90exec():
    """
    CLI :: -c --f90exec
    """
    # TODO: populate
    pass


def test_npd_f77flags():
    """
    CLI :: -c --f77flags
    """
    # TODO: populate
    pass


def test_npd_f90flags():
    """
    CLI :: -c --f90flags
    """
    # TODO: populate
    pass


def test_npd_opt():
    """
    CLI :: -c --opt
    """
    # TODO: populate
    pass


def test_npd_arch():
    """
    CLI :: -c --arch
    """
    # TODO: populate
    pass


def test_npd_noopt():
    """
    CLI :: -c --noopt
    """
    # TODO: populate
    pass


def test_npd_noarch():
    """
    CLI :: -c --noarch
    """
    # TODO: populate
    pass


def test_npd_debug():
    """
    CLI :: -c --debug
    """
    # TODO: populate
    pass


def test_npd_link_auto():
    """
    CLI :: -c --link-<resource>
    """
    # TODO: populate
    pass


def test_npd_lib():
    """
    CLI :: -c -L/path/to/lib/ -l<libname>
    """
    # TODO: populate
    pass


def test_npd_define():
    """
    CLI :: -D<define>
    """
    # TODO: populate
    pass


def test_npd_undefine():
    """
    CLI :: -U<name>
    """
    # TODO: populate
    pass


def test_npd_incl():
    """
    CLI :: -I/path/to/include/
    """
    # TODO: populate
    pass


def test_npd_linker():
    """
    CLI :: <filename>.o <filename>.so <filename>.a
    """
    # TODO: populate
    pass
