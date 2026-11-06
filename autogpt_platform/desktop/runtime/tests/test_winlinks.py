"""A path RabbitMQ's scripts can take, on a Windows volume without 8.3 short
names: a junction, in a place another user of the machine cannot redirect.

The choice of place and of what to trust is plain logic and runs everywhere;
the junctions and the closed directory are made for real on Windows.
"""

import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from autogpt_desktop import install, layout, rabbitmq, winlinks

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="junctions are a Windows thing"
)

USER = "S-1-5-21-1-2-3-1001"


def no_short_names(path: str) -> str:
    return path


def test_what_the_batch_scripts_can_take():
    assert winlinks.plain(r"C:\ProgramData\AutoGPT-S-1-5-21\0123abcd")
    assert not winlinks.plain(r"C:\Users\John Smith\AppData\Local")
    assert not winlinks.plain("C:\\Users\\J\u00fcrgen\\AppData\\Local")
    assert not winlinks.plain("C:\\a\tb")


def test_the_users_own_profile_comes_first_and_program_data_second():
    environ = {
        "LOCALAPPDATA": r"C:\Users\runner\AppData\Local",
        "PROGRAMDATA": r"C:\ProgramData",
    }
    found = winlinks.roots("AutoGPT", environ, USER, no_short_names)
    assert found == [
        winlinks.Root(
            Path(r"C:\Users\runner\AppData\Local") / "AutoGPT" / "links", guarded=False
        ),
        winlinks.Root(Path(r"C:\ProgramData") / f"AutoGPT-{USER}", guarded=True),
    ]


def test_a_profile_with_a_space_in_it_is_used_by_its_short_name_or_not_at_all():
    environ = {
        "LOCALAPPDATA": r"C:\Users\John Smith\AppData\Local",
        "PROGRAMDATA": r"C:\ProgramData",
    }

    shortened = winlinks.roots(
        "AutoGPT", environ, USER, lambda path: r"C:\Users\JOHNSM~1\AppData\Local"
    )
    assert shortened[0] == winlinks.Root(
        Path(r"C:\Users\JOHNSM~1\AppData\Local") / "AutoGPT" / "links", guarded=False
    )

    without = winlinks.roots("AutoGPT", environ, USER, no_short_names)
    assert without == [
        winlinks.Root(Path(r"C:\ProgramData") / f"AutoGPT-{USER}", guarded=True)
    ]


def test_each_install_and_each_user_has_a_folder_of_their_own():
    environ = {
        "LOCALAPPDATA": r"C:\Users\runner\AppData\Local",
        "PROGRAMDATA": r"C:\ProgramData",
    }
    normal = winlinks.roots("AutoGPT", environ, USER, no_short_names)
    variant = winlinks.roots("AutoGPT-voice", environ, USER, no_short_names)
    other_user = winlinks.roots(
        "AutoGPT", environ, "S-1-5-21-1-2-3-1002", no_short_names
    )
    assert not {root.path for root in normal} & {root.path for root in variant}
    assert normal[1].path != other_user[1].path


def test_nowhere_to_put_a_junction_is_an_empty_answer_not_an_error():
    environ = {
        "LOCALAPPDATA": r"D:\My Profile\Local",
        "PROGRAMDATA": r"D:\Program Data",
    }
    assert winlinks.roots("AutoGPT", environ, USER, no_short_names) == []
    assert winlinks.roots("AutoGPT", {}, USER, no_short_names) == []


def test_only_the_user_administrators_and_the_system_are_trusted():
    assert winlinks.trusted(USER, USER)
    assert winlinks.trusted(winlinks.ADMINISTRATORS, USER)  # an elevated run made it
    assert winlinks.trusted(winlinks.SYSTEM, USER)
    assert not winlinks.trusted(OTHER, USER)


OTHER = "S-1-5-21-1-2-3-1002"
USERS = "S-1-5-32-545"
DIRECTORY = stat.FILE_ATTRIBUTE_DIRECTORY
FULL, READ, MODIFY = 0x1F01FF, 0x1200A9, 0x1301BF


def allowed(sid: str, mask: int = FULL) -> winlinks.Ace:
    return winlinks.Ace(winlinks.ACCESS_ALLOWED, mask, sid)


CLOSED = (allowed(USER), allowed(winlinks.SYSTEM), allowed(winlinks.ADMINISTRATORS))


def test_a_closed_folder_of_the_users_or_an_administrators_may_hold_the_junctions():
    assert winlinks.refusal(winlinks.Seen(DIRECTORY, USER, CLOSED), USER) is None
    assert (
        winlinks.refusal(
            winlinks.Seen(DIRECTORY, winlinks.ADMINISTRATORS, CLOSED), USER
        )
        is None
    )
    # Reading is no way to redirect anything, and a refusal gives nothing.
    read_by_all = (
        *CLOSED,
        allowed(USERS, READ),
        winlinks.Ace(winlinks.ACCESS_DENIED, FULL, OTHER),
    )
    assert winlinks.refusal(winlinks.Seen(DIRECTORY, USER, read_by_all), USER) is None


def test_a_guarded_folder_someone_else_made_is_refused():
    theirs = winlinks.Seen(DIRECTORY, OTHER, CLOSED)
    assert "belongs to another user" in str(winlinks.refusal(theirs, USER))


def test_a_link_where_the_folder_should_be_is_refused_whoever_owns_what_it_leads_to():
    """What the handle is open on is the link itself, so the owner read from
    it is the link's: one that points at a folder of the system's does not
    borrow that folder's owner."""
    link = winlinks.Seen(
        DIRECTORY | stat.FILE_ATTRIBUTE_REPARSE_POINT, winlinks.SYSTEM, CLOSED
    )
    assert "is a link" in str(winlinks.refusal(link, USER))
    assert "not a folder" in str(winlinks.refusal(winlinks.Seen(0, USER, CLOSED), USER))


@pytest.mark.parametrize(
    "right",
    [
        0x2,
        0x4,
        0x40,
        0x100,
        0x10000,
        0x40000,
        0x80000,
        0x10000000,
        0x40000000,
        MODIFY,
        FULL,
    ],
)
def test_a_folder_of_the_users_that_someone_else_may_change_is_refused(right: int):
    """Owned by the right account is not enough: whoever can add to the
    folder, delete from it or rewrite its access list can swap a junction."""
    open_to_users = winlinks.Seen(
        DIRECTORY, USER, (*CLOSED, allowed(USERS, READ | right))
    )
    assert f"changed by someone else ({USERS})" in str(
        winlinks.refusal(open_to_users, USER)
    )


def test_a_folder_with_no_access_list_or_an_entry_that_cannot_be_read_is_refused():
    assert "no access list" in str(
        winlinks.refusal(winlinks.Seen(DIRECTORY, USER, None), USER)
    )
    callback_allowed = 9
    odd = (*CLOSED, winlinks.Ace(callback_allowed, FULL, ""))
    assert "cannot be judged" in str(
        winlinks.refusal(winlinks.Seen(DIRECTORY, USER, odd), USER)
    )


@pytest.fixture
def directory_api(monkeypatch):
    import ctypes

    kernel = Mock()
    advapi = Mock()
    descriptor_address = 1234

    def convert(sddl, revision, descriptor, size):
        descriptor._obj.value = descriptor_address
        return True

    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.side_effect = convert
    kernel.CreateDirectoryW.return_value = True
    monkeypatch.setattr(winlinks, "_kernel32", lambda: kernel)
    monkeypatch.setattr(winlinks, "_advapi32", lambda: advapi)
    error = OSError("mock Windows error")
    monkeypatch.setattr(ctypes, "WinError", Mock(return_value=error), raising=False)
    last_error = Mock(return_value=5)
    monkeypatch.setattr(ctypes, "get_last_error", last_error, raising=False)
    return SimpleNamespace(
        kernel=kernel,
        advapi=advapi,
        descriptor_address=descriptor_address,
        error=error,
        last_error=last_error,
    )


@pytest.mark.parametrize("creator", ["create_private", "create_closed"])
def test_directory_creation_supplies_the_exact_protected_acl(directory_api, creator):
    import ctypes

    api = directory_api
    path = Path("new-directory")

    def create(name, attributes):
        assert name == str(path)
        value = attributes._obj
        assert value.nLength == ctypes.sizeof(value)
        assert value.lpSecurityDescriptor == api.descriptor_address
        assert not value.bInheritHandle
        return True

    api.kernel.CreateDirectoryW.side_effect = create
    assert getattr(winlinks, creator)(path, USER) is True
    conversion = api.advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW
    expected = f"D:P(A;OICI;FA;;;{USER})(A;OICI;FA;;;SY)"
    if creator == "create_closed":
        expected += "(A;OICI;FA;;;BA)"
    assert conversion.call_count == 1
    assert conversion.call_args.args[0:2] == (expected, 1)
    assert conversion.call_args.args[3] is None
    api.kernel.LocalFree.assert_called_once()
    assert api.kernel.LocalFree.call_args.args[0].value == api.descriptor_address


@pytest.mark.parametrize("creator", ["create_private", "create_closed"])
@pytest.mark.parametrize("error_code", [winlinks.ERROR_ALREADY_EXISTS, 5, 3])
def test_directory_creation_errors_free_the_descriptor(
    directory_api, creator, error_code
):
    api = directory_api
    api.kernel.CreateDirectoryW.return_value = False
    api.last_error.return_value = error_code
    create = getattr(winlinks, creator)
    if error_code == winlinks.ERROR_ALREADY_EXISTS:
        assert create(Path("existing-name"), USER) is False
    else:
        with pytest.raises(OSError) as caught:
            create(Path("new-directory"), USER)
        assert caught.value is api.error
    api.kernel.LocalFree.assert_called_once()
    assert api.kernel.LocalFree.call_args.args[0].value == api.descriptor_address


@pytest.mark.parametrize("creator", ["create_private", "create_closed"])
def test_directory_descriptor_conversion_failure_never_creates(directory_api, creator):
    api = directory_api
    convert = api.advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.side_effect = None
    convert.return_value = False
    with pytest.raises(OSError) as caught:
        getattr(winlinks, creator)(Path("new-directory"), USER)
    assert caught.value is api.error
    api.kernel.CreateDirectoryW.assert_not_called()
    api.kernel.LocalFree.assert_not_called()


@pytest.mark.parametrize("creator", ["create_private", "create_closed"])
def test_directory_creation_exception_frees_the_descriptor(directory_api, creator):
    api = directory_api
    api.kernel.CreateDirectoryW.side_effect = api.error
    with pytest.raises(OSError) as caught:
        getattr(winlinks, creator)(Path("new-directory"), USER)
    assert caught.value is api.error
    api.kernel.LocalFree.assert_called_once()


# --- for real, on Windows ----------------------------------------------------


@windows_only
def test_fresh_data_root_and_private_files_inherit_exact_user_system_access(tmp_path):
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    owner = seen(ordinary).owner
    user = winlinks.current_user_sid()
    data = layout.DataDir(tmp_path / "profile")
    data.prepare()
    nested = data.config / "nested"
    nested.mkdir()
    ordinary_file = nested / "ordinary.txt"
    ordinary_file.write_text("kept")
    private_file = data.run / "private.txt"
    layout.write_private(private_file, "first")
    layout.write_private(private_file, "replacement")
    paths = (data.root, data.config, data.run, nested, ordinary_file, private_file)
    before = {path: seen(path) for path in paths}
    for information in before.values():
        assert information.owner == owner
        assert information.access is not None
        assert len(information.access) == 2
        assert set(information.access) == {allowed(user), allowed(winlinks.SYSTEM)}
    assert winlinks.create_private(data.root, user) is False
    data.prepare()
    assert {path: seen(path) for path in paths} == before
    assert ordinary_file.read_text() == "kept"
    assert private_file.read_text() == "replacement"


@pytest.fixture
def places(tmp_path: Path, monkeypatch) -> Path:
    """A profile and a %ProgramData% of the test's own, and a volume that
    behaves as if it kept no short names."""
    if not winlinks.plain(str(tmp_path)):
        pytest.skip("the temporary directory's own path has a space in it")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "shared"))
    (tmp_path / "local").mkdir()
    (tmp_path / "shared").mkdir()
    monkeypatch.setattr(winlinks, "short_name", no_short_names)
    return tmp_path


def spaced_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "autogpt smoke" / "rabbitmq"
    directory.mkdir(parents=True)
    (directory / "marker").write_text("x")
    return directory


@windows_only
def test_a_folder_with_a_space_is_reached_through_a_junction(places: Path):
    target = spaced_directory(places)

    alias = Path(rabbitmq._short(target))

    assert winlinks.plain(str(alias)), alias
    assert alias.parent == places / "local" / "AutoGPT" / "links"
    assert (alias / "marker").read_text() == "x"
    assert rabbitmq._short(target) == str(alias)  # the same one every start


@windows_only
def test_a_variant_keeps_its_junctions_apart(places: Path, monkeypatch):
    target = spaced_directory(places)
    monkeypatch.setenv(install.VARIABLE, "AutoGPT-voice")
    assert (
        Path(rabbitmq._short(target)).parent
        == places / "local" / "AutoGPT-voice" / "links"
    )


@windows_only
def test_a_stale_junction_is_pointed_at_the_right_folder_again(places: Path):
    target = spaced_directory(places)
    elsewhere = places / "an old data folder"
    elsewhere.mkdir()
    alias = Path(rabbitmq._short(target))
    os.rmdir(alias)
    winlinks.create_junction(elsewhere, alias)
    assert not (alias / "marker").exists()

    assert rabbitmq._short(target) == str(alias)
    assert (alias / "marker").read_text() == "x"
    assert elsewhere.is_dir()  # the junction went, not what it pointed at


@windows_only
def test_a_real_folder_in_the_junctions_place_is_never_followed(places: Path):
    target = spaced_directory(places)
    alias = Path(rabbitmq._short(target))
    os.rmdir(alias)
    alias.mkdir()
    (alias / "planted").write_text("x")
    # The profile's folder cannot be used, so the guarded one is.
    assert Path(rabbitmq._short(target)).parent.parent == places / "shared"
    assert (alias / "planted").exists()


@windows_only
def test_a_profile_with_a_space_falls_back_to_a_closed_folder_in_program_data(
    places: Path, monkeypatch
):
    monkeypatch.setenv("LOCALAPPDATA", str(places / "John Smith" / "Local"))
    target = spaced_directory(places)

    alias = Path(rabbitmq._short(target))

    user = winlinks.current_user_sid()
    assert alias.parent == places / "shared" / f"AutoGPT-{user}"
    assert (alias / "marker").read_text() == "x"
    assert winlinks.trusted(seen(alias.parent).owner, user)
    assert access_entries(alias.parent) == {
        user,
        winlinks.SYSTEM,
        winlinks.ADMINISTRATORS,
    }
    assert rabbitmq._short(target) == str(
        alias
    )  # the folder it made passes its own check


def seen(path: Path) -> winlinks.Seen:
    handle = winlinks.open_as_it_is(path)
    try:
        return winlinks.look(handle)
    finally:
        winlinks._kernel32().CloseHandle(handle)


@windows_only
def test_what_is_looked_at_is_the_link_itself_not_where_it_leads(places: Path):
    """The owner and the kind come from one open handle, and a junction is
    opened, not followed: there is no second look by path for whoever made
    it to answer differently."""
    closed = places / "closed"
    assert winlinks.create_closed(closed, winlinks.current_user_sid())
    link = places / "link"
    winlinks.create_junction(closed, link)

    assert seen(link).attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
    assert not seen(closed).attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
    assert winlinks.refusal(seen(closed), winlinks.current_user_sid()) is None
    assert "is a link" in str(winlinks.refusal(seen(link), winlinks.current_user_sid()))


@windows_only
def test_the_folder_cannot_be_renamed_or_removed_while_it_is_being_used(places: Path):
    user = winlinks.current_user_sid()
    root = winlinks.Root(places / "shared" / f"AutoGPT-{user}", guarded=True)
    with winlinks.held(root, user):
        with pytest.raises(OSError):
            os.rename(root.path, places / "shared" / "moved")
        with pytest.raises(OSError):
            os.rmdir(root.path)
        assert root.path.is_dir()
    os.rmdir(root.path)


@windows_only
def test_an_existing_folder_of_the_users_own_that_others_may_write_is_refused(
    places: Path, monkeypatch
):
    monkeypatch.setenv("LOCALAPPDATA", str(places / "John Smith" / "Local"))
    target = spaced_directory(places)
    user = winlinks.current_user_sid()
    theirs = places / "shared" / f"AutoGPT-{user}"
    theirs.mkdir()
    grant = subprocess.run(
        ["icacls", str(theirs), "/grant", f"*{USERS}:(OI)(CI)M"],
        capture_output=True,
        text=True,
    )
    assert grant.returncode == 0, grant.stdout + grant.stderr

    with pytest.raises(RuntimeError, match="could not make a link"):
        rabbitmq._short(target)
    assert list(theirs.iterdir()) == []
    with pytest.raises(PermissionError, match="changed by someone else"), winlinks.held(
        winlinks.Root(theirs, guarded=True), user
    ):
        pass


@windows_only
def test_junctions_to_folders_that_are_gone_are_cleared_at_the_next_start(places: Path):
    old = places / "an old data folder" / "rabbitmq"
    old.mkdir(parents=True)
    stale = Path(rabbitmq._short(old))
    os.rmdir(old)
    kept = places / "local" / "AutoGPT" / "links" / "a folder someone put here"
    kept.mkdir()

    alias = Path(rabbitmq._short(spaced_directory(places)))

    assert alias.exists() and kept.is_dir()
    assert not os.path.lexists(stale)


@windows_only
def test_a_junction_that_cannot_be_made_to_lead_there_is_not_left_behind(
    places: Path, monkeypatch
):
    target = spaced_directory(places)
    elsewhere = places / "elsewhere"
    elsewhere.mkdir()
    real = winlinks.create_junction
    monkeypatch.setattr(
        winlinks, "create_junction", lambda target, link: real(elsewhere, link)
    )
    root = places / "local" / "links"
    root.mkdir()

    with pytest.raises(OSError, match="does not lead to"):
        winlinks.junction(root, target)

    assert list(root.iterdir()) == []


@windows_only
def test_a_junction_planted_where_the_closed_folder_goes_is_refused(
    places: Path, monkeypatch
):
    """Another user got there first, with a link to a folder they control."""
    monkeypatch.setenv("LOCALAPPDATA", str(places / "John Smith" / "Local"))
    target = spaced_directory(places)
    theirs = places / "theirs"
    theirs.mkdir()
    winlinks.create_junction(
        theirs, places / "shared" / f"AutoGPT-{winlinks.current_user_sid()}"
    )

    with pytest.raises(RuntimeError, match="could not make a link"):
        rabbitmq._short(target)
    assert list(theirs.iterdir()) == []


@windows_only
def test_without_a_place_for_a_junction_the_error_says_what_to_do(
    places: Path, monkeypatch
):
    monkeypatch.setenv("LOCALAPPDATA", str(places / "John Smith" / "Local"))
    monkeypatch.setenv("PROGRAMDATA", str(places / "Program Data"))
    monkeypatch.delenv("ALLUSERSPROFILE", raising=False)
    with pytest.raises(RuntimeError, match="folders without spaces"):
        rabbitmq._short(spaced_directory(places))


@windows_only
def test_a_folder_with_a_short_name_needs_no_junction(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setattr(winlinks, "short_name", lambda path: r"C:\PROGRA~1\AutoGPT")
    assert (
        rabbitmq._short(tmp_path / "Program Files" / "AutoGPT")
        == r"C:\PROGRA~1\AutoGPT"
    )
    assert not (tmp_path / "local").exists()


def access_entries(path: Path) -> set[str]:
    """The SIDs the directory's own access list names, and nothing inherited:
    what `icacls` prints, read back through the same API the module uses."""
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    dacl, descriptor = wintypes.LPVOID(), wintypes.LPVOID()
    file_object, dacl_information = 1, 4
    advapi32.GetNamedSecurityInfoW.argtypes = (
        wintypes.LPCWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.LPVOID),
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.LPVOID),
    )
    error = advapi32.GetNamedSecurityInfoW(
        str(path),
        file_object,
        dacl_information,
        None,
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(descriptor),
    )
    assert error == 0, error

    class AclSize(ctypes.Structure):
        _fields_ = [
            ("count", wintypes.DWORD),
            ("used", wintypes.DWORD),
            ("free", wintypes.DWORD),
        ]

    size = AclSize()
    acl_size_information = 2
    advapi32.GetAclInformation.argtypes = (
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.c_int,
    )
    assert advapi32.GetAclInformation(
        dacl, ctypes.byref(size), ctypes.sizeof(size), acl_size_information
    )
    advapi32.GetAce.argtypes = (
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
    )
    inherited_ace, header_and_mask = 0x10, 8
    sids = set()
    for index in range(size.count):
        ace = wintypes.LPVOID()
        assert advapi32.GetAce(dacl, index, ctypes.byref(ace))
        flags = ctypes.cast(ace, ctypes.POINTER(ctypes.c_ubyte))[1]
        assert not flags & inherited_ace, "the folder inherits access from its parent"
        sids.add(
            winlinks._sid_text(wintypes.LPVOID((ace.value or 0) + header_and_mask))
        )
    return sids
