"""Hosted diagnostic fixtures only; create new directories, never edit an ACL."""

from __future__ import annotations

import ctypes
import time
from pathlib import Path

import installed_smoke_windows as smoke

PRIVATE_DIRS = (
    ".",
    "private",
    "private/logs",
    "private/home",
    "private/appdata",
    "private/localappdata",
    "private/temp",
    "private/data",
)
DATA_DIRS = (
    "config",
    "valkey",
    "rabbitmq",
    "logs",
    "run",
    "home",
    "workspaces",
    "store-media",
    "cache",
    "cache/backend",
    "cache/next",
    "frontend-home",
)
FULL_ACCESS = 0x1F01FF


def require(ok):
    if not ok:
        raise ValueError("fixture security check failed")


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError


def owner_category(owner, user, administrators):
    if owner == user:
        return "token_user"
    if owner == administrators:
        return "administrators"
    raise ValueError("unknown fixture owner")


def validate_descriptor(record, *, protected, directory, principals):
    require(
        type(record) is dict and set(record) == {"owner", "protected", "aces", "label"}
    )
    require(type(record["protected"]) is bool and record["protected"] == protected)
    flags = (3 if directory else 0) | (0 if protected else 16)
    require(len(record["aces"]) == len(principals))
    require(set(record["aces"]) == {(0, flags, FULL_ACCESS, sid) for sid in principals})


class PairACL:
    def __init__(self, baseline: Path, control: Path, deadline: float, native=None):
        for path in (baseline, control):
            smoke.plain_path(path)
        require(baseline.is_dir() and not control.exists())
        require(baseline.parent == control.parent and baseline != control)
        require(len(str(baseline)) == len(str(control)))
        for protected in (smoke.DESKTOP.parents[1],):
            require(
                not control.is_relative_to(protected)
                and not protected.is_relative_to(control)
            )
        self.baseline = baseline
        self.control = control
        self.deadline = deadline
        self.created = {}
        self.baseline_records = None
        check_time(deadline)
        self.native = native if native is not None else NativeAccess()
        owner_category(
            self.native.default_owner, self.native.user, self.native.administrators
        )

    def create_private_tree(self):
        require(self.baseline_records is not None)
        for relative in PRIVATE_DIRS:
            self.create(relative)

    def create(self, relative):
        require(self.baseline_records is not None)
        require(relative in PRIVATE_DIRS and relative not in self.created)
        path = self.control / relative
        smoke.plain_path(path)
        require(not path.exists())
        if relative != ".":
            parent = path.parent.relative_to(self.control).as_posix()
            require(parent in self.created)
            require(path.parent.stat().st_ino == self.created[parent])
        check_time(self.deadline)
        self.native.create_directory(path)
        smoke.plain_path(path)
        self.created[relative] = path.stat().st_ino
        self.inspect_one(path, True, True, "control")
        check_time(self.deadline)

    def inspect_one(self, path, protected, directory, policy):
        check_time(self.deadline)
        smoke.plain_path(path)
        require(path.is_dir() if directory else path.is_file())
        record = self.native.descriptor(path)
        principals = (
            (self.native.user, self.native.system)
            if policy == "control"
            else (
                self.native.system,
                self.native.administrators,
                self.native.owner_rights,
            )
        )
        validate_descriptor(
            record, protected=protected, directory=directory, principals=principals
        )
        owner_category(record["owner"], self.native.user, self.native.administrators)
        return record

    @staticmethod
    def roles(payload):
        roles = dict.fromkeys(PRIVATE_DIRS, True)
        roles["private/runtime"] = True
        roles["private/runtime/postgres"] = True
        roles["private/pair.secret"] = False
        roles.update({f"private/data/{name}": True for name in DATA_DIRS})
        roles.update(
            {
                f"private/runtime/postgres/{name}": value[0] == "directory"
                for name, value in payload.items()
            }
        )
        return roles

    def inspect_baseline(self, payload):
        require(not self.created and self.baseline_records is None)
        records = {}
        for relative, directory in self.roles(payload).items():
            path = self.baseline / relative
            require(path.is_relative_to(self.baseline) and ".." not in path.parts)
            records[relative] = self.inspect_one(
                path, relative in PRIVATE_DIRS, directory, "baseline"
            )
        self.baseline_records = records

    def compare(self, payload):
        require(set(self.created) == set(PRIVATE_DIRS))
        roles = self.roles(payload)
        require(
            self.baseline_records is not None
            and set(roles) == set(self.baseline_records)
        )
        summary = {
            "token": self.native.token_summary(),
            "baseline": {
                "objects": 0,
                "token_user_owned": 0,
                "administrators_owned": 0,
                "protected": 0,
                "exact_user_grants": 0,
                "ace_shape": "python_0700",
            },
            "control": {
                "objects": 0,
                "token_user_owned": 0,
                "administrators_owned": 0,
                "protected": 0,
                "exact_user_grants": 0,
                "ace_shape": "exact_user_system",
            },
            "owner_match": True,
            "integrity_match": True,
            "ordinary_children_inherit": True,
        }
        for relative, directory in roles.items():
            records = []
            for policy, root in (
                ("baseline", self.baseline),
                ("control", self.control),
            ):
                require(
                    (root / relative).is_relative_to(root)
                    and ".." not in Path(relative).parts
                )
                if policy == "control" and relative in self.created:
                    require((root / relative).stat().st_ino == self.created[relative])
                record = self.inspect_one(
                    root / relative, relative in PRIVATE_DIRS, directory, policy
                )
                if policy == "baseline":
                    require(record == self.baseline_records[relative])
                records.append(record)
                result = summary[policy]
                result["objects"] += 1
                result["protected"] += int(record["protected"])
                category = owner_category(
                    record["owner"], self.native.user, self.native.administrators
                )
                result[f"{category}_owned"] += 1
                result["exact_user_grants"] += int(
                    any(ace[3] == self.native.user for ace in record["aces"])
                )
            require(records[0]["owner"] == records[1]["owner"])
            require(records[0]["label"] == records[1]["label"])
        check_time(self.deadline)
        return summary


class NativeAccess:
    def __init__(self):
        smoke.require_clean_host()
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        advapi = ctypes.WinDLL("advapi32", use_last_error=True)
        from ctypes import wintypes as w

        p = ctypes.c_void_p
        pp = ctypes.POINTER(p)
        d = w.DWORD
        pd = ctypes.POINTER(d)
        self.p, self.d = p, d

        class SidAttr(ctypes.Structure):
            _fields_ = [("sid", p), ("attributes", d)]

        class Groups(ctypes.Structure):
            _fields_ = [("count", d), ("groups", SidAttr * 1)]

        class SecurityAttributes(ctypes.Structure):
            _fields_ = [("length", d), ("descriptor", p), ("inherit", w.BOOL)]

        self.security_attributes = SecurityAttributes

        def bind(lib, name, args, result=w.BOOL):
            function = getattr(lib, name)
            function.argtypes = args
            function.restype = result
            return function

        self.free = bind(kernel, "LocalFree", [p], p)
        self.sid_length = bind(advapi, "GetLengthSid", [p], d)
        self.sid_valid = bind(advapi, "IsValidSid", [p])
        self.get_info = bind(advapi, "GetTokenInformation", [p, ctypes.c_int, p, d, pd])
        self.convert_sid = bind(advapi, "ConvertStringSidToSidW", [w.LPCWSTR, pp])
        self.sid_string = bind(advapi, "ConvertSidToStringSidW", [p, pp])
        self.convert_sd = bind(
            advapi,
            "ConvertStringSecurityDescriptorToSecurityDescriptorW",
            [w.LPCWSTR, d, pp, pd],
        )
        self.create = bind(kernel, "CreateDirectoryW", [w.LPCWSTR, p])
        self.get_security = bind(
            advapi,
            "GetNamedSecurityInfoW",
            [w.LPCWSTR, ctypes.c_int, d, pp, pp, pp, pp, pp],
            d,
        )
        self.get_control = bind(
            advapi, "GetSecurityDescriptorControl", [p, ctypes.POINTER(w.WORD), pd]
        )
        self.get_ace = bind(advapi, "GetAce", [p, d, pp])
        self.acl_valid = bind(advapi, "IsValidAcl", [p])
        current = bind(kernel, "GetCurrentProcess", [], p)
        open_token = bind(advapi, "OpenProcessToken", [p, d, pp])
        close = bind(kernel, "CloseHandle", [p])
        token = p()
        require(open_token(current(), 8, ctypes.byref(token)))
        try:
            user = self.info(token, 1)
            user_ptr = p.from_buffer(user)
            self.user = self.sid_bytes(user_ptr)
            self.user_text = self.string_sid(user_ptr)
            owner = self.info(token, 4)
            self.default_owner = self.sid_bytes(p.from_buffer(owner))
            self.system = self.known_sid("S-1-5-18")
            self.administrators = self.known_sid("S-1-5-32-544")
            self.owner_rights = self.known_sid("S-1-3-4")
            require(
                self.user not in {self.system, self.administrators, self.owner_rights}
            )
            groups = self.info(token, 2)
            count = d.from_buffer(groups).value
            require(
                count <= 256
                and Groups.groups.offset + count * ctypes.sizeof(SidAttr) <= len(groups)
            )
            entries = (SidAttr * count).from_buffer(groups, Groups.groups.offset)
            admin = [
                entry.attributes
                for entry in entries
                if self.sid_bytes(entry.sid) == self.administrators
            ]
            require(len(admin) <= 1)
            attrs = admin[0] if admin else None
            admin_state = (
                "absent"
                if attrs is None
                else (
                    "deny_only"
                    if attrs & 16 and not attrs & 4
                    else (
                        "enabled"
                        if attrs & 4 and not attrs & 16
                        else "disabled" if not attrs & 20 else "unknown"
                    )
                )
            )
            require(admin_state != "unknown")
            elevation = d.from_buffer(self.info(token, 20)).value
            elevation_type = d.from_buffer(self.info(token, 18)).value
            require(elevation in {0, 1} and elevation_type in {1, 2, 3})
            self.summary = {
                "observer": "parent_process",
                "elevated": bool(elevation),
                "elevation_type": {1: "default", 2: "full", 3: "limited"}[
                    elevation_type
                ],
                "default_owner": owner_category(
                    self.default_owner, self.user, self.administrators
                ),
                "administrators": admin_state,
                "child_token_lineage": "not_observed",
            }
        finally:
            require(close(token))

    def info(self, token, kind):
        needed = self.d()
        ok = self.get_info(token, kind, None, 0, ctypes.byref(needed))
        require(
            not ok
            and ctypes.get_last_error() in {24, 122}
            and 0 < needed.value <= 65536
        )
        buffer = ctypes.create_string_buffer(needed.value)
        require(self.get_info(token, kind, buffer, len(buffer), ctypes.byref(needed)))
        require(0 < needed.value <= len(buffer))
        return buffer

    def sid_bytes(self, sid):
        require(sid and self.sid_valid(sid))
        length = self.sid_length(sid)
        require(8 <= length <= 68)
        return ctypes.string_at(sid, length)

    def known_sid(self, text):
        sid = self.p()
        require(self.convert_sid(text, ctypes.byref(sid)))
        try:
            return self.sid_bytes(sid)
        finally:
            require(not self.free(sid))

    def string_sid(self, sid):
        value = self.p()
        require(self.sid_string(sid, ctypes.byref(value)))
        try:
            return ctypes.wstring_at(value)
        finally:
            require(not self.free(value))

    def token_summary(self):
        return dict(self.summary)

    def create_directory(self, path):
        descriptor = self.p()
        text = f"D:P(A;OICI;FA;;;{self.user_text})(A;OICI;FA;;;SY)"
        require(self.convert_sd(text, 1, ctypes.byref(descriptor), None))
        try:
            attributes = self.security_attributes(
                ctypes.sizeof(self.security_attributes), descriptor, False
            )
            require(self.create(str(path), ctypes.byref(attributes)))
        finally:
            require(not self.free(descriptor))

    def aces(self, acl, *, label=False):
        if not acl and label:
            return ()
        require(acl and self.acl_valid(acl))
        header = ctypes.string_at(acl, 8)
        count = int.from_bytes(header[4:6], "little")
        require(count <= 16)
        result = []
        for index in range(count):
            pointer = self.p()
            require(self.get_ace(acl, index, ctypes.byref(pointer)))
            raw = ctypes.string_at(pointer, 8)
            require(raw[0] == (17 if label else 0))
            sid = self.sid_bytes(pointer.value + 8)
            require(int.from_bytes(raw[2:4], "little") == 8 + len(sid))
            result.append((raw[0], raw[1], int.from_bytes(raw[4:8], "little"), sid))
        return tuple(result)

    def descriptor(self, path):
        from ctypes import wintypes as w

        owner, dacl, label, descriptor = (self.p() for _ in range(4))
        require(
            self.get_security(
                str(path),
                1,
                1 | 4 | 16,
                ctypes.byref(owner),
                None,
                ctypes.byref(dacl),
                ctypes.byref(label),
                ctypes.byref(descriptor),
            )
            == 0
        )
        try:
            control, revision = w.WORD(), self.d()
            require(
                self.get_control(
                    descriptor, ctypes.byref(control), ctypes.byref(revision)
                )
            )
            return {
                "owner": self.sid_bytes(owner),
                "protected": bool(control.value & 0x1000),
                "aces": self.aces(dacl),
                "label": self.aces(label, label=True),
            }
        finally:
            require(not self.free(descriptor))
