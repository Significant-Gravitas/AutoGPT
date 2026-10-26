"""Two one-line changes to Erlang/OTP, compiled into the bundle.

The desktop app must not open a socket beyond loopback: nothing needs one,
and on Windows any such socket makes the firewall ask the user whether
"Erlang" may accept connections from public networks, on first launch.
RabbitMQ's own listeners are configured to loopback, but two things are not
configurable:

* `inet:gethostname/0` opens a UDP socket only to ask the OS for the host
  name, bound to every interface. Every distributed Erlang node calls it.
* RabbitMQ checks that its distribution port is free by listening on it,
  again on every interface, and closing it at once
  (`rabbit_prelaunch_dist:dist_port_use_check_ipv4/2`).

Both go through OTP's `inet_udp` and `inet_tcp`. The patched modules bind to
127.0.0.1 when the caller gave no address; an explicit address still wins.
They are compiled from the pinned OTP source and put ahead of the stock
modules on the code path (`-pa`, see runtime/autogpt_desktop/rabbitmq.py),
so OTP itself ships unmodified.

Found by tracing socket calls inside the VM: each socket lives for
microseconds, so sampling the process's sockets from outside never sees it,
and the firewall always does.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from artifacts import ERLANG_VERSION, Artifact
from fetch import download

_SOURCE = (
    f"https://raw.githubusercontent.com/erlang/otp/OTP-{ERLANG_VERSION}/lib/kernel/src"
)

HEADER = Artifact(
    f"{_SOURCE}/inet_int.hrl",
    "e2bf01f7fc89b43f2667757976152c7240aaa51ebd25d85d5e3ce64b8228d3eb",
)

HELPER = """\
%% AutoGPT desktop: without an explicit address, bind to loopback instead of
%% every interface. See build/erlang_patches.py.
loopback_unless_given(undefined) -> {127,0,0,1};
loopback_unless_given(Addr) -> Addr.

"""


@dataclass(frozen=True)
class Patch:
    source: Artifact
    # The call that binds the new socket, and the same call with the address
    # passed through loopback_unless_given/1.
    bind_call: str
    patched_bind_call: str
    # A line that starts a later function; the helper is inserted before it.
    helper_before: str

    @property
    def module(self) -> str:
        return self.source.filename

    def apply(self, text: str) -> str:
        if text.count(self.bind_call) != 1 or text.count(self.helper_before) != 1:
            raise ValueError(
                f"{self.module} no longer matches the patch; review it for OTP {ERLANG_VERSION}"
            )
        return text.replace(self.bind_call, self.patched_bind_call).replace(
            self.helper_before, HELPER + self.helper_before
        )


PATCHES = (
    Patch(
        source=Artifact(
            f"{_SOURCE}/inet_udp.erl",
            "4d12cd389b5047b73fdc3136c59f0f473aba229c180730949b3e0616b02e3997",
        ),
        bind_call=(
            "\t    inet:open_bind(\n"
            "\t      Fd, BAddr, BPort, SockOpts, ?PROTO, ?FAMILY, ?TYPE, ?MODULE);"
        ),
        patched_bind_call=(
            "\t    inet:open_bind(\n"
            "\t      Fd, loopback_unless_given(BAddr), BPort, SockOpts,\n"
            "\t      ?PROTO, ?FAMILY, ?TYPE, ?MODULE);"
        ),
        helper_before="send(S, {A,B,C,D} = IP, Port, Data)",
    ),
    Patch(
        source=Artifact(
            f"{_SOURCE}/inet_tcp.erl",
            "f39570dfe93f2a9730b29d00f211f7fe0f741374a393be19eac45abe201c9b38",
        ),
        # inet_tcp has exactly one open_bind call, in listen/2. connect/3,4
        # binds its local side elsewhere and is not affected.
        bind_call=(
            "                inet:open_bind(\n"
            "                  Fd, BAddr, BPort, SockOpts,\n"
            "                  proto(Protocol), ?FAMILY, ?TYPE, ?MODULE)"
        ),
        patched_bind_call=(
            "                inet:open_bind(\n"
            "                  Fd, loopback_unless_given(BAddr), BPort, SockOpts,\n"
            "                  proto(Protocol), ?FAMILY, ?TYPE, ?MODULE)"
        ),
        helper_before="accept(L) ->",
    ),
)


def build(erlc: Path, cache: Path, output: Path) -> None:
    work = cache / f"otp-{ERLANG_VERSION}-src"
    patched = work / "patched"
    patched.mkdir(parents=True, exist_ok=True)
    header = download(HEADER, work)
    (patched / header.name).write_bytes(header.read_bytes())
    output.mkdir(parents=True, exist_ok=True)
    for patch in PATCHES:
        original = (
            download(patch.source, work)
            .read_text(encoding="utf-8")
            .replace("\r\n", "\n")
        )
        target = patched / patch.module
        target.write_text(patch.apply(original), encoding="utf-8", newline="\n")
        subprocess.run(
            [str(erlc), "-o", str(output), str(target)], check=True, cwd=patched
        )
        if not (output / target.with_suffix(".beam").name).is_file():
            raise RuntimeError(f"erlc produced no beam file for {patch.module}")
