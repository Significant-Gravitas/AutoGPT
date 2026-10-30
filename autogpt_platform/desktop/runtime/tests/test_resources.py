from pathlib import Path

import pytest

from autogpt_desktop import apps, ports, postgres, resources, settings
from autogpt_desktop.layout import Bundle, DataDir
from autogpt_desktop.runs import Cache

GB = 1024**3


@pytest.fixture
def data(tmp_path: Path) -> DataDir:
    data = DataDir(tmp_path / "data")
    data.prepare()
    return data


@pytest.fixture
def bundle(tmp_path: Path) -> Bundle:
    return Bundle(tmp_path / "runtime")


@pytest.mark.parametrize(
    ("ram_gb", "cores", "name", "graph_workers", "copilot_workers"),
    [
        (4, 2, resources.COMPACT, "4", "2"),
        (7.6, 8, resources.COMPACT, "4", "2"),  # what a machine sold with 8 GB reports
        (15.8, 2, resources.BALANCED, "4", "2"),
        (16, 8, resources.BALANCED, "8", "4"),
        (
            64,
            32,
            resources.BALANCED,
            "10",
            "5",
        ),  # never more than upstream's own defaults
    ],
)
def test_the_profile_fits_the_machine(
    ram_gb, cores, name, graph_workers, copilot_workers
):
    profile = resources.choose({}, ram_bytes=int(ram_gb * GB), cores=cores)

    assert profile.name == name
    assert profile.merged
    assert profile.backend_env == {
        "NUM_GRAPH_WORKERS": graph_workers,
        "NUM_COPILOT_WORKERS": copilot_workers,
        "SCHEDULER_DB_POOL_SIZE": "3",
    }
    assert f"{cores} cores" in profile.describe()


@pytest.mark.parametrize("asked", ["isolated", " Isolated "])
def test_settings_env_can_ask_for_a_process_per_service(asked: str):
    profile = resources.choose(
        {resources.PROFILE_SETTING: asked}, ram_bytes=4 * GB, cores=2
    )

    assert profile.name == resources.ISOLATED
    assert not profile.merged
    # Upstream's own worker pools; only what the connection budget counts on.
    assert profile.backend_env == {"SCHEDULER_DB_POOL_SIZE": "3"}
    assert "settings.env" in profile.describe()


def test_a_small_machine_can_be_told_it_is_a_big_one():
    profile = resources.choose(
        {resources.PROFILE_SETTING: "balanced"}, ram_bytes=4 * GB, cores=16
    )
    assert profile.name == resources.BALANCED
    assert profile.backend_env["NUM_GRAPH_WORKERS"] == "10"


def test_a_profile_that_does_not_exist_is_ignored_with_a_warning(caplog):
    profile = resources.choose(
        {resources.PROFILE_SETTING: "turbo"}, ram_bytes=32 * GB, cores=8
    )
    assert profile.name == resources.BALANCED
    assert "turbo" in caplog.text


def backend_env(bundle: Bundle, data: DataDir, user: dict[str, str]) -> dict[str, str]:
    port = {name: 20000 + index for index, name in enumerate(ports.PORT_NAMES)}
    secret = settings.ensure_secrets(bundle, data)
    profile = resources.choose(user, ram_bytes=4 * GB, cores=2)
    return settings.backend_environment(
        bundle, data, port, secret, user, profile.backend_env
    )


def test_the_profile_sets_the_pool_sizes(bundle: Bundle, data: DataDir):
    env = backend_env(bundle, data, {})
    assert env["NUM_GRAPH_WORKERS"] == "4"
    assert env["NUM_COPILOT_WORKERS"] == "2"


def test_a_pool_size_in_settings_env_wins_over_the_profile(
    bundle: Bundle, data: DataDir
):
    env = backend_env(bundle, data, {"NUM_GRAPH_WORKERS": "32"})
    assert env["NUM_GRAPH_WORKERS"] == "32"
    assert env["NUM_COPILOT_WORKERS"] == "2"


def test_the_runtimes_own_wiring_still_wins_over_both(bundle: Bundle, data: DataDir):
    env = backend_env(bundle, data, {"REDIS_HOST": "elsewhere"})
    assert env["REDIS_HOST"] == "127.0.0.1"


def test_no_profile_is_given_without_one(bundle: Bundle, data: DataDir):
    port = {name: 20000 + index for index, name in enumerate(ports.PORT_NAMES)}
    secret = settings.ensure_secrets(bundle, data)
    env = settings.backend_environment(bundle, data, port, secret, {})
    assert "NUM_GRAPH_WORKERS" not in env


def test_postgres_is_given_its_limits_on_every_start(bundle: Bundle, data: DataDir):
    """On the command line: postgresql.conf is written once, when the
    cluster is created, and installs from before the limits must get them."""
    argv = postgres.process(bundle, data, 25432, resources.POSTGRES_LIMITS).argv
    pairs = [argv[index + 1] for index, argument in enumerate(argv) if argument == "-c"]

    assert pairs == [
        "unix_socket_directories=",
        "max_connections=50",
        "max_worker_processes=4",
        "max_parallel_workers=0",
        "autovacuum_max_workers=1",
    ]
    assert postgres.process(bundle, data, 25432).argv.count("-c") == 1


def test_the_connection_limit_covers_every_pool():
    """Three Prisma engines (the most, with a process per service), the
    skills-catalog publisher's, the frontend's, the scheduler's two, the
    runtime's own, and PostgreSQL's reserved slots."""
    engines = sum(service.database for service in apps.SERVICES) + 1
    scheduler = resources.SCHEDULER_POOLS * resources.SCHEDULER_DB_POOL_SIZE
    needed = engines * settings.DB_CONNECTION_LIMIT + 10 + scheduler + 2 + 3
    assert needed <= int(resources.POSTGRES_LIMITS["max_connections"])


@pytest.mark.parametrize("name", resources.PROFILES)
def test_every_profile_pins_the_scheduler_pool_the_connection_limit_counts_on(
    bundle: Bundle, data: DataDir, name: str
):
    """Upstream raising its default must not use up PostgreSQL's connections
    unseen; a value in settings.env is the user's to raise."""
    port = {name: 20000 + index for index, name in enumerate(ports.PORT_NAMES)}
    secret = settings.ensure_secrets(bundle, data)
    pinned = str(resources.SCHEDULER_DB_POOL_SIZE)
    for user, expected in (({}, pinned), ({"SCHEDULER_DB_POOL_SIZE": "8"}, "8")):
        profile = resources.choose(
            {**user, resources.PROFILE_SETTING: name}, 16 * GB, 8
        )
        env = settings.backend_environment(
            bundle, data, port, secret, user, profile.backend_env
        )
        assert env["SCHEDULER_DB_POOL_SIZE"] == expected


def test_by_default_nine_services_run_in_five_processes():
    groups = apps.layout(merged=True)

    assert [group.name for group in groups] == [
        "database-manager",
        "swap-credentials",
        "workers",
        "websocket",
        "api",
    ]
    hosted = [name for group in groups for name in group.services]
    assert sorted(hosted) == sorted(service.name for service in apps.SERVICES)
    assert groups[2].services[0] == "executor"  # it imports the most as it starts


def test_services_that_connect_to_the_database_never_share_with_ones_that_do_not():
    """The backend decides per process whether a query goes to the database
    or to database-manager."""
    for group in apps.layout(merged=True):
        connecting = {apps.service(name).database for name in group.services}
        assert len(connecting) == 1, group
    workers = next(
        group for group in apps.layout(merged=True) if group.name == "workers"
    )
    assert not any(apps.service(name).database for name in workers.services)


@pytest.mark.parametrize("merged", [True, False])
def test_websocket_queries_through_database_manager_in_its_own_host(merged: bool):
    groups = apps.layout(merged=merged)
    websocket = next(group for group in groups if "websocket" in group.services)
    rest = next(group for group in groups if "rest" in group.services)

    assert not apps.service("websocket").database
    assert apps.service("rest").database
    assert websocket.services == ("websocket",)
    assert rest.services == ("rest",)
    assert websocket.name != rest.name


def test_isolated_is_a_process_per_service_in_the_appliances_order():
    groups = apps.layout(merged=False)
    assert [group.name for group in groups] == [
        service.name for service in apps.SERVICES
    ]
    assert all(group.services == (group.name,) for group in groups)


def flags(bundle: Bundle, group: apps.Group) -> list[str]:
    argv = apps.host_argv(bundle, group)
    assert argv[1:4] == ["-c", apps.HOST_CODE, str(apps.PACKAGE_PARENT)]
    return argv[4:]


def test_how_each_host_is_started(bundle: Bundle):
    database, swap, workers, websocket, api = apps.layout(merged=True)
    budget = ["--stop-budget", "2.5"]

    assert flags(bundle, database) == [
        "--name", "database-manager", *budget, "database-manager=autogpt_desktop.backend_entries:database_main",
    ]  # fmt: skip
    assert flags(bundle, swap) == [
        "--name",
        "swap-credentials",
        "--stop-budget",
        "29.5",
        "swap-credentials=autogpt_desktop.backend_entries:swap_main",
    ]
    budget = ["--stop-budget", "29.5"]
    assert flags(bundle, workers)[:5] == ["--name", "workers", *budget, "--no-database"]
    assert flags(bundle, workers)[5] == "executor=backend.exec:main"
    assert flags(bundle, websocket) == [
        "--name", "websocket", *budget, "--shared-loop", "websocket=backend.ws:main",
    ]  # fmt: skip
    assert flags(bundle, api) == [
        "--name", "api", *budget, "--shared-loop", "rest=backend.rest:main",
    ]  # fmt: skip
    # Alone, a worker may do what it likes with the database; nothing shares
    # its process.
    alone = apps.Group("executor", ("executor",))
    assert "--no-database" not in flags(bundle, alone)
    assert "--shared-loop" in flags(bundle, apps.Group("rest", ("rest",)))


def test_the_host_stops_inside_the_time_the_supervisor_allows():
    assert 0 < apps.HOST_STOP_BUDGET_SECONDS < apps.STOP_TIMEOUT_SECONDS


def test_every_service_has_a_health_address_on_a_port_of_its_own():
    port = {name: 20000 + index for index, name in enumerate(ports.PORT_NAMES)}
    urls = apps.health_urls(port)

    assert set(urls) == {service.name for service in apps.SERVICES} | {apps.FRONTEND}
    assert len(set(urls.values())) == len(urls)
    assert urls["rest"] == f"http://127.0.0.1:{port['agent_api']}/health"
    assert urls["executor"] == f"http://127.0.0.1:{port['execution_manager']}/metrics"


def test_a_host_process_is_what_the_supervisor_stops_and_restarts(
    bundle: Bundle, data: DataDir
):
    workers = next(
        group for group in apps.layout(merged=True) if group.name == "workers"
    )
    process = apps.host_process(bundle, data, {"A": "b"}, workers, Cache(1, "password"))

    assert process.name == "workers"
    assert process.cwd == bundle.backend_dir
    assert process.stop_timeout == apps.REFRESH_STOP_TIMEOUT_SECONDS
    assert process.env["A"] == "b"
    assert process.before_start is not None


def test_the_bundle_version_follows_the_backend(bundle: Bundle):
    backend = bundle.backend_dir / "backend"
    backend.mkdir(parents=True)
    (backend / "app.py").write_text("one")
    first = apps.bundle_version(bundle)
    assert apps.bundle_version(bundle) == first

    (backend / "app.py").write_text("one and more")
    assert apps.bundle_version(bundle) != first
