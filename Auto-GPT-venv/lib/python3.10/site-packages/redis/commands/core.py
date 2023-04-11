# from __future__ import annotations

import datetime
import hashlib
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
    AbsExpiryT,
    AnyKeyT,
    BitfieldOffsetT,
    ChannelT,
    CommandsProtocol,
    ConsumerT,
    EncodableT,
    ExpiryT,
    FieldT,
    GroupT,
    KeysT,
    KeyT,
    PatternT,
    ScriptTextT,
    StreamIdT,
    TimeoutSecT,
    ZScoreBoundT,
)

from .helpers import list_or_args

if TYPE_CHECKING:
    from redis.asyncio.client import Redis as AsyncRedis
    from redis.client import Redis

ResponseT = Union[Awaitable, Any]


class ACLCommands(CommandsProtocol):
    """
    Redis Access Control List (ACL) commands.
    see: https://redis.io/topics/acl
    """

    def acl_cat(self, category: Union[str, None] = None, **kwargs) -> ResponseT:
        """
        Returns a list of categories or commands within a category.

        If ``category`` is not supplied, returns a list of all categories.
        If ``category`` is supplied, returns a list of all commands within
        that category.

        For more information see https://redis.io/commands/acl-cat
        """
        pieces: list[EncodableT] = [category] if category else []
        return self.execute_command("ACL CAT", *pieces, **kwargs)

    def acl_dryrun(self, username, *args, **kwargs):
        """
        Simulate the execution of a given command by a given ``username``.

        For more information see https://redis.io/commands/acl-dryrun
        """
        return self.execute_command("ACL DRYRUN", username, *args, **kwargs)

    def acl_deluser(self, *username: str, **kwargs) -> ResponseT:
        """
        Delete the ACL for the specified ``username``s

        For more information see https://redis.io/commands/acl-deluser
        """
        return self.execute_command("ACL DELUSER", *username, **kwargs)

    def acl_genpass(self, bits: Union[int, None] = None, **kwargs) -> ResponseT:
        """Generate a random password value.
        If ``bits`` is supplied then use this number of bits, rounded to
        the next multiple of 4.
        See: https://redis.io/commands/acl-genpass
        """
        pieces = []
        if bits is not None:
            try:
                b = int(bits)
                if b < 0 or b > 4096:
                    raise ValueError
            except ValueError:
                raise DataError(
                    "genpass optionally accepts a bits argument, between 0 and 4096."
                )
        return self.execute_command("ACL GENPASS", *pieces, **kwargs)

    def acl_getuser(self, username: str, **kwargs) -> ResponseT:
        """
        Get the ACL details for the specified ``username``.

        If ``username`` does not exist, return None

        For more information see https://redis.io/commands/acl-getuser
        """
        return self.execute_command("ACL GETUSER", username, **kwargs)

    def acl_help(self, **kwargs) -> ResponseT:
        """The ACL HELP command returns helpful text describing
        the different subcommands.

        For more information see https://redis.io/commands/acl-help
        """
        return self.execute_command("ACL HELP", **kwargs)

    def acl_list(self, **kwargs) -> ResponseT:
        """
        Return a list of all ACLs on the server

        For more information see https://redis.io/commands/acl-list
        """
        return self.execute_command("ACL LIST", **kwargs)

    def acl_log(self, count: Union[int, None] = None, **kwargs) -> ResponseT:
        """
        Get ACL logs as a list.
        :param int count: Get logs[0:count].
        :rtype: List.

        For more information see https://redis.io/commands/acl-log
        """
        args = []
        if count is not None:
            if not isinstance(count, int):
                raise DataError("ACL LOG count must be an integer")
            args.append(count)

        return self.execute_command("ACL LOG", *args, **kwargs)

    def acl_log_reset(self, **kwargs) -> ResponseT:
        """
        Reset ACL logs.
        :rtype: Boolean.

        For more information see https://redis.io/commands/acl-log
        """
        args = [b"RESET"]
        return self.execute_command("ACL LOG", *args, **kwargs)

    def acl_load(self, **kwargs) -> ResponseT:
        """
        Load ACL rules from the configured ``aclfile``.

        Note that the server must be configured with the ``aclfile``
        directive to be able to load ACL rules from an aclfile.

        For more information see https://redis.io/commands/acl-load
        """
        return self.execute_command("ACL LOAD", **kwargs)

    def acl_save(self, **kwargs) -> ResponseT:
        """
        Save ACL rules to the configured ``aclfile``.

        Note that the server must be configured with the ``aclfile``
        directive to be able to save ACL rules to an aclfile.

        For more information see https://redis.io/commands/acl-save
        """
        return self.execute_command("ACL SAVE", **kwargs)

    def acl_setuser(
        self,
        username: str,
        enabled: bool = False,
        nopass: bool = False,
        passwords: Union[str, Iterable[str], None] = None,
        hashed_passwords: Union[str, Iterable[str], None] = None,
        categories: Optional[Iterable[str]] = None,
        commands: Optional[Iterable[str]] = None,
        keys: Optional[Iterable[KeyT]] = None,
        channels: Optional[Iterable[ChannelT]] = None,
        selectors: Optional[Iterable[Tuple[str, KeyT]]] = None,
        reset: bool = False,
        reset_keys: bool = False,
        reset_channels: bool = False,
        reset_passwords: bool = False,
        **kwargs,
    ) -> ResponseT:
        """
        Create or update an ACL user.

        Create or update the ACL for ``username``. If the user already exists,
        the existing ACL is completely overwritten and replaced with the
        specified values.

        ``enabled`` is a boolean indicating whether the user should be allowed
        to authenticate or not. Defaults to ``False``.

        ``nopass`` is a boolean indicating whether the can authenticate without
        a password. This cannot be True if ``passwords`` are also specified.

        ``passwords`` if specified is a list of plain text passwords
        to add to or remove from the user. Each password must be prefixed with
        a '+' to add or a '-' to remove. For convenience, the value of
        ``passwords`` can be a simple prefixed string when adding or
        removing a single password.

        ``hashed_passwords`` if specified is a list of SHA-256 hashed passwords
        to add to or remove from the user. Each hashed password must be
        prefixed with a '+' to add or a '-' to remove. For convenience,
        the value of ``hashed_passwords`` can be a simple prefixed string when
        adding or removing a single password.

        ``categories`` if specified is a list of strings representing category
        permissions. Each string must be prefixed with either a '+' to add the
        category permission or a '-' to remove the category permission.

        ``commands`` if specified is a list of strings representing command
        permissions. Each string must be prefixed with either a '+' to add the
        command permission or a '-' to remove the command permission.

        ``keys`` if specified is a list of key patterns to grant the user
        access to. Keys patterns allow '*' to support wildcard matching. For
        example, '*' grants access to all keys while 'cache:*' grants access
        to all keys that are prefixed with 'cache:'. ``keys`` should not be
        prefixed with a '~'.

        ``reset`` is a boolean indicating whether the user should be fully
        reset prior to applying the new ACL. Setting this to True will
        remove all existing passwords, flags and privileges from the user and
        then apply the specified rules. If this is False, the user's existing
        passwords, flags and privileges will be kept and any new specified
        rules will be applied on top.

        ``reset_keys`` is a boolean indicating whether the user's key
        permissions should be reset prior to applying any new key permissions
        specified in ``keys``. If this is False, the user's existing
        key permissions will be kept and any new specified key permissions
        will be applied on top.

        ``reset_channels`` is a boolean indicating whether the user's channel
        permissions should be reset prior to applying any new channel permissions
        specified in ``channels``.If this is False, the user's existing
        channel permissions will be kept and any new specified channel permissions
        will be applied on top.

        ``reset_passwords`` is a boolean indicating whether to remove all
        existing passwords and the 'nopass' flag from the user prior to
        applying any new passwords specified in 'passwords' or
        'hashed_passwords'. If this is False, the user's existing passwords
        and 'nopass' status will be kept and any new specified passwords
        or hashed_passwords will be applied on top.

        For more information see https://redis.io/commands/acl-setuser
        """
        encoder = self.get_encoder()
        pieces: List[EncodableT] = [username]

        if reset:
            pieces.append(b"reset")

        if reset_keys:
            pieces.append(b"resetkeys")

        if reset_channels:
            pieces.append(b"resetchannels")

        if reset_passwords:
            pieces.append(b"resetpass")

        if enabled:
            pieces.append(b"on")
        else:
            pieces.append(b"off")

        if (passwords or hashed_passwords) and nopass:
            raise DataError(
                "Cannot set 'nopass' and supply 'passwords' or 'hashed_passwords'"
            )

        if passwords:
            # as most users will have only one password, allow remove_passwords
            # to be specified as a simple string or a list
            passwords = list_or_args(passwords, [])
            for i, password in enumerate(passwords):
                password = encoder.encode(password)
                if password.startswith(b"+"):
                    pieces.append(b">%s" % password[1:])
                elif password.startswith(b"-"):
                    pieces.append(b"<%s" % password[1:])
                else:
                    raise DataError(
                        f"Password {i} must be prefixed with a "
                        f'"+" to add or a "-" to remove'
                    )

        if hashed_passwords:
            # as most users will have only one password, allow remove_passwords
            # to be specified as a simple string or a list
            hashed_passwords = list_or_args(hashed_passwords, [])
            for i, hashed_password in enumerate(hashed_passwords):
                hashed_password = encoder.encode(hashed_password)
                if hashed_password.startswith(b"+"):
                    pieces.append(b"#%s" % hashed_password[1:])
                elif hashed_password.startswith(b"-"):
                    pieces.append(b"!%s" % hashed_password[1:])
                else:
                    raise DataError(
                        f"Hashed password {i} must be prefixed with a "
                        f'"+" to add or a "-" to remove'
                    )

        if nopass:
            pieces.append(b"nopass")

        if categories:
            for category in categories:
                category = encoder.encode(category)
                # categories can be prefixed with one of (+@, +, -@, -)
                if category.startswith(b"+@"):
                    pieces.append(category)
                elif category.startswith(b"+"):
                    pieces.append(b"+@%s" % category[1:])
                elif category.startswith(b"-@"):
                    pieces.append(category)
                elif category.startswith(b"-"):
                    pieces.append(b"-@%s" % category[1:])
                else:
                    raise DataError(
                        f'Category "{encoder.decode(category, force=True)}" '
                        'must be prefixed with "+" or "-"'
                    )
        if commands:
            for cmd in commands:
                cmd = encoder.encode(cmd)
                if not cmd.startswith(b"+") and not cmd.startswith(b"-"):
                    raise DataError(
                        f'Command "{encoder.decode(cmd, force=True)}" '
                        'must be prefixed with "+" or "-"'
                    )
                pieces.append(cmd)

        if keys:
            for key in keys:
                key = encoder.encode(key)
                if not key.startswith(b"%") and not key.startswith(b"~"):
                    key = b"~%s" % key
                pieces.append(key)

        if channels:
            for channel in channels:
                channel = encoder.encode(channel)
                pieces.append(b"&%s" % channel)

        if selectors:
            for cmd, key in selectors:
                cmd = encoder.encode(cmd)
                if not cmd.startswith(b"+") and not cmd.startswith(b"-"):
                    raise DataError(
                        f'Command "{encoder.decode(cmd, force=True)}" '
                        'must be prefixed with "+" or "-"'
                    )

                key = encoder.encode(key)
                if not key.startswith(b"%") and not key.startswith(b"~"):
                    key = b"~%s" % key

                pieces.append(b"(%s %s)" % (cmd, key))

        return self.execute_command("ACL SETUSER", *pieces, **kwargs)

    def acl_users(self, **kwargs) -> ResponseT:
        """Returns a list of all registered users on the server.

        For more information see https://redis.io/commands/acl-users
        """
        return self.execute_command("ACL USERS", **kwargs)

    def acl_whoami(self, **kwargs) -> ResponseT:
        """Get the username for the current connection

        For more information see https://redis.io/commands/acl-whoami
        """
        return self.execute_command("ACL WHOAMI", **kwargs)


AsyncACLCommands = ACLCommands


class ManagementCommands(CommandsProtocol):
    """
    Redis management commands
    """

    def auth(self, password, username=None, **kwargs):
        """
        Authenticates the user. If you do not pass username, Redis will try to
        authenticate for the "default" user. If you do pass username, it will
        authenticate for the given user.
        For more information see https://redis.io/commands/auth
        """
        pieces = []
        if username is not None:
            pieces.append(username)
        pieces.append(password)
        return self.execute_command("AUTH", *pieces, **kwargs)

    def bgrewriteaof(self, **kwargs):
        """Tell the Redis server to rewrite the AOF file from data in memory.

        For more information see https://redis.io/commands/bgrewriteaof
        """
        return self.execute_command("BGREWRITEAOF", **kwargs)

    def bgsave(self, schedule: bool = True, **kwargs) -> ResponseT:
        """
        Tell the Redis server to save its data to disk.  Unlike save(),
        this method is asynchronous and returns immediately.

        For more information see https://redis.io/commands/bgsave
        """
        pieces = []
        if schedule:
            pieces.append("SCHEDULE")
        return self.execute_command("BGSAVE", *pieces, **kwargs)

    def role(self) -> ResponseT:
        """
        Provide information on the role of a Redis instance in
        the context of replication, by returning if the instance
        is currently a master, slave, or sentinel.

        For more information see https://redis.io/commands/role
        """
        return self.execute_command("ROLE")

    def client_kill(self, address: str, **kwargs) -> ResponseT:
        """Disconnects the client at ``address`` (ip:port)

        For more information see https://redis.io/commands/client-kill
        """
        return self.execute_command("CLIENT KILL", address, **kwargs)

    def client_kill_filter(
        self,
        _id: Union[str, None] = None,
        _type: Union[str, None] = None,
        addr: Union[str, None] = None,
        skipme: Union[bool, None] = None,
        laddr: Union[bool, None] = None,
        user: str = None,
        **kwargs,
    ) -> ResponseT:
        """
        Disconnects client(s) using a variety of filter options
        :param _id: Kills a client by its unique ID field
        :param _type: Kills a client by type where type is one of 'normal',
        'master', 'slave' or 'pubsub'
        :param addr: Kills a client by its 'address:port'
        :param skipme: If True, then the client calling the command
        will not get killed even if it is identified by one of the filter
        options. If skipme is not provided, the server defaults to skipme=True
        :param laddr: Kills a client by its 'local (bind) address:port'
        :param user: Kills a client for a specific user name
        """
        args = []
        if _type is not None:
            client_types = ("normal", "master", "slave", "pubsub")
            if str(_type).lower() not in client_types:
                raise DataError(f"CLIENT KILL type must be one of {client_types!r}")
            args.extend((b"TYPE", _type))
        if skipme is not None:
            if not isinstance(skipme, bool):
                raise DataError("CLIENT KILL skipme must be a bool")
            if skipme:
                args.extend((b"SKIPME", b"YES"))
            else:
                args.extend((b"SKIPME", b"NO"))
        if _id is not None:
            args.extend((b"ID", _id))
        if addr is not None:
            args.extend((b"ADDR", addr))
        if laddr is not None:
            args.extend((b"LADDR", laddr))
        if user is not None:
            args.extend((b"USER", user))
        if not args:
            raise DataError(
                "CLIENT KILL <filter> <value> ... ... <filter> "
                "<value> must specify at least one filter"
            )
        return self.execute_command("CLIENT KILL", *args, **kwargs)

    def client_info(self, **kwargs) -> ResponseT:
        """
        Returns information and statistics about the current
        client connection.

        For more information see https://redis.io/commands/client-info
        """
        return self.execute_command("CLIENT INFO", **kwargs)

    def client_list(
        self, _type: Union[str, None] = None, client_id: List[EncodableT] = [], **kwargs
    ) -> ResponseT:
        """
        Returns a list of currently connected clients.
        If type of client specified, only that type will be returned.
        :param _type: optional. one of the client types (normal, master,
         replica, pubsub)
        :param client_id: optional. a list of client ids

        For more information see https://redis.io/commands/client-list
        """
        args = []
        if _type is not None:
            client_types = ("normal", "master", "replica", "pubsub")
            if str(_type).lower() not in client_types:
                raise DataError(f"CLIENT LIST _type must be one of {client_types!r}")
            args.append(b"TYPE")
            args.append(_type)
        if not isinstance(client_id, list):
            raise DataError("client_id must be a list")
        if client_id:
            args.append(b"ID")
            args.append(" ".join(client_id))
        return self.execute_command("CLIENT LIST", *args, **kwargs)

    def client_getname(self, **kwargs) -> ResponseT:
        """
        Returns the current connection name

        For more information see https://redis.io/commands/client-getname
        """
        return self.execute_command("CLIENT GETNAME", **kwargs)

    def client_getredir(self, **kwargs) -> ResponseT:
        """
        Returns the ID (an integer) of the client to whom we are
        redirecting tracking notifications.

        see: https://redis.io/commands/client-getredir
        """
        return self.execute_command("CLIENT GETREDIR", **kwargs)

    def client_reply(
        self, reply: Union[Literal["ON"], Literal["OFF"], Literal["SKIP"]], **kwargs
    ) -> ResponseT:
        """
        Enable and disable redis server replies.
        ``reply`` Must be ON OFF or SKIP,
            ON - The default most with server replies to commands
            OFF - Disable server responses to commands
            SKIP - Skip the response of the immediately following command.

        Note: When setting OFF or SKIP replies, you will need a client object
        with a timeout specified in seconds, and will need to catch the
        TimeoutError.
              The test_client_reply unit test illustrates this, and
              conftest.py has a client with a timeout.

        See https://redis.io/commands/client-reply
        """
        replies = ["ON", "OFF", "SKIP"]
        if reply not in replies:
            raise DataError(f"CLIENT REPLY must be one of {replies!r}")
        return self.execute_command("CLIENT REPLY", reply, **kwargs)

    def client_id(self, **kwargs) -> ResponseT:
        """
        Returns the current connection id

        For more information see https://redis.io/commands/client-id
        """
        return self.execute_command("CLIENT ID", **kwargs)

    def client_tracking_on(
        self,
        clientid: Union[int, None] = None,
        prefix: Sequence[KeyT] = [],
        bcast: bool = False,
        optin: bool = False,
        optout: bool = False,
        noloop: bool = False,
    ) -> ResponseT:
        """
        Turn on the tracking mode.
        For more information about the options look at client_tracking func.

        See https://redis.io/commands/client-tracking
        """
        return self.client_tracking(
            True, clientid, prefix, bcast, optin, optout, noloop
        )

    def client_tracking_off(
        self,
        clientid: Union[int, None] = None,
        prefix: Sequence[KeyT] = [],
        bcast: bool = False,
        optin: bool = False,
        optout: bool = False,
        noloop: bool = False,
    ) -> ResponseT:
        """
        Turn off the tracking mode.
        For more information about the options look at client_tracking func.

        See https://redis.io/commands/client-tracking
        """
        return self.client_tracking(
            False, clientid, prefix, bcast, optin, optout, noloop
        )

    def client_tracking(
        self,
        on: bool = True,
        clientid: Union[int, None] = None,
        prefix: Sequence[KeyT] = [],
        bcast: bool = False,
        optin: bool = False,
        optout: bool = False,
        noloop: bool = False,
        **kwargs,
    ) -> ResponseT:
        """
        Enables the tracking feature of the Redis server, that is used
        for server assisted client side caching.

        ``on`` indicate for tracking on or tracking off. The dafualt is on.

        ``clientid`` send invalidation messages to the connection with
        the specified ID.

        ``bcast`` enable tracking in broadcasting mode. In this mode
        invalidation messages are reported for all the prefixes
        specified, regardless of the keys requested by the connection.

        ``optin``  when broadcasting is NOT active, normally don't track
        keys in read only commands, unless they are called immediately
        after a CLIENT CACHING yes command.

        ``optout`` when broadcasting is NOT active, normally track keys in
        read only commands, unless they are called immediately after a
        CLIENT CACHING no command.

        ``noloop`` don't send notifications about keys modified by this
        connection itself.

        ``prefix``  for broadcasting, register a given key prefix, so that
        notifications will be provided only for keys starting with this string.

        See https://redis.io/commands/client-tracking
        """

        if len(prefix) != 0 and bcast is False:
            raise DataError("Prefix can only be used with bcast")

        pieces = ["ON"] if on else ["OFF"]
        if clientid is not None:
            pieces.extend(["REDIRECT", clientid])
        for p in prefix:
            pieces.extend(["PREFIX", p])
        if bcast:
            pieces.append("BCAST")
        if optin:
            pieces.append("OPTIN")
        if optout:
            pieces.append("OPTOUT")
        if noloop:
            pieces.append("NOLOOP")

        return self.execute_command("CLIENT TRACKING", *pieces)

    def client_trackinginfo(self, **kwargs) -> ResponseT:
        """
        Returns the information about the current client connection's
        use of the server assisted client side cache.

        See https://redis.io/commands/client-trackinginfo
        """
        return self.execute_command("CLIENT TRACKINGINFO", **kwargs)

    def client_setname(self, name: str, **kwargs) -> ResponseT:
        """
        Sets the current connection name

        For more information see https://redis.io/commands/client-setname

        .. note::
           This method sets client name only for **current** connection.

           If you want to set a common name for all connections managed
           by this client, use ``client_name`` constructor argument.
        """
        return self.execute_command("CLIENT SETNAME", name, **kwargs)

    def client_unblock(
        self, client_id: int, error: bool = False, **kwargs
    ) -> ResponseT:
        """
        Unblocks a connection by its client id.
        If ``error`` is True, unblocks the client with a special error message.
        If ``error`` is False (default), the client is unblocked using the
        regular timeout mechanism.

        For more information see https://redis.io/commands/client-unblock
        """
        args = ["CLIENT UNBLOCK", int(client_id)]
        if error:
            args.append(b"ERROR")
        return self.execute_command(*args, **kwargs)

    def client_pause(self, timeout: int, all: bool = True, **kwargs) -> ResponseT:
        """
        Suspend all the Redis clients for the specified amount of time
        :param timeout: milliseconds to pause clients

        For more information see https://redis.io/commands/client-pause
        :param all: If true (default) all client commands are blocked.
             otherwise, clients are only blocked if they attempt to execute
             a write command.
             For the WRITE mode, some commands have special behavior:
                 EVAL/EVALSHA: Will block client for all scripts.
                 PUBLISH: Will block client.
                 PFCOUNT: Will block client.
                 WAIT: Acknowledgments will be delayed, so this command will
                 appear blocked.
        """
        args = ["CLIENT PAUSE", str(timeout)]
        if not isinstance(timeout, int):
            raise DataError("CLIENT PAUSE timeout must be an integer")
        if not all:
            args.append("WRITE")
        return self.execute_command(*args, **kwargs)

    def client_unpause(self, **kwargs) -> ResponseT:
        """
        Unpause all redis clients

        For more information see https://redis.io/commands/client-unpause
        """
        return self.execute_command("CLIENT UNPAUSE", **kwargs)

    def client_no_evict(self, mode: str) -> Union[Awaitable[str], str]:
        """
        Sets the client eviction mode for the current connection.

        For more information see https://redis.io/commands/client-no-evict
        """
        return self.execute_command("CLIENT NO-EVICT", mode)

    def command(self, **kwargs):
        """
        Returns dict reply of details about all Redis commands.

        For more information see https://redis.io/commands/command
        """
        return self.execute_command("COMMAND", **kwargs)

    def command_info(self, **kwargs) -> None:
        raise NotImplementedError(
            "COMMAND INFO is intentionally not implemented in the client."
        )

    def command_count(self, **kwargs) -> ResponseT:
        return self.execute_command("COMMAND COUNT", **kwargs)

    def command_list(
        self,
        module: Optional[str] = None,
        category: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> ResponseT:
        """
        Return an array of the server's command names.
        You can use one of the following filters:
        ``module``: get the commands that belong to the module
        ``category``: get the commands in the ACL category
        ``pattern``: get the commands that match the given pattern

        For more information see https://redis.io/commands/command-list/
        """
        pieces = []
        if module is not None:
            pieces.extend(["MODULE", module])
        if category is not None:
            pieces.extend(["ACLCAT", category])
        if pattern is not None:
            pieces.extend(["PATTERN", pattern])

        if pieces:
            pieces.insert(0, "FILTERBY")

        return self.execute_command("COMMAND LIST", *pieces)

    def command_getkeysandflags(self, *args: List[str]) -> List[Union[str, List[str]]]:
        """
        Returns array of keys from a full Redis command and their usage flags.

        For more information see https://redis.io/commands/command-getkeysandflags
        """
        return self.execute_command("COMMAND GETKEYSANDFLAGS", *args)

    def command_docs(self, *args):
        """
        This function throws a NotImplementedError since it is intentionally
        not supported.
        """
        raise NotImplementedError(
            "COMMAND DOCS is intentionally not implemented in the client."
        )

    def config_get(
        self, pattern: PatternT = "*", *args: List[PatternT], **kwargs
    ) -> ResponseT:
        """
        Return a dictionary of configuration based on the ``pattern``

        For more information see https://redis.io/commands/config-get
        """
        return self.execute_command("CONFIG GET", pattern, *args, **kwargs)

    def config_set(
        self,
        name: KeyT,
        value: EncodableT,
        *args: List[Union[KeyT, EncodableT]],
        **kwargs,
    ) -> ResponseT:
        """Set config item ``name`` with ``value``

        For more information see https://redis.io/commands/config-set
        """
        return self.execute_command("CONFIG SET", name, value, *args, **kwargs)

    def config_resetstat(self, **kwargs) -> ResponseT:
        """
        Reset runtime statistics

        For more information see https://redis.io/commands/config-resetstat
        """
        return self.execute_command("CONFIG RESETSTAT", **kwargs)

    def config_rewrite(self, **kwargs) -> ResponseT:
        """
        Rewrite config file with the minimal change to reflect running config.

        For more information see https://redis.io/commands/config-rewrite
        """
        return self.execute_command("CONFIG REWRITE", **kwargs)

    def dbsize(self, **kwargs) -> ResponseT:
        """
        Returns the number of keys in the current database

        For more information see https://redis.io/commands/dbsize
        """
        return self.execute_command("DBSIZE", **kwargs)

    def debug_object(self, key: KeyT, **kwargs) -> ResponseT:
        """
        Returns version specific meta information about a given key

        For more information see https://redis.io/commands/debug-object
        """
        return self.execute_command("DEBUG OBJECT", key, **kwargs)

    def debug_segfault(self, **kwargs) -> None:
        raise NotImplementedError(
            """
            DEBUG SEGFAULT is intentionally not implemented in the client.

            For more information see https://redis.io/commands/debug-segfault
            """
        )

    def echo(self, value: EncodableT, **kwargs) -> ResponseT:
        """
        Echo the string back from the server

        For more information see https://redis.io/commands/echo
        """
        return self.execute_command("ECHO", value, **kwargs)

    def flushall(self, asynchronous: bool = False, **kwargs) -> ResponseT:
        """
        Delete all keys in all databases on the current host.

        ``asynchronous`` indicates whether the operation is
        executed asynchronously by the server.

        For more information see https://redis.io/commands/flushall
        """
        args = []
        if asynchronous:
            args.append(b"ASYNC")
        return self.execute_command("FLUSHALL", *args, **kwargs)

    def flushdb(self, asynchronous: bool = False, **kwargs) -> ResponseT:
        """
        Delete all keys in the current database.

        ``asynchronous`` indicates whether the operation is
        executed asynchronously by the server.

        For more information see https://redis.io/commands/flushdb
        """
        args = []
        if asynchronous:
            args.append(b"ASYNC")
        return self.execute_command("FLUSHDB", *args, **kwargs)

    def sync(self) -> ResponseT:
        """
        Initiates a replication stream from the master.

        For more information see https://redis.io/commands/sync
        """
        from redis.client import NEVER_DECODE

        options = {}
        options[NEVER_DECODE] = []
        return self.execute_command("SYNC", **options)

    def psync(self, replicationid: str, offset: int):
        """
        Initiates a replication stream from the master.
        Newer version for `sync`.

        For more information see https://redis.io/commands/sync
        """
        from redis.client import NEVER_DECODE

        options = {}
        options[NEVER_DECODE] = []
        return self.execute_command("PSYNC", replicationid, offset, **options)

    def swapdb(self, first: int, second: int, **kwargs) -> ResponseT:
        """
        Swap two databases

        For more information see https://redis.io/commands/swapdb
        """
        return self.execute_command("SWAPDB", first, second, **kwargs)

    def select(self, index: int, **kwargs) -> ResponseT:
        """Select the Redis logical database at index.

        See: https://redis.io/commands/select
        """
        return self.execute_command("SELECT", index, **kwargs)

    def info(
        self, section: Union[str, None] = None, *args: List[str], **kwargs
    ) -> ResponseT:
        """
        Returns a dictionary containing information about the Redis server

        The ``section`` option can be used to select a specific section
        of information

        The section option is not supported by older versions of Redis Server,
        and will generate ResponseError

        For more information see https://redis.io/commands/info
        """
        if section is None:
            return self.execute_command("INFO", **kwargs)
        else:
            return self.execute_command("INFO", section, *args, **kwargs)

    def lastsave(self, **kwargs) -> ResponseT:
        """
        Return a Python datetime object representing the last time the
        Redis database was saved to disk

        For more information see https://redis.io/commands/lastsave
        """
        return self.execute_command("LASTSAVE", **kwargs)

    def latency_doctor(self):
        """Raise a NotImplementedError, as the client will not support LATENCY DOCTOR.
        This funcion is best used within the redis-cli.

        For more information see https://redis.io/commands/latency-doctor
        """
        raise NotImplementedError(
            """
            LATENCY DOCTOR is intentionally not implemented in the client.

            For more information see https://redis.io/commands/latency-doctor
            """
        )

    def latency_graph(self):
        """Raise a NotImplementedError, as the client will not support LATENCY GRAPH.
        This funcion is best used within the redis-cli.

        For more information see https://redis.io/commands/latency-graph.
        """
        raise NotImplementedError(
            """
            LATENCY GRAPH is intentionally not implemented in the client.

            For more information see https://redis.io/commands/latency-graph
            """
        )

    def lolwut(self, *version_numbers: Union[str, float], **kwargs) -> ResponseT:
        """
        Get the Redis version and a piece of generative computer art

        See: https://redis.io/commands/lolwut
        """
        if version_numbers:
            return self.execute_command("LOLWUT VERSION", *version_numbers, **kwargs)
        else:
            return self.execute_command("LOLWUT", **kwargs)

    def reset(self) -> ResponseT:
        """Perform a full reset on the connection's server side contenxt.

        See: https://redis.io/commands/reset
        """
        return self.execute_command("RESET")

    def migrate(
        self,
        host: str,
        port: int,
        keys: KeysT,
        destination_db: int,
        timeout: int,
        copy: bool = False,
        replace: bool = False,
        auth: Union[str, None] = None,
        **kwargs,
    ) -> ResponseT:
        """
        Migrate 1 or more keys from the current Redis server to a different
        server specified by the ``host``, ``port`` and ``destination_db``.

        The ``timeout``, specified in milliseconds, indicates the maximum
        time the connection between the two servers can be idle before the
        command is interrupted.

        If ``copy`` is True, the specified ``keys`` are NOT deleted from
        the source server.

        If ``replace`` is True, this operation will overwrite the keys
        on the destination server if they exist.

        If ``auth`` is specified, authenticate to the destination server with
        the password provided.

        For more information see https://redis.io/commands/migrate
        """
        keys = list_or_args(keys, [])
        if not keys:
            raise DataError("MIGRATE requires at least one key")
        pieces = []
        if copy:
            pieces.append(b"COPY")
        if replace:
            pieces.append(b"REPLACE")
        if auth:
            pieces.append(b"AUTH")
            pieces.append(auth)
        pieces.append(b"KEYS")
        pieces.extend(keys)
        return self.execute_command(
            "MIGRATE", host, port, "", destination_db, timeout, *pieces, **kwargs
        )

    def object(self, infotype: str, key: KeyT, **kwargs) -> ResponseT:
        """
        Return the encoding, idletime, or refcount about the key
        """
        return self.execute_command(
            "OBJECT", infotype, key, infotype=infotype, **kwargs
        )

    def memory_doctor(self, **kwargs) -> None:
        raise NotImplementedError(
            """
            MEMORY DOCTOR is intentionally not implemented in the client.

            For more information see https://redis.io/commands/memory-doctor
            """
        )

    def memory_help(self, **kwargs) -> None:
        raise NotImplementedError(
            """
            MEMORY HELP is intentionally not implemented in the client.

            For more information see https://redis.io/commands/memory-help
            """
        )

    def memory_stats(self, **kwargs) -> ResponseT:
        """
        Return a dictionary of memory stats

        For more information see https://redis.io/commands/memory-stats
        """
        return self.execute_command("MEMORY STATS", **kwargs)

    def memory_malloc_stats(self, **kwargs) -> ResponseT:
        """
        Return an internal statistics report from the memory allocator.

        See: https://redis.io/commands/memory-malloc-stats
        """
        return self.execute_command("MEMORY MALLOC-STATS", **kwargs)

    def memory_usage(
        self, key: KeyT, samples: Union[int, None] = None, **kwargs
    ) -> ResponseT:
        """
        Return the total memory usage for key, its value and associated
        administrative overheads.

        For nested data structures, ``samples`` is the number of elements to
        sample. If left unspecified, the server's default is 5. Use 0 to sample
        all elements.

        For more information see https://redis.io/commands/memory-usage
        """
        args = []
        if isinstance(samples, int):
            args.extend([b"SAMPLES", samples])
        return self.execute_command("MEMORY USAGE", key, *args, **kwargs)

    def memory_purge(self, **kwargs) -> ResponseT:
        """
        Attempts to purge dirty pages for reclamation by allocator

        For more information see https://redis.io/commands/memory-purge
        """
        return self.execute_command("MEMORY PURGE", **kwargs)

    def latency_histogram(self, *args):
        """
        This function throws a NotImplementedError since it is intentionally
        not supported.
        """
        raise NotImplementedError(
            "LATENCY HISTOGRAM is intentionally not implemented in the client."
        )

    def latency_history(self, event: str) -> ResponseT:
        """
        Returns the raw data of the ``event``'s latency spikes time series.

        For more information see https://redis.io/commands/latency-history
        """
        return self.execute_command("LATENCY HISTORY", event)

    def latency_latest(self) -> ResponseT:
        """
        Reports the latest latency events logged.

        For more information see https://redis.io/commands/latency-latest
        """
        return self.execute_command("LATENCY LATEST")

    def latency_reset(self, *events: str) -> ResponseT:
        """
        Resets the latency spikes time series of all, or only some, events.

        For more information see https://redis.io/commands/latency-reset
        """
        return self.execute_command("LATENCY RESET", *events)

    def ping(self, **kwargs) -> ResponseT:
        """
        Ping the Redis server

        For more information see https://redis.io/commands/ping
        """
        return self.execute_command("PING", **kwargs)

    def quit(self, **kwargs) -> ResponseT:
        """
        Ask the server to close the connection.

        For more information see https://redis.io/commands/quit
        """
        return self.execute_command("QUIT", **kwargs)

    def replicaof(self, *args, **kwargs) -> ResponseT:
        """
        Update the replication settings of a redis replica, on the fly.
        Examples of valid arguments include:
            NO ONE (set no replication)
            host port (set to the host and port of a redis server)

        For more information see  https://redis.io/commands/replicaof
        """
        return self.execute_command("REPLICAOF", *args, **kwargs)

    def save(self, **kwargs) -> ResponseT:
        """
        Tell the Redis server to save its data to disk,
        blocking until the save is complete

        For more information see https://redis.io/commands/save
        """
        return self.execute_command("SAVE", **kwargs)

    def shutdown(
        self,
        save: bool = False,
        nosave: bool = False,
        now: bool = False,
        force: bool = False,
        abort: bool = False,
        **kwargs,
    ) -> None:
        """Shutdown the Redis server.  If Redis has persistence configured,
        data will be flushed before shutdown.
        It is possible to specify modifiers to alter the behavior of the command:
        ``save`` will force a DB saving operation even if no save points are configured.
        ``nosave`` will prevent a DB saving operation even if one or more save points
        are configured.
        ``now`` skips waiting for lagging replicas, i.e. it bypasses the first step in
        the shutdown sequence.
        ``force`` ignores any errors that would normally prevent the server from exiting
        ``abort`` cancels an ongoing shutdown and cannot be combined with other flags.

        For more information see https://redis.io/commands/shutdown
        """
        if save and nosave:
            raise DataError("SHUTDOWN save and nosave cannot both be set")
        args = ["SHUTDOWN"]
        if save:
            args.append("SAVE")
        if nosave:
            args.append("NOSAVE")
        if now:
            args.append("NOW")
        if force:
            args.append("FORCE")
        if abort:
            args.append("ABORT")
        try:
            self.execute_command(*args, **kwargs)
        except ConnectionError:
            # a ConnectionError here is expected
            return
        raise RedisError("SHUTDOWN seems to have failed.")

    def slaveof(
        self, host: Union[str, None] = None, port: Union[int, None] = None, **kwargs
    ) -> ResponseT:
        """
        Set the server to be a replicated slave of the instance identified
        by the ``host`` and ``port``. If called without arguments, the
        instance is promoted to a master instead.

        For more information see https://redis.io/commands/slaveof
        """
        if host is None and port is None:
            return self.execute_command("SLAVEOF", b"NO", b"ONE", **kwargs)
        return self.execute_command("SLAVEOF", host, port, **kwargs)

    def slowlog_get(self, num: Union[int, None] = None, **kwargs) -> ResponseT:
        """
        Get the entries from the slowlog. If ``num`` is specified, get the
        most recent ``num`` items.

        For more information see https://redis.io/commands/slowlog-get
        """
        from redis.client import NEVER_DECODE

        args = ["SLOWLOG GET"]
        if num is not None:
            args.append(num)
        decode_responses = self.get_connection_kwargs().get("decode_responses", False)
        if decode_responses is True:
            kwargs[NEVER_DECODE] = []
        return self.execute_command(*args, **kwargs)

    def slowlog_len(self, **kwargs) -> ResponseT:
        """
        Get the number of items in the slowlog

        For more information see https://redis.io/commands/slowlog-len
        """
        return self.execute_command("SLOWLOG LEN", **kwargs)

    def slowlog_reset(self, **kwargs) -> ResponseT:
        """
        Remove all items in the slowlog

        For more information see https://redis.io/commands/slowlog-reset
        """
        return self.execute_command("SLOWLOG RESET", **kwargs)

    def time(self, **kwargs) -> ResponseT:
        """
        Returns the server time as a 2-item tuple of ints:
        (seconds since epoch, microseconds into this second).

        For more information see https://redis.io/commands/time
        """
        return self.execute_command("TIME", **kwargs)

    def wait(self, num_replicas: int, timeout: int, **kwargs) -> ResponseT:
        """
        Redis synchronous replication
        That returns the number of replicas that processed the query when
        we finally have at least ``num_replicas``, or when the ``timeout`` was
        reached.

        For more information see https://redis.io/commands/wait
        """
        return self.execute_command("WAIT", num_replicas, timeout, **kwargs)

    def hello(self):
        """
        This function throws a NotImplementedError since it is intentionally
        not supported.
        """
        raise NotImplementedError(
            "HELLO is intentionally not implemented in the client."
        )

    def failover(self):
        """
        This function throws a NotImplementedError since it is intentionally
        not supported.
        """
        raise NotImplementedError(
            "FAILOVER is intentionally not implemented in the client."
        )


AsyncManagementCommands = ManagementCommands


class AsyncManagementCommands(ManagementCommands):
    async def command_info(self, **kwargs) -> None:
        return super().command_info(**kwargs)

    async def debug_segfault(self, **kwargs) -> None:
        return super().debug_segfault(**kwargs)

    async def memory_doctor(self, **kwargs) -> None:
        return super().memory_doctor(**kwargs)

    async def memory_help(self, **kwargs) -> None:
        return super().memory_help(**kwargs)

    async def shutdown(
        self,
        save: bool = False,
        nosave: bool = False,
        now: bool = False,
        force: bool = False,
        abort: bool = False,
        **kwargs,
    ) -> None:
        """Shutdown the Redis server.  If Redis has persistence configured,
        data will be flushed before shutdown.  If the "save" option is set,
        a data flush will be attempted even if there is no persistence
        configured.  If the "nosave" option is set, no data flush will be
        attempted.  The "save" and "nosave" options cannot both be set.

        For more information see https://redis.io/commands/shutdown
        """
        if save and nosave:
            raise DataError("SHUTDOWN save and nosave cannot both be set")
        args = ["SHUTDOWN"]
        if save:
            args.append("SAVE")
        if nosave:
            args.append("NOSAVE")
        if now:
            args.append("NOW")
        if force:
            args.append("FORCE")
        if abort:
            args.append("ABORT")
        try:
            await self.execute_command(*args, **kwargs)
        except ConnectionError:
            # a ConnectionError here is expected
            return
        raise RedisError("SHUTDOWN seems to have failed.")


class BitFieldOperation:
    """
    Command builder for BITFIELD commands.
    """

    def __init__(
        self,
        client: Union["Redis", "AsyncRedis"],
        key: str,
        default_overflow: Union[str, None] = None,
    ):
        self.client = client
        self.key = key
        self._default_overflow = default_overflow
        # for typing purposes, run the following in constructor and in reset()
        self.operations: list[tuple[EncodableT, ...]] = []
        self._last_overflow = "WRAP"
        self.reset()

    def reset(self):
        """
        Reset the state of the instance to when it was constructed
        """
        self.operations = []
        self._last_overflow = "WRAP"
        self.overflow(self._default_overflow or self._last_overflow)

    def overflow(self, overflow: str):
        """
        Update the overflow algorithm of successive INCRBY operations
        :param overflow: Overflow algorithm, one of WRAP, SAT, FAIL. See the
            Redis docs for descriptions of these algorithmsself.
        :returns: a :py:class:`BitFieldOperation` instance.
        """
        overflow = overflow.upper()
        if overflow != self._last_overflow:
            self._last_overflow = overflow
            self.operations.append(("OVERFLOW", overflow))
        return self

    def incrby(
        self,
        fmt: str,
        offset: BitfieldOffsetT,
        increment: int,
        overflow: Union[str, None] = None,
    ):
        """
        Increment a bitfield by a given amount.
        :param fmt: format-string for the bitfield being updated, e.g. 'u8'
            for an unsigned 8-bit integer.
        :param offset: offset (in number of bits). If prefixed with a
            '#', this is an offset multiplier, e.g. given the arguments
            fmt='u8', offset='#2', the offset will be 16.
        :param int increment: value to increment the bitfield by.
        :param str overflow: overflow algorithm. Defaults to WRAP, but other
            acceptable values are SAT and FAIL. See the Redis docs for
            descriptions of these algorithms.
        :returns: a :py:class:`BitFieldOperation` instance.
        """
        if overflow is not None:
            self.overflow(overflow)

        self.operations.append(("INCRBY", fmt, offset, increment))
        return self

    def get(self, fmt: str, offset: BitfieldOffsetT):
        """
        Get the value of a given bitfield.
        :param fmt: format-string for the bitfield being read, e.g. 'u8' for
            an unsigned 8-bit integer.
        :param offset: offset (in number of bits). If prefixed with a
            '#', this is an offset multiplier, e.g. given the arguments
            fmt='u8', offset='#2', the offset will be 16.
        :returns: a :py:class:`BitFieldOperation` instance.
        """
        self.operations.append(("GET", fmt, offset))
        return self

    def set(self, fmt: str, offset: BitfieldOffsetT, value: int):
        """
        Set the value of a given bitfield.
        :param fmt: format-string for the bitfield being read, e.g. 'u8' for
            an unsigned 8-bit integer.
        :param offset: offset (in number of bits). If prefixed with a
            '#', this is an offset multiplier, e.g. given the arguments
            fmt='u8', offset='#2', the offset will be 16.
        :param int value: value to set at the given position.
        :returns: a :py:class:`BitFieldOperation` instance.
        """
        self.operations.append(("SET", fmt, offset, value))
        return self

    @property
    def command(self):
        cmd = ["BITFIELD", self.key]
        for ops in self.operations:
            cmd.extend(ops)
        return cmd

    def execute(self) -> ResponseT:
        """
        Execute the operation(s) in a single BITFIELD command. The return value
        is a list of values corresponding to each operation. If the client
        used to create this instance was a pipeline, the list of values
        will be present within the pipeline's execute.
        """
        command = self.command
        self.reset()
        return self.client.execute_command(*command)


class BasicKeyCommands(CommandsProtocol):
    """
    Redis basic key-based commands
    """

    def append(self, key: KeyT, value: EncodableT) -> ResponseT:
        """
        Appends the string ``value`` to the value at ``key``. If ``key``
        doesn't already exist, create it with a value of ``value``.
        Returns the new length of the value at ``key``.

        For more information see https://redis.io/commands/append
        """
        return self.execute_command("APPEND", key, value)

    def bitcount(
        self,
        key: KeyT,
        start: Union[int, None] = None,
        end: Union[int, None] = None,
        mode: Optional[str] = None,
    ) -> ResponseT:
        """
        Returns the count of set bits in the value of ``key``.  Optional
        ``start`` and ``end`` parameters indicate which bytes to consider

        For more information see https://redis.io/commands/bitcount
        """
        params = [key]
        if start is not None and end is not None:
            params.append(start)
            params.append(end)
        elif (start is not None and end is None) or (end is not None and start is None):
            raise DataError("Both start and end must be specified")
        if mode is not None:
            params.append(mode)
        return self.execute_command("BITCOUNT", *params)

    def bitfield(
        self: Union["Redis", "AsyncRedis"],
        key: KeyT,
        default_overflow: Union[str, None] = None,
    ) -> BitFieldOperation:
        """
        Return a BitFieldOperation instance to conveniently construct one or
        more bitfield operations on ``key``.

        For more information see https://redis.io/commands/bitfield
        """
        return BitFieldOperation(self, key, default_overflow=default_overflow)

    def bitfield_ro(
        self: Union["Redis", "AsyncRedis"],
        key: KeyT,
        encoding: str,
        offset: BitfieldOffsetT,
        items: Optional[list] = None,
    ) -> ResponseT:
        """
        Return an array of the specified bitfield values
        where the first value is found using ``encoding`` and ``offset``
        parameters and remaining values are result of corresponding
        encoding/offset pairs in optional list ``items``
        Read-only variant of the BITFIELD command.

        For more information see https://redis.io/commands/bitfield_ro
        """
        params = [key, "GET", encoding, offset]

        items = items or []
        for encoding, offset in items:
            params.extend(["GET", encoding, offset])
        return self.execute_command("BITFIELD_RO", *params)

    def bitop(self, operation: str, dest: KeyT, *keys: KeyT) -> ResponseT:
        """
        Perform a bitwise operation using ``operation`` between ``keys`` and
        store the result in ``dest``.

        For more information see https://redis.io/commands/bitop
        """
        return self.execute_command("BITOP", operation, dest, *keys)

    def bitpos(
        self,
        key: KeyT,
        bit: int,
        start: Union[int, None] = None,
        end: Union[int, None] = None,
        mode: Optional[str] = None,
    ) -> ResponseT:
        """
        Return the position of the first bit set to 1 or 0 in a string.
        ``start`` and ``end`` defines search range. The range is interpreted
        as a range of bytes and not a range of bits, so start=0 and end=2
        means to look at the first three bytes.

        For more information see https://redis.io/commands/bitpos
        """
        if bit not in (0, 1):
            raise DataError("bit must be 0 or 1")
        params = [key, bit]

        start is not None and params.append(start)

        if start is not None and end is not None:
            params.append(end)
        elif start is None and end is not None:
            raise DataError("start argument is not set, when end is specified")

        if mode is not None:
            params.append(mode)
        return self.execute_command("BITPOS", *params)

    def copy(
        self,
        source: str,
        destination: str,
        destination_db: Union[str, None] = None,
        replace: bool = False,
    ) -> ResponseT:
        """
        Copy the value stored in the ``source`` key to the ``destination`` key.

        ``destination_db`` an alternative destination database. By default,
        the ``destination`` key is created in the source Redis database.

        ``replace`` whether the ``destination`` key should be removed before
        copying the value to it. By default, the value is not copied if
        the ``destination`` key already exists.

        For more information see https://redis.io/commands/copy
        """
        params = [source, destination]
        if destination_db is not None:
            params.extend(["DB", destination_db])
        if replace:
            params.append("REPLACE")
        return self.execute_command("COPY", *params)

    def decrby(self, name: KeyT, amount: int = 1) -> ResponseT:
        """
        Decrements the value of ``key`` by ``amount``.  If no key exists,
        the value will be initialized as 0 - ``amount``

        For more information see https://redis.io/commands/decrby
        """
        return self.execute_command("DECRBY", name, amount)

    decr = decrby

    def delete(self, *names: KeyT) -> ResponseT:
        """
        Delete one or more keys specified by ``names``
        """
        return self.execute_command("DEL", *names)

    def __delitem__(self, name: KeyT):
        self.delete(name)

    def dump(self, name: KeyT) -> ResponseT:
        """
        Return a serialized version of the value stored at the specified key.
        If key does not exist a nil bulk reply is returned.

        For more information see https://redis.io/commands/dump
        """
        from redis.client import NEVER_DECODE

        options = {}
        options[NEVER_DECODE] = []
        return self.execute_command("DUMP", name, **options)

    def exists(self, *names: KeyT) -> ResponseT:
        """
        Returns the number of ``names`` that exist

        For more information see https://redis.io/commands/exists
        """
        return self.execute_command("EXISTS", *names)

    __contains__ = exists

    def expire(
        self,
        name: KeyT,
        time: ExpiryT,
        nx: bool = False,
        xx: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> ResponseT:
        """
        Set an expire flag on key ``name`` for ``time`` seconds with given
        ``option``. ``time`` can be represented by an integer or a Python timedelta
        object.

        Valid options are:
            NX -> Set expiry only when the key has no expiry
            XX -> Set expiry only when the key has an existing expiry
            GT -> Set expiry only when the new expiry is greater than current one
            LT -> Set expiry only when the new expiry is less than current one

        For more information see https://redis.io/commands/expire
        """
        if isinstance(time, datetime.timedelta):
            time = int(time.total_seconds())

        exp_option = list()
        if nx:
            exp_option.append("NX")
        if xx:
            exp_option.append("XX")
        if gt:
            exp_option.append("GT")
        if lt:
            exp_option.append("LT")

        return self.execute_command("EXPIRE", name, time, *exp_option)

    def expireat(
        self,
        name: KeyT,
        when: AbsExpiryT,
        nx: bool = False,
        xx: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> ResponseT:
        """
        Set an expire flag on key ``name`` with given ``option``. ``when``
        can be represented as an integer indicating unix time or a Python
        datetime object.

        Valid options are:
            -> NX -- Set expiry only when the key has no expiry
            -> XX -- Set expiry only when the key has an existing expiry
            -> GT -- Set expiry only when the new expiry is greater than current one
            -> LT -- Set expiry only when the new expiry is less than current one

        For more information see https://redis.io/commands/expireat
        """
        if isinstance(when, datetime.datetime):
            when = int(when.timestamp())

        exp_option = list()
        if nx:
            exp_option.append("NX")
        if xx:
            exp_option.append("XX")
        if gt:
            exp_option.append("GT")
        if lt:
            exp_option.append("LT")

        return self.execute_command("EXPIREAT", name, when, *exp_option)

    def expiretime(self, key: str) -> int:
        """
        Returns the absolute Unix timestamp (since January 1, 1970) in seconds
        at which the given key will expire.

        For more information see https://redis.io/commands/expiretime
        """
        return self.execute_command("EXPIRETIME", key)

    def get(self, name: KeyT) -> ResponseT:
        """
        Return the value at key ``name``, or None if the key doesn't exist

        For more information see https://redis.io/commands/get
        """
        return self.execute_command("GET", name)

    def getdel(self, name: KeyT) -> ResponseT:
        """
        Get the value at key ``name`` and delete the key. This command
        is similar to GET, except for the fact that it also deletes
        the key on success (if and only if the key's value type
        is a string).

        For more information see https://redis.io/commands/getdel
        """
        return self.execute_command("GETDEL", name)

    def getex(
        self,
        name: KeyT,
        ex: Union[ExpiryT, None] = None,
        px: Union[ExpiryT, None] = None,
        exat: Union[AbsExpiryT, None] = None,
        pxat: Union[AbsExpiryT, None] = None,
        persist: bool = False,
    ) -> ResponseT:
        """
        Get the value of key and optionally set its expiration.
        GETEX is similar to GET, but is a write command with
        additional options. All time parameters can be given as
        datetime.timedelta or integers.

        ``ex`` sets an expire flag on key ``name`` for ``ex`` seconds.

        ``px`` sets an expire flag on key ``name`` for ``px`` milliseconds.

        ``exat`` sets an expire flag on key ``name`` for ``ex`` seconds,
        specified in unix time.

        ``pxat`` sets an expire flag on key ``name`` for ``ex`` milliseconds,
        specified in unix time.

        ``persist`` remove the time to live associated with ``name``.

        For more information see https://redis.io/commands/getex
        """

        opset = {ex, px, exat, pxat}
        if len(opset) > 2 or len(opset) > 1 and persist:
            raise DataError(
                "``ex``, ``px``, ``exat``, ``pxat``, "
                "and ``persist`` are mutually exclusive."
            )

        pieces: list[EncodableT] = []
        # similar to set command
        if ex is not None:
            pieces.append("EX")
            if isinstance(ex, datetime.timedelta):
                ex = int(ex.total_seconds())
            pieces.append(ex)
        if px is not None:
            pieces.append("PX")
            if isinstance(px, datetime.timedelta):
                px = int(px.total_seconds() * 1000)
            pieces.append(px)
        # similar to pexpireat command
        if exat is not None:
            pieces.append("EXAT")
            if isinstance(exat, datetime.datetime):
                exat = int(exat.timestamp())
            pieces.append(exat)
        if pxat is not None:
            pieces.append("PXAT")
            if isinstance(pxat, datetime.datetime):
                pxat = int(pxat.timestamp() * 1000)
            pieces.append(pxat)
        if persist:
            pieces.append("PERSIST")

        return self.execute_command("GETEX", name, *pieces)

    def __getitem__(self, name: KeyT):
        """
        Return the value at key ``name``, raises a KeyError if the key
        doesn't exist.
        """
        value = self.get(name)
        if value is not None:
            return value
        raise KeyError(name)

    def getbit(self, name: KeyT, offset: int) -> ResponseT:
        """
        Returns an integer indicating the value of ``offset`` in ``name``

        For more information see https://redis.io/commands/getbit
        """
        return self.execute_command("GETBIT", name, offset)

    def getrange(self, key: KeyT, start: int, end: int) -> ResponseT:
        """
        Returns the substring of the string value stored at ``key``,
        determined by the offsets ``start`` and ``end`` (both are inclusive)

        For more information see https://redis.io/commands/getrange
        """
        return self.execute_command("GETRANGE", key, start, end)

    def getset(self, name: KeyT, value: EncodableT) -> ResponseT:
        """
        Sets the value at key ``name`` to ``value``
        and returns the old value at key ``name`` atomically.

        As per Redis 6.2, GETSET is considered deprecated.
        Please use SET with GET parameter in new code.

        For more information see https://redis.io/commands/getset
        """
        return self.execute_command("GETSET", name, value)

    def incrby(self, name: KeyT, amount: int = 1) -> ResponseT:
        """
        Increments the value of ``key`` by ``amount``.  If no key exists,
        the value will be initialized as ``amount``

        For more information see https://redis.io/commands/incrby
        """
        return self.execute_command("INCRBY", name, amount)

    incr = incrby

    def incrbyfloat(self, name: KeyT, amount: float = 1.0) -> ResponseT:
        """
        Increments the value at key ``name`` by floating ``amount``.
        If no key exists, the value will be initialized as ``amount``

        For more information see https://redis.io/commands/incrbyfloat
        """
        return self.execute_command("INCRBYFLOAT", name, amount)

    def keys(self, pattern: PatternT = "*", **kwargs) -> ResponseT:
        """
        Returns a list of keys matching ``pattern``

        For more information see https://redis.io/commands/keys
        """
        return self.execute_command("KEYS", pattern, **kwargs)

    def lmove(
        self, first_list: str, second_list: str, src: str = "LEFT", dest: str = "RIGHT"
    ) -> ResponseT:
        """
        Atomically returns and removes the first/last element of a list,
        pushing it as the first/last element on the destination list.
        Returns the element being popped and pushed.

        For more information see https://redis.io/commands/lmove
        """
        params = [first_list, second_list, src, dest]
        return self.execute_command("LMOVE", *params)

    def blmove(
        self,
        first_list: str,
        second_list: str,
        timeout: int,
        src: str = "LEFT",
        dest: str = "RIGHT",
    ) -> ResponseT:
        """
        Blocking version of lmove.

        For more information see https://redis.io/commands/blmove
        """
        params = [first_list, second_list, src, dest, timeout]
        return self.execute_command("BLMOVE", *params)

    def mget(self, keys: KeysT, *args: EncodableT) -> ResponseT:
        """
        Returns a list of values ordered identically to ``keys``

        For more information see https://redis.io/commands/mget
        """
        from redis.client import EMPTY_RESPONSE

        args = list_or_args(keys, args)
        options = {}
        if not args:
            options[EMPTY_RESPONSE] = []
        return self.execute_command("MGET", *args, **options)

    def mset(self, mapping: Mapping[AnyKeyT, EncodableT]) -> ResponseT:
        """
        Sets key/values based on a mapping. Mapping is a dictionary of
        key/value pairs. Both keys and values should be strings or types that
        can be cast to a string via str().

        For more information see https://redis.io/commands/mset
        """
        items = []
        for pair in mapping.items():
            items.extend(pair)
        return self.execute_command("MSET", *items)

    def msetnx(self, mapping: Mapping[AnyKeyT, EncodableT]) -> ResponseT:
        """
        Sets key/values based on a mapping if none of the keys are already set.
        Mapping is a dictionary of key/value pairs. Both keys and values
        should be strings or types that can be cast to a string via str().
        Returns a boolean indicating if the operation was successful.

        For more information see https://redis.io/commands/msetnx
        """
        items = []
        for pair in mapping.items():
            items.extend(pair)
        return self.execute_command("MSETNX", *items)

    def move(self, name: KeyT, db: int) -> ResponseT:
        """
        Moves the key ``name`` to a different Redis database ``db``

        For more information see https://redis.io/commands/move
        """
        return self.execute_command("MOVE", name, db)

    def persist(self, name: KeyT) -> ResponseT:
        """
        Removes an expiration on ``name``

        For more information see https://redis.io/commands/persist
        """
        return self.execute_command("PERSIST", name)

    def pexpire(
        self,
        name: KeyT,
        time: ExpiryT,
        nx: bool = False,
        xx: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> ResponseT:
        """
        Set an expire flag on key ``name`` for ``time`` milliseconds
        with given ``option``. ``time`` can be represented by an
        integer or a Python timedelta object.

        Valid options are:
            NX -> Set expiry only when the key has no expiry
            XX -> Set expiry only when the key has an existing expiry
            GT -> Set expiry only when the new expiry is greater than current one
            LT -> Set expiry only when the new expiry is less than current one

        For more information see https://redis.io/commands/pexpire
        """
        if isinstance(time, datetime.timedelta):
            time = int(time.total_seconds() * 1000)

        exp_option = list()
        if nx:
            exp_option.append("NX")
        if xx:
            exp_option.append("XX")
        if gt:
            exp_option.append("GT")
        if lt:
            exp_option.append("LT")
        return self.execute_command("PEXPIRE", name, time, *exp_option)

    def pexpireat(
        self,
        name: KeyT,
        when: AbsExpiryT,
        nx: bool = False,
        xx: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> ResponseT:
        """
        Set an expire flag on key ``name`` with given ``option``. ``when``
        can be represented as an integer representing unix time in
        milliseconds (unix time * 1000) or a Python datetime object.

        Valid options are:
            NX -> Set expiry only when the key has no expiry
            XX -> Set expiry only when the key has an existing expiry
            GT -> Set expiry only when the new expiry is greater than current one
            LT -> Set expiry only when the new expiry is less than current one

        For more information see https://redis.io/commands/pexpireat
        """
        if isinstance(when, datetime.datetime):
            when = int(when.timestamp() * 1000)
        exp_option = list()
        if nx:
            exp_option.append("NX")
        if xx:
            exp_option.append("XX")
        if gt:
            exp_option.append("GT")
        if lt:
            exp_option.append("LT")
        return self.execute_command("PEXPIREAT", name, when, *exp_option)

    def pexpiretime(self, key: str) -> int:
        """
        Returns the absolute Unix timestamp (since January 1, 1970) in milliseconds
        at which the given key will expire.

        For more information see https://redis.io/commands/pexpiretime
        """
        return self.execute_command("PEXPIRETIME", key)

    def psetex(self, name: KeyT, time_ms: ExpiryT, value: EncodableT):
        """
        Set the value of key ``name`` to ``value`` that expires in ``time_ms``
        milliseconds. ``time_ms`` can be represented by an integer or a Python
        timedelta object

        For more information see https://redis.io/commands/psetex
        """
        if isinstance(time_ms, datetime.timedelta):
            time_ms = int(time_ms.total_seconds() * 1000)
        return self.execute_command("PSETEX", name, time_ms, value)

    def pttl(self, name: KeyT) -> ResponseT:
        """
        Returns the number of milliseconds until the key ``name`` will expire

        For more information see https://redis.io/commands/pttl
        """
        return self.execute_command("PTTL", name)

    def hrandfield(
        self, key: str, count: int = None, withvalues: bool = False
    ) -> ResponseT:
        """
        Return a random field from the hash value stored at key.

        count: if the argument is positive, return an array of distinct fields.
        If called with a negative count, the behavior changes and the command
        is allowed to return the same field multiple times. In this case,
        the number of returned fields is the absolute value of the
        specified count.
        withvalues: The optional WITHVALUES modifier changes the reply so it
        includes the respective values of the randomly selected hash fields.

        For more information see https://redis.io/commands/hrandfield
        """
        params = []
        if count is not None:
            params.append(count)
        if withvalues:
            params.append("WITHVALUES")

        return self.execute_command("HRANDFIELD", key, *params)

    def randomkey(self, **kwargs) -> ResponseT:
        """
        Returns the name of a random key

        For more information see https://redis.io/commands/randomkey
        """
        return self.execute_command("RANDOMKEY", **kwargs)

    def rename(self, src: KeyT, dst: KeyT) -> ResponseT:
        """
        Rename key ``src`` to ``dst``

        For more information see https://redis.io/commands/rename
        """
        return self.execute_command("RENAME", src, dst)

    def renamenx(self, src: KeyT, dst: KeyT):
        """
        Rename key ``src`` to ``dst`` if ``dst`` doesn't already exist

        For more information see https://redis.io/commands/renamenx
        """
        return self.execute_command("RENAMENX", src, dst)

    def restore(
        self,
        name: KeyT,
        ttl: float,
        value: EncodableT,
        replace: bool = False,
        absttl: bool = False,
        idletime: Union[int, None] = None,
        frequency: Union[int, None] = None,
    ) -> ResponseT:
        """
        Create a key using the provided serialized value, previously obtained
        using DUMP.

        ``replace`` allows an existing key on ``name`` to be overridden. If
        it's not specified an error is raised on collision.

        ``absttl`` if True, specified ``ttl`` should represent an absolute Unix
        timestamp in milliseconds in which the key will expire. (Redis 5.0 or
        greater).

        ``idletime`` Used for eviction, this is the number of seconds the
        key must be idle, prior to execution.

        ``frequency`` Used for eviction, this is the frequency counter of
        the object stored at the key, prior to execution.

        For more information see https://redis.io/commands/restore
        """
        params = [name, ttl, value]
        if replace:
            params.append("REPLACE")
        if absttl:
            params.append("ABSTTL")
        if idletime is not None:
            params.append("IDLETIME")
            try:
                params.append(int(idletime))
            except ValueError:
                raise DataError("idletimemust be an integer")

        if frequency is not None:
            params.append("FREQ")
            try:
                params.append(int(frequency))
            except ValueError:
                raise DataError("frequency must be an integer")

        return self.execute_command("RESTORE", *params)

    def set(
        self,
        name: KeyT,
        value: EncodableT,
        ex: Union[ExpiryT, None] = None,
        px: Union[ExpiryT, None] = None,
        nx: bool = False,
        xx: bool = False,
        keepttl: bool = False,
        get: bool = False,
        exat: Union[AbsExpiryT, None] = None,
        pxat: Union[AbsExpiryT, None] = None,
    ) -> ResponseT:
        """
        Set the value at key ``name`` to ``value``

        ``ex`` sets an expire flag on key ``name`` for ``ex`` seconds.

        ``px`` sets an expire flag on key ``name`` for ``px`` milliseconds.

        ``nx`` if set to True, set the value at key ``name`` to ``value`` only
            if it does not exist.

        ``xx`` if set to True, set the value at key ``name`` to ``value`` only
            if it already exists.

        ``keepttl`` if True, retain the time to live associated with the key.
            (Available since Redis 6.0)

        ``get`` if True, set the value at key ``name`` to ``value`` and return
            the old value stored at key, or None if the key did not exist.
            (Available since Redis 6.2)

        ``exat`` sets an expire flag on key ``name`` for ``ex`` seconds,
            specified in unix time.

        ``pxat`` sets an expire flag on key ``name`` for ``ex`` milliseconds,
            specified in unix time.

        For more information see https://redis.io/commands/set
        """
        pieces: list[EncodableT] = [name, value]
        options = {}
        if ex is not None:
            pieces.append("EX")
            if isinstance(ex, datetime.timedelta):
                pieces.append(int(ex.total_seconds()))
            elif isinstance(ex, int):
                pieces.append(ex)
            elif isinstance(ex, str) and ex.isdigit():
                pieces.append(int(ex))
            else:
                raise DataError("ex must be datetime.timedelta or int")
        if px is not None:
            pieces.append("PX")
            if isinstance(px, datetime.timedelta):
                pieces.append(int(px.total_seconds() * 1000))
            elif isinstance(px, int):
                pieces.append(px)
            else:
                raise DataError("px must be datetime.timedelta or int")
        if exat is not None:
            pieces.append("EXAT")
            if isinstance(exat, datetime.datetime):
                exat = int(exat.timestamp())
            pieces.append(exat)
        if pxat is not None:
            pieces.append("PXAT")
            if isinstance(pxat, datetime.datetime):
                pxat = int(pxat.timestamp() * 1000)
            pieces.append(pxat)
        if keepttl:
            pieces.append("KEEPTTL")

        if nx:
            pieces.append("NX")
        if xx:
            pieces.append("XX")

        if get:
            pieces.append("GET")
            options["get"] = True

        return self.execute_command("SET", *pieces, **options)

    def __setitem__(self, name: KeyT, value: EncodableT):
        self.set(name, value)

    def setbit(self, name: KeyT, offset: int, value: int) -> ResponseT:
        """
        Flag the ``offset`` in ``name`` as ``value``. Returns an integer
        indicating the previous value of ``offset``.

        For more information see https://redis.io/commands/setbit
        """
        value = value and 1 or 0
        return self.execute_command("SETBIT", name, offset, value)

    def setex(self, name: KeyT, time: ExpiryT, value: EncodableT) -> ResponseT:
        """
        Set the value of key ``name`` to ``value`` that expires in ``time``
        seconds. ``time`` can be represented by an integer or a Python
        timedelta object.

        For more information see https://redis.io/commands/setex
        """
        if isinstance(time, datetime.timedelta):
            time = int(time.total_seconds())
        return self.execute_command("SETEX", name, time, value)

    def setnx(self, name: KeyT, value: EncodableT) -> ResponseT:
        """
        Set the value of key ``name`` to ``value`` if key doesn't exist

        For more information see https://redis.io/commands/setnx
        """
        return self.execute_command("SETNX", name, value)

    def setrange(self, name: KeyT, offset: int, value: EncodableT) -> ResponseT:
        """
        Overwrite bytes in the value of ``name`` starting at ``offset`` with
        ``value``. If ``offset`` plus the length of ``value`` exceeds the
        length of the original value, the new value will be larger than before.
        If ``offset`` exceeds the length of the original value, null bytes
        will be used to pad between the end of the previous value and the start
        of what's being injected.

        Returns the length of the new string.

        For more information see https://redis.io/commands/setrange
        """
        return self.execute_command("SETRANGE", name, offset, value)

    def stralgo(
        self,
        algo: Literal["LCS"],
        value1: KeyT,
        value2: KeyT,
        specific_argument: Union[Literal["strings"], Literal["keys"]] = "strings",
        len: bool = False,
        idx: bool = False,
        minmatchlen: Union[int, None] = None,
        withmatchlen: bool = False,
        **kwargs,
    ) -> ResponseT:
        """
        Implements complex algorithms that operate on strings.
        Right now the only algorithm implemented is the LCS algorithm
        (longest common substring). However new algorithms could be
        implemented in the future.

        ``algo`` Right now must be LCS
        ``value1`` and ``value2`` Can be two strings or two keys
        ``specific_argument`` Specifying if the arguments to the algorithm
        will be keys or strings. strings is the default.
        ``len`` Returns just the len of the match.
        ``idx`` Returns the match positions in each string.
        ``minmatchlen`` Restrict the list of matches to the ones of a given
        minimal length. Can be provided only when ``idx`` set to True.
        ``withmatchlen`` Returns the matches with the len of the match.
        Can be provided only when ``idx`` set to True.

        For more information see https://redis.io/commands/stralgo
        """
        # check validity
        supported_algo = ["LCS"]
        if algo not in supported_algo:
            supported_algos_str = ", ".join(supported_algo)
            raise DataError(f"The supported algorithms are: {supported_algos_str}")
        if specific_argument not in ["keys", "strings"]:
            raise DataError("specific_argument can be only keys or strings")
        if len and idx:
            raise DataError("len and idx cannot be provided together.")

        pieces: list[EncodableT] = [algo, specific_argument.upper(), value1, value2]
        if len:
            pieces.append(b"LEN")
        if idx:
            pieces.append(b"IDX")
        try:
            int(minmatchlen)
            pieces.extend([b"MINMATCHLEN", minmatchlen])
        except TypeError:
            pass
        if withmatchlen:
            pieces.append(b"WITHMATCHLEN")

        return self.execute_command(
            "STRALGO",
            *pieces,
            len=len,
            idx=idx,
            minmatchlen=minmatchlen,
            withmatchlen=withmatchlen,
            **kwargs,
        )

    def strlen(self, name: KeyT) -> ResponseT:
        """
        Return the number of bytes stored in the value of ``name``

        For more information see https://redis.io/commands/strlen
        """
        return self.execute_command("STRLEN", name)

    def substr(self, name: KeyT, start: int, end: int = -1) -> ResponseT:
        """
        Return a substring of the string at key ``name``. ``start`` and ``end``
        are 0-based integers specifying the portion of the string to return.
        """
        return self.execute_command("SUBSTR", name, start, end)

    def touch(self, *args: KeyT) -> ResponseT:
        """
        Alters the last access time of a key(s) ``*args``. A key is ignored
        if it does not exist.

        For more information see https://redis.io/commands/touch
        """
        return self.execute_command("TOUCH", *args)

    def ttl(self, name: KeyT) -> ResponseT:
        """
        Returns the number of seconds until the key ``name`` will expire

        For more information see https://redis.io/commands/ttl
        """
        return self.execute_command("TTL", name)

    def type(self, name: KeyT) -> ResponseT:
        """
        Returns the type of key ``name``

        For more information see https://redis.io/commands/type
        """
        return self.execute_command("TYPE", name)

    def watch(self, *names: KeyT) -> None:
        """
        Watches the values at keys ``names``, or None if the key doesn't exist

        For more information see https://redis.io/commands/watch
        """
        warnings.warn(DeprecationWarning("Call WATCH from a Pipeline object"))

    def unwatch(self) -> None:
        """
        Unwatches the value at key ``name``, or None of the key doesn't exist

        For more information see https://redis.io/commands/unwatch
        """
        warnings.warn(DeprecationWarning("Call UNWATCH from a Pipeline object"))

    def unlink(self, *names: KeyT) -> ResponseT:
        """
        Unlink one or more keys specified by ``names``

        For more information see https://redis.io/commands/unlink
        """
        return self.execute_command("UNLINK", *names)

    def lcs(
        self,
        key1: str,
        key2: str,
        len: Optional[bool] = False,
        idx: Optional[bool] = False,
        minmatchlen: Optional[int] = 0,
        withmatchlen: Optional[bool] = False,
    ) -> Union[str, int, list]:
        """
        Find the longest common subsequence between ``key1`` and ``key2``.
        If ``len`` is true the length of the match will will be returned.
        If ``idx`` is true the match position in each strings will be returned.
        ``minmatchlen`` restrict the list of matches to the ones of
        the given ``minmatchlen``.
        If ``withmatchlen`` the length of the match also will be returned.
        For more information see https://redis.io/commands/lcs
        """
        pieces = [key1, key2]
        if len:
            pieces.append("LEN")
        if idx:
            pieces.append("IDX")
        if minmatchlen != 0:
            pieces.extend(["MINMATCHLEN", minmatchlen])
        if withmatchlen:
            pieces.append("WITHMATCHLEN")
        return self.execute_command("LCS", *pieces)


class AsyncBasicKeyCommands(BasicKeyCommands):
    def __delitem__(self, name: KeyT):
        raise TypeError("Async Redis client does not support class deletion")

    def __contains__(self, name: KeyT):
        raise TypeError("Async Redis client does not support class inclusion")

    def __getitem__(self, name: KeyT):
        raise TypeError("Async Redis client does not support class retrieval")

    def __setitem__(self, name: KeyT, value: EncodableT):
        raise TypeError("Async Redis client does not support class assignment")

    async def watch(self, *names: KeyT) -> None:
        return super().watch(*names)

    async def unwatch(self) -> None:
        return super().unwatch()


class ListCommands(CommandsProtocol):
    """
    Redis commands for List data type.
    see: https://redis.io/topics/data-types#lists
    """

    def blpop(
        self, keys: List, timeout: Optional[int] = 0
    ) -> Union[Awaitable[list], list]:
        """
        LPOP a value off of the first non-empty list
        named in the ``keys`` list.

        If none of the lists in ``keys`` has a value to LPOP, then block
        for ``timeout`` seconds, or until a value gets pushed on to one
        of the lists.

        If timeout is 0, then block indefinitely.

        For more information see https://redis.io/commands/blpop
        """
        if timeout is None:
            timeout = 0
        keys = list_or_args(keys, None)
        keys.append(timeout)
        return self.execute_command("BLPOP", *keys)

    def brpop(
        self, keys: List, timeout: Optional[int] = 0
    ) -> Union[Awaitable[list], list]:
        """
        RPOP a value off of the first non-empty list
        named in the ``keys`` list.

        If none of the lists in ``keys`` has a value to RPOP, then block
        for ``timeout`` seconds, or until a value gets pushed on to one
        of the lists.

        If timeout is 0, then block indefinitely.

        For more information see https://redis.io/commands/brpop
        """
        if timeout is None:
            timeout = 0
        keys = list_or_args(keys, None)
        keys.append(timeout)
        return self.execute_command("BRPOP", *keys)

    def brpoplpush(
        self, src: str, dst: str, timeout: Optional[int] = 0
    ) -> Union[Awaitable[Optional[str]], Optional[str]]:
        """
        Pop a value off the tail of ``src``, push it on the head of ``dst``
        and then return it.

        This command blocks until a value is in ``src`` or until ``timeout``
        seconds elapse, whichever is first. A ``timeout`` value of 0 blocks
        forever.

        For more information see https://redis.io/commands/brpoplpush
        """
        if timeout is None:
            timeout = 0
        return self.execute_command("BRPOPLPUSH", src, dst, timeout)

    def blmpop(
        self,
        timeout: float,
        numkeys: int,
        *args: List[str],
        direction: str,
        count: Optional[int] = 1,
    ) -> Optional[list]:
        """
        Pop ``count`` values (default 1) from first non-empty in the list
        of provided key names.

        When all lists are empty this command blocks the connection until another
        client pushes to it or until the timeout, timeout of 0 blocks indefinitely

        For more information see https://redis.io/commands/blmpop
        """
        args = [timeout, numkeys, *args, direction, "COUNT", count]

        return self.execute_command("BLMPOP", *args)

    def lmpop(
        self,
        num_keys: int,
        *args: List[str],
        direction: str,
        count: Optional[int] = 1,
    ) -> Union[Awaitable[list], list]:
        """
        Pop ``count`` values (default 1) first non-empty list key from the list
        of args provided key names.

        For more information see https://redis.io/commands/lmpop
        """
        args = [num_keys] + list(args) + [direction]
        if count != 1:
            args.extend(["COUNT", count])

        return self.execute_command("LMPOP", *args)

    def lindex(
        self, name: str, index: int
    ) -> Union[Awaitable[Optional[str]], Optional[str]]:
        """
        Return the item from list ``name`` at position ``index``

        Negative indexes are supported and will return an item at the
        end of the list

        For more information see https://redis.io/commands/lindex
        """
        return self.execute_command("LINDEX", name, index)

    def linsert(
        self, name: str, where: str, refvalue: str, value: str
    ) -> Union[Awaitable[int], int]:
        """
        Insert ``value`` in list ``name`` either immediately before or after
        [``where``] ``refvalue``

        Returns the new length of the list on success or -1 if ``refvalue``
        is not in the list.

        For more information see https://redis.io/commands/linsert
        """
        return self.execute_command("LINSERT", name, where, refvalue, value)

    def llen(self, name: str) -> Union[Awaitable[int], int]:
        """
        Return the length of the list ``name``

        For more information see https://redis.io/commands/llen
        """
        return self.execute_command("LLEN", name)

    def lpop(
        self,
        name: str,
        count: Optional[int] = None,
    ) -> Union[Awaitable[Union[str, List, None]], Union[str, List, None]]:
        """
        Removes and returns the first elements of the list ``name``.

        By default, the command pops a single element from the beginning of
        the list. When provided with the optional ``count`` argument, the reply
        will consist of up to count elements, depending on the list's length.

        For more information see https://redis.io/commands/lpop
        """
        if count is not None:
            return self.execute_command("LPOP", name, count)
        else:
            return self.execute_command("LPOP", name)

    def lpush(self, name: str, *values: FieldT) -> Union[Awaitable[int], int]:
        """
        Push ``values`` onto the head of the list ``name``

        For more information see https://redis.io/commands/lpush
        """
        return self.execute_command("LPUSH", name, *values)

    def lpushx(self, name: str, *values: FieldT) -> Union[Awaitable[int], int]:
        """
        Push ``value`` onto the head of the list ``name`` if ``name`` exists

        For more information see https://redis.io/commands/lpushx
        """
        return self.execute_command("LPUSHX", name, *values)

    def lrange(self, name: str, start: int, end: int) -> Union[Awaitable[list], list]:
        """
        Return a slice of the list ``name`` between
        position ``start`` and ``end``

        ``start`` and ``end`` can be negative numbers just like
        Python slicing notation

        For more information see https://redis.io/commands/lrange
        """
        return self.execute_command("LRANGE", name, start, end)

    def lrem(self, name: str, count: int, value: str) -> Union[Awaitable[int], int]:
        """
        Remove the first ``count`` occurrences of elements equal to ``value``
        from the list stored at ``name``.

        The count argument influences the operation in the following ways:
            count > 0: Remove elements equal to value moving from head to tail.
            count < 0: Remove elements equal to value moving from tail to head.
            count = 0: Remove all elements equal to value.

            For more information see https://redis.io/commands/lrem
        """
        return self.execute_command("LREM", name, count, value)

    def lset(self, name: str, index: int, value: str) -> Union[Awaitable[str], str]:
        """
        Set element at ``index`` of list ``name`` to ``value``

        For more information see https://redis.io/commands/lset
        """
        return self.execute_command("LSET", name, index, value)

    def ltrim(self, name: str, start: int, end: int) -> Union[Awaitable[str], str]:
        """
        Trim the list ``name``, removing all values not within the slice
        between ``start`` and ``end``

        ``start`` and ``end`` can be negative numbers just like
        Python slicing notation

        For more information see https://redis.io/commands/ltrim
        """
        return self.execute_command("LTRIM", name, start, end)

    def rpop(
        self,
        name: str,
        count: Optional[int] = None,
    ) -> Union[Awaitable[Union[str, List, None]], Union[str, List, None]]:
        """
        Removes and returns the last elements of the list ``name``.

        By default, the command pops a single element from the end of the list.
        When provided with the optional ``count`` argument, the reply will
        consist of up to count elements, depending on the list's length.

        For more information see https://redis.io/commands/rpop
        """
        if count is not None:
            return self.execute_command("RPOP", name, count)
        else:
            return self.execute_command("RPOP", name)

    def rpoplpush(self, src: str, dst: str) -> Union[Awaitable[str], str]:
        """
        RPOP a value off of the ``src`` list and atomically LPUSH it
        on to the ``dst`` list.  Returns the value.

        For more information see https://redis.io/commands/rpoplpush
        """
        return self.execute_command("RPOPLPUSH", src, dst)

    def rpush(self, name: str, *values: FieldT) -> Union[Awaitable[int], int]:
        """
        Push ``values`` onto the tail of the list ``name``

        For more information see https://redis.io/commands/rpush
        """
        return self.execute_command("RPUSH", name, *values)

    def rpushx(self, name: str, value: str) -> Union[Awaitable[int], int]:
        """
        Push ``value`` onto the tail of the list ``name`` if ``name`` exists

        For more information see https://redis.io/commands/rpushx
        """
        return self.execute_command("RPUSHX", name, value)

    def lpos(
        self,
        name: str,
        value: str,
        rank: Optional[int] = None,
        count: Optional[int] = None,
        maxlen: Optional[int] = None,
    ) -> Union[str, List, None]:
        """
        Get position of ``value`` within the list ``name``

         If specified, ``rank`` indicates the "rank" of the first element to
         return in case there are multiple copies of ``value`` in the list.
         By default, LPOS returns the position of the first occurrence of
         ``value`` in the list. When ``rank`` 2, LPOS returns the position of
         the second ``value`` in the list. If ``rank`` is negative, LPOS
         searches the list in reverse. For example, -1 would return the
         position of the last occurrence of ``value`` and -2 would return the
         position of the next to last occurrence of ``value``.

         If specified, ``count`` indicates that LPOS should return a list of
         up to ``count`` positions. A ``count`` of 2 would return a list of
         up to 2 positions. A ``count`` of 0 returns a list of all positions
         matching ``value``. When ``count`` is specified and but ``value``
         does not exist in the list, an empty list is returned.

         If specified, ``maxlen`` indicates the maximum number of list
         elements to scan. A ``maxlen`` of 1000 will only return the
         position(s) of items within the first 1000 entries in the list.
         A ``maxlen`` of 0 (the default) will scan the entire list.

         For more information see https://redis.io/commands/lpos
        """
        pieces: list[EncodableT] = [name, value]
        if rank is not None:
            pieces.extend(["RANK", rank])

        if count is not None:
            pieces.extend(["COUNT", count])

        if maxlen is not None:
            pieces.extend(["MAXLEN", maxlen])

        return self.execute_command("LPOS", *pieces)

    def sort(
        self,
        name: str,
        start: Optional[int] = None,
        num: Optional[int] = None,
        by: Optional[str] = None,
        get: Optional[List[str]] = None,
        desc: bool = False,
        alpha: bool = False,
        store: Optional[str] = None,
        groups: Optional[bool] = False,
    ) -> Union[List, int]:
        """
        Sort and return the list, set or sorted set at ``name``.

        ``start`` and ``num`` allow for paging through the sorted data

        ``by`` allows using an external key to weight and sort the items.
            Use an "*" to indicate where in the key the item value is located

        ``get`` allows for returning items from external keys rather than the
            sorted data itself.  Use an "*" to indicate where in the key
            the item value is located

        ``desc`` allows for reversing the sort

        ``alpha`` allows for sorting lexicographically rather than numerically

        ``store`` allows for storing the result of the sort into
            the key ``store``

        ``groups`` if set to True and if ``get`` contains at least two
            elements, sort will return a list of tuples, each containing the
            values fetched from the arguments to ``get``.

        For more information see https://redis.io/commands/sort
        """
        if (start is not None and num is None) or (num is not None and start is None):
            raise DataError("``start`` and ``num`` must both be specified")

        pieces: list[EncodableT] = [name]
        if by is not None:
            pieces.extend([b"BY", by])
        if start is not None and num is not None:
            pieces.extend([b"LIMIT", start, num])
        if get is not None:
            # If get is a string assume we want to get a single value.
            # Otherwise assume it's an interable and we want to get multiple
            # values. We can't just iterate blindly because strings are
            # iterable.
            if isinstance(get, (bytes, str)):
                pieces.extend([b"GET", get])
            else:
                for g in get:
                    pieces.extend([b"GET", g])
        if desc:
            pieces.append(b"DESC")
        if alpha:
            pieces.append(b"ALPHA")
        if store is not None:
            pieces.extend([b"STORE", store])
        if groups:
            if not get or isinstance(get, (bytes, str)) or len(get) < 2:
                raise DataError(
                    'when using "groups" the "get" argument '
                    "must be specified and contain at least "
                    "two keys"
                )

        options = {"groups": len(get) if groups else None}
        return self.execute_command("SORT", *pieces, **options)

    def sort_ro(
        self,
        key: str,
        start: Optional[int] = None,
        num: Optional[int] = None,
        by: Optional[str] = None,
        get: Optional[List[str]] = None,
        desc: bool = False,
        alpha: bool = False,
    ) -> list:
        """
        Returns the elements contained in the list, set or sorted set at key.
        (read-only variant of the SORT command)

        ``start`` and ``num`` allow for paging through the sorted data

        ``by`` allows using an external key to weight and sort the items.
            Use an "*" to indicate where in the key the item value is located

        ``get`` allows for returning items from external keys rather than the
            sorted data itself.  Use an "*" to indicate where in the key
            the item value is located

        ``desc`` allows for reversing the sort

        ``alpha`` allows for sorting lexicographically rather than numerically

        For more information see https://redis.io/commands/sort_ro
        """
        return self.sort(
            key, start=start, num=num, by=by, get=get, desc=desc, alpha=alpha
        )


AsyncListCommands = ListCommands


class ScanCommands(CommandsProtocol):
    """
    Redis SCAN commands.
    see: https://redis.io/commands/scan
    """

    def scan(
        self,
        cursor: int = 0,
        match: Union[PatternT, None] = None,
        count: Union[int, None] = None,
        _type: Union[str, None] = None,
        **kwargs,
    ) -> ResponseT:
        """
        Incrementally return lists of key names. Also return a cursor
        indicating the scan position.

        ``match`` allows for filtering the keys by pattern

        ``count`` provides a hint to Redis about the number of keys to
            return per batch.

        ``_type`` filters the returned values by a particular Redis type.
            Stock Redis instances allow for the following types:
            HASH, LIST, SET, STREAM, STRING, ZSET
            Additionally, Redis modules can expose other types as well.

        For more information see https://redis.io/commands/scan
        """
        pieces: list[EncodableT] = [cursor]
        if match is not None:
            pieces.extend([b"MATCH", match])
        if count is not None:
            pieces.extend([b"COUNT", count])
        if _type is not None:
            pieces.extend([b"TYPE", _type])
        return self.execute_command("SCAN", *pieces, **kwargs)

    def scan_iter(
        self,
        match: Union[PatternT, None] = None,
        count: Union[int, None] = None,
        _type: Union[str, None] = None,
        **kwargs,
    ) -> Iterator:
        """
        Make an iterator using the SCAN command so that the client doesn't
        need to remember the cursor position.

        ``match`` allows for filtering the keys by pattern

        ``count`` provides a hint to Redis about the number of keys to
            return per batch.

        ``_type`` filters the returned values by a particular Redis type.
            Stock Redis instances allow for the following types:
            HASH, LIST, SET, STREAM, STRING, ZSET
            Additionally, Redis modules can expose other types as well.
        """
        cursor = "0"
        while cursor != 0:
            cursor, data = self.scan(
                cursor=cursor, match=match, count=count, _type=_type, **kwargs
            )
            yield from data

    def sscan(
        self,
        name: KeyT,
        cursor: int = 0,
        match: Union[PatternT, None] = None,
        count: Union[int, None] = None,
    ) -> ResponseT:
        """
        Incrementally return lists of elements in a set. Also return a cursor
        indicating the scan position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns

        For more information see https://redis.io/commands/sscan
        """
        pieces: list[EncodableT] = [name, cursor]
        if match is not None:
            pieces.extend([b"MATCH", match])
        if count is not None:
            pieces.extend([b"COUNT", count])
        return self.execute_command("SSCAN", *pieces)

    def sscan_iter(
        self,
        name: KeyT,
        match: Union[PatternT, None] = None,
        count: Union[int, None] = None,
    ) -> Iterator:
        """
        Make an iterator using the SSCAN command so that the client doesn't
        need to remember the cursor position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns
        """
        cursor = "0"
        while cursor != 0:
            cursor, data = self.sscan(name, cursor=cursor, match=match, count=count)
            yield from data

    def hscan(
        self,
        name: KeyT,
        cursor: int = 0,
        match: Union[PatternT, None] = None,
        count: Union[int, None] = None,
    ) -> ResponseT:
        """
        Incrementally return key/value slices in a hash. Also return a cursor
        indicating the scan position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns

        For more information see https://redis.io/commands/hscan
        """
        pieces: list[EncodableT] = [name, cursor]
        if match is not None:
            pieces.extend([b"MATCH", match])
        if count is not None:
            pieces.extend([b"COUNT", count])
        return self.execute_command("HSCAN", *pieces)

    def hscan_iter(
        self,
        name: str,
        match: Union[PatternT, None] = None,
        count: Union[int, None] = None,
    ) -> Iterator:
        """
        Make an iterator using the HSCAN command so that the client doesn't
        need to remember the cursor position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns
        """
        cursor = "0"
        while cursor != 0:
            cursor, data = self.hscan(name, cursor=cursor, match=match, count=count)
            yield from data.items()

    def zscan(
        self,
        name: KeyT,
        cursor: int = 0,
        match: Union[PatternT, None] = None,
        count: Union[int, None] = None,
        score_cast_func: Union[type, Callable] = float,
    ) -> ResponseT:
        """
        Incrementally return lists of elements in a sorted set. Also return a
        cursor indicating the scan position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns

        ``score_cast_func`` a callable used to cast the score return value

        For more information see https://redis.io/commands/zscan
        """
        pieces = [name, cursor]
        if match is not None:
            pieces.extend([b"MATCH", match])
        if count is not None:
            pieces.extend([b"COUNT", count])
        options = {"score_cast_func": score_cast_func}
        return self.execute_command("ZSCAN", *pieces, **options)

    def zscan_iter(
        self,
        name: KeyT,
        match: Union[PatternT, None] = None,
        count: Union[int, None] = None,
        score_cast_func: Union[type, Callable] = float,
    ) -> Iterator:
        """
        Make an iterator using the ZSCAN command so that the client doesn't
        need to remember the cursor position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns

        ``score_cast_func`` a callable used to cast the score return value
        """
        cursor = "0"
        while cursor != 0:
            cursor, data = self.zscan(
                name,
                cursor=cursor,
                match=match,
                count=count,
                score_cast_func=score_cast_func,
            )
            yield from data


class AsyncScanCommands(ScanCommands):
    async def scan_iter(
        self,
        match: Union[PatternT, None] = None,
        count: Union[int, None] = None,
        _type: Union[str, None] = None,
        **kwargs,
    ) -> AsyncIterator:
        """
        Make an iterator using the SCAN command so that the client doesn't
        need to remember the cursor position.

        ``match`` allows for filtering the keys by pattern

        ``count`` provides a hint to Redis about the number of keys to
            return per batch.

        ``_type`` filters the returned values by a particular Redis type.
            Stock Redis instances allow for the following types:
            HASH, LIST, SET, STREAM, STRING, ZSET
            Additionally, Redis modules can expose other types as well.
        """
        cursor = "0"
        while cursor != 0:
            cursor, data = await self.scan(
                cursor=cursor, match=match, count=count, _type=_type, **kwargs
            )
            for d in data:
                yield d

    async def sscan_iter(
        self,
        name: KeyT,
        match: Union[PatternT, None] = None,
        count: Union[int, None] = None,
    ) -> AsyncIterator:
        """
        Make an iterator using the SSCAN command so that the client doesn't
        need to remember the cursor position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns
        """
        cursor = "0"
        while cursor != 0:
            cursor, data = await self.sscan(
                name, cursor=cursor, match=match, count=count
            )
            for d in data:
                yield d

    async def hscan_iter(
        self,
        name: str,
        match: Union[PatternT, None] = None,
        count: Union[int, None] = None,
    ) -> AsyncIterator:
        """
        Make an iterator using the HSCAN command so that the client doesn't
        need to remember the cursor position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns
        """
        cursor = "0"
        while cursor != 0:
            cursor, data = await self.hscan(
                name, cursor=cursor, match=match, count=count
            )
            for it in data.items():
                yield it

    async def zscan_iter(
        self,
        name: KeyT,
        match: Union[PatternT, None] = None,
        count: Union[int, None] = None,
        score_cast_func: Union[type, Callable] = float,
    ) -> AsyncIterator:
        """
        Make an iterator using the ZSCAN command so that the client doesn't
        need to remember the cursor position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns

        ``score_cast_func`` a callable used to cast the score return value
        """
        cursor = "0"
        while cursor != 0:
            cursor, data = await self.zscan(
                name,
                cursor=cursor,
                match=match,
                count=count,
                score_cast_func=score_cast_func,
            )
            for d in data:
                yield d


class SetCommands(CommandsProtocol):
    """
    Redis commands for Set data type.
    see: https://redis.io/topics/data-types#sets
    """

    def sadd(self, name: str, *values: FieldT) -> Union[Awaitable[int], int]:
        """
        Add ``value(s)`` to set ``name``

        For more information see https://redis.io/commands/sadd
        """
        return self.execute_command("SADD", name, *values)

    def scard(self, name: str) -> Union[Awaitable[int], int]:
        """
        Return the number of elements in set ``name``

        For more information see https://redis.io/commands/scard
        """
        return self.execute_command("SCARD", name)

    def sdiff(self, keys: List, *args: List) -> Union[Awaitable[list], list]:
        """
        Return the difference of sets specified by ``keys``

        For more information see https://redis.io/commands/sdiff
        """
        args = list_or_args(keys, args)
        return self.execute_command("SDIFF", *args)

    def sdiffstore(
        self, dest: str, keys: List, *args: List
    ) -> Union[Awaitable[int], int]:
        """
        Store the difference of sets specified by ``keys`` into a new
        set named ``dest``.  Returns the number of keys in the new set.

        For more information see https://redis.io/commands/sdiffstore
        """
        args = list_or_args(keys, args)
        return self.execute_command("SDIFFSTORE", dest, *args)

    def sinter(self, keys: List, *args: List) -> Union[Awaitable[list], list]:
        """
        Return the intersection of sets specified by ``keys``

        For more information see https://redis.io/commands/sinter
        """
        args = list_or_args(keys, args)
        return self.execute_command("SINTER", *args)

    def sintercard(
        self, numkeys: int, keys: List[str], limit: int = 0
    ) -> Union[Awaitable[int], int]:
        """
        Return the cardinality of the intersect of multiple sets specified by ``keys`.

        When LIMIT provided (defaults to 0 and means unlimited), if the intersection
        cardinality reaches limit partway through the computation, the algorithm will
        exit and yield limit as the cardinality

        For more information see https://redis.io/commands/sintercard
        """
        args = [numkeys, *keys, "LIMIT", limit]
        return self.execute_command("SINTERCARD", *args)

    def sinterstore(
        self, dest: str, keys: List, *args: List
    ) -> Union[Awaitable[int], int]:
        """
        Store the intersection of sets specified by ``keys`` into a new
        set named ``dest``.  Returns the number of keys in the new set.

        For more information see https://redis.io/commands/sinterstore
        """
        args = list_or_args(keys, args)
        return self.execute_command("SINTERSTORE", dest, *args)

    def sismember(self, name: str, value: str) -> Union[Awaitable[bool], bool]:
        """
        Return a boolean indicating if ``value`` is a member of set ``name``

        For more information see https://redis.io/commands/sismember
        """
        return self.execute_command("SISMEMBER", name, value)

    def smembers(self, name: str) -> Union[Awaitable[Set], Set]:
        """
        Return all members of the set ``name``

        For more information see https://redis.io/commands/smembers
        """
        return self.execute_command("SMEMBERS", name)

    def smismember(
        self, name: str, values: List, *args: List
    ) -> Union[
        Awaitable[List[Union[Literal[0], Literal[1]]]],
        List[Union[Literal[0], Literal[1]]],
    ]:
        """
        Return whether each value in ``values`` is a member of the set ``name``
        as a list of ``int`` in the order of ``values``:
        - 1 if the value is a member of the set.
        - 0 if the value is not a member of the set or if key does not exist.

        For more information see https://redis.io/commands/smismember
        """
        args = list_or_args(values, args)
        return self.execute_command("SMISMEMBER", name, *args)

    def smove(self, src: str, dst: str, value: str) -> Union[Awaitable[bool], bool]:
        """
        Move ``value`` from set ``src`` to set ``dst`` atomically

        For more information see https://redis.io/commands/smove
        """
        return self.execute_command("SMOVE", src, dst, value)

    def spop(self, name: str, count: Optional[int] = None) -> Union[str, List, None]:
        """
        Remove and return a random member of set ``name``

        For more information see https://redis.io/commands/spop
        """
        args = (count is not None) and [count] or []
        return self.execute_command("SPOP", name, *args)

    def srandmember(
        self, name: str, number: Optional[int] = None
    ) -> Union[str, List, None]:
        """
        If ``number`` is None, returns a random member of set ``name``.

        If ``number`` is supplied, returns a list of ``number`` random
        members of set ``name``. Note this is only available when running
        Redis 2.6+.

        For more information see https://redis.io/commands/srandmember
        """
        args = (number is not None) and [number] or []
        return self.execute_command("SRANDMEMBER", name, *args)

    def srem(self, name: str, *values: FieldT) -> Union[Awaitable[int], int]:
        """
        Remove ``values`` from set ``name``

        For more information see https://redis.io/commands/srem
        """
        return self.execute_command("SREM", name, *values)

    def sunion(self, keys: List, *args: List) -> Union[Awaitable[List], List]:
        """
        Return the union of sets specified by ``keys``

        For more information see https://redis.io/commands/sunion
        """
        args = list_or_args(keys, args)
        return self.execute_command("SUNION", *args)

    def sunionstore(
        self, dest: str, keys: List, *args: List
    ) -> Union[Awaitable[int], int]:
        """
        Store the union of sets specified by ``keys`` into a new
        set named ``dest``.  Returns the number of keys in the new set.

        For more information see https://redis.io/commands/sunionstore
        """
        args = list_or_args(keys, args)
        return self.execute_command("SUNIONSTORE", dest, *args)


AsyncSetCommands = SetCommands


class StreamCommands(CommandsProtocol):
    """
    Redis commands for Stream data type.
    see: https://redis.io/topics/streams-intro
    """

    def xack(self, name: KeyT, groupname: GroupT, *ids: StreamIdT) -> ResponseT:
        """
        Acknowledges the successful processing of one or more messages.
        name: name of the stream.
        groupname: name of the consumer group.
        *ids: message ids to acknowledge.

        For more information see https://redis.io/commands/xack
        """
        return self.execute_command("XACK", name, groupname, *ids)

    def xadd(
        self,
        name: KeyT,
        fields: Dict[FieldT, EncodableT],
        id: StreamIdT = "*",
        maxlen: Union[int, None] = None,
        approximate: bool = True,
        nomkstream: bool = False,
        minid: Union[StreamIdT, None] = None,
        limit: Union[int, None] = None,
    ) -> ResponseT:
        """
        Add to a stream.
        name: name of the stream
        fields: dict of field/value pairs to insert into the stream
        id: Location to insert this record. By default it is appended.
        maxlen: truncate old stream members beyond this size.
        Can't be specified with minid.
        approximate: actual stream length may be slightly more than maxlen
        nomkstream: When set to true, do not make a stream
        minid: the minimum id in the stream to query.
        Can't be specified with maxlen.
        limit: specifies the maximum number of entries to retrieve

        For more information see https://redis.io/commands/xadd
        """
        pieces: list[EncodableT] = []
        if maxlen is not None and minid is not None:
            raise DataError("Only one of ```maxlen``` or ```minid``` may be specified")

        if maxlen is not None:
            if not isinstance(maxlen, int) or maxlen < 1:
                raise DataError("XADD maxlen must be a positive integer")
            pieces.append(b"MAXLEN")
            if approximate:
                pieces.append(b"~")
            pieces.append(str(maxlen))
        if minid is not None:
            pieces.append(b"MINID")
            if approximate:
                pieces.append(b"~")
            pieces.append(minid)
        if limit is not None:
            pieces.extend([b"LIMIT", limit])
        if nomkstream:
            pieces.append(b"NOMKSTREAM")
        pieces.append(id)
        if not isinstance(fields, dict) or len(fields) == 0:
            raise DataError("XADD fields must be a non-empty dict")
        for pair in fields.items():
            pieces.extend(pair)
        return self.execute_command("XADD", name, *pieces)

    def xautoclaim(
        self,
        name: KeyT,
        groupname: GroupT,
        consumername: ConsumerT,
        min_idle_time: int,
        start_id: StreamIdT = "0-0",
        count: Union[int, None] = None,
        justid: bool = False,
    ) -> ResponseT:
        """
        Transfers ownership of pending stream entries that match the specified
        criteria. Conceptually, equivalent to calling XPENDING and then XCLAIM,
        but provides a more straightforward way to deal with message delivery
        failures via SCAN-like semantics.
        name: name of the stream.
        groupname: name of the consumer group.
        consumername: name of a consumer that claims the message.
        min_idle_time: filter messages that were idle less than this amount of
        milliseconds.
        start_id: filter messages with equal or greater ID.
        count: optional integer, upper limit of the number of entries that the
        command attempts to claim. Set to 100 by default.
        justid: optional boolean, false by default. Return just an array of IDs
        of messages successfully claimed, without returning the actual message

        For more information see https://redis.io/commands/xautoclaim
        """
        try:
            if int(min_idle_time) < 0:
                raise DataError(
                    "XAUTOCLAIM min_idle_time must be a nonnegative integer"
                )
        except TypeError:
            pass

        kwargs = {}
        pieces = [name, groupname, consumername, min_idle_time, start_id]

        try:
            if int(count) < 0:
                raise DataError("XPENDING count must be a integer >= 0")
            pieces.extend([b"COUNT", count])
        except TypeError:
            pass
        if justid:
            pieces.append(b"JUSTID")
            kwargs["parse_justid"] = True

        return self.execute_command("XAUTOCLAIM", *pieces, **kwargs)

    def xclaim(
        self,
        name: KeyT,
        groupname: GroupT,
        consumername: ConsumerT,
        min_idle_time: int,
        message_ids: Union[List[StreamIdT], Tuple[StreamIdT]],
        idle: Union[int, None] = None,
        time: Union[int, None] = None,
        retrycount: Union[int, None] = None,
        force: bool = False,
        justid: bool = False,
    ) -> ResponseT:
        """
        Changes the ownership of a pending message.
        name: name of the stream.
        groupname: name of the consumer group.
        consumername: name of a consumer that claims the message.
        min_idle_time: filter messages that were idle less than this amount of
        milliseconds
        message_ids: non-empty list or tuple of message IDs to claim
        idle: optional. Set the idle time (last time it was delivered) of the
         message in ms
        time: optional integer. This is the same as idle but instead of a
         relative amount of milliseconds, it sets the idle time to a specific
         Unix time (in milliseconds).
        retrycount: optional integer. set the retry counter to the specified
         value. This counter is incremented every time a message is delivered
         again.
        force: optional boolean, false by default. Creates the pending message
         entry in the PEL even if certain specified IDs are not already in the
         PEL assigned to a different client.
        justid: optional boolean, false by default. Return just an array of IDs
         of messages successfully claimed, without returning the actual message

         For more information see https://redis.io/commands/xclaim
        """
        if not isinstance(min_idle_time, int) or min_idle_time < 0:
            raise DataError("XCLAIM min_idle_time must be a non negative integer")
        if not isinstance(message_ids, (list, tuple)) or not message_ids:
            raise DataError(
                "XCLAIM message_ids must be a non empty list or "
                "tuple of message IDs to claim"
            )

        kwargs = {}
        pieces: list[EncodableT] = [name, groupname, consumername, str(min_idle_time)]
        pieces.extend(list(message_ids))

        if idle is not None:
            if not isinstance(idle, int):
                raise DataError("XCLAIM idle must be an integer")
            pieces.extend((b"IDLE", str(idle)))
        if time is not None:
            if not isinstance(time, int):
                raise DataError("XCLAIM time must be an integer")
            pieces.extend((b"TIME", str(time)))
        if retrycount is not None:
            if not isinstance(retrycount, int):
                raise DataError("XCLAIM retrycount must be an integer")
            pieces.extend((b"RETRYCOUNT", str(retrycount)))

        if force:
            if not isinstance(force, bool):
                raise DataError("XCLAIM force must be a boolean")
            pieces.append(b"FORCE")
        if justid:
            if not isinstance(justid, bool):
                raise DataError("XCLAIM justid must be a boolean")
            pieces.append(b"JUSTID")
            kwargs["parse_justid"] = True
        return self.execute_command("XCLAIM", *pieces, **kwargs)

    def xdel(self, name: KeyT, *ids: StreamIdT) -> ResponseT:
        """
        Deletes one or more messages from a stream.
        name: name of the stream.
        *ids: message ids to delete.

        For more information see https://redis.io/commands/xdel
        """
        return self.execute_command("XDEL", name, *ids)

    def xgroup_create(
        self,
        name: KeyT,
        groupname: GroupT,
        id: StreamIdT = "$",
        mkstream: bool = False,
        entries_read: Optional[int] = None,
    ) -> ResponseT:
        """
        Create a new consumer group associated with a stream.
        name: name of the stream.
        groupname: name of the consumer group.
        id: ID of the last item in the stream to consider already delivered.

        For more information see https://redis.io/commands/xgroup-create
        """
        pieces: list[EncodableT] = ["XGROUP CREATE", name, groupname, id]
        if mkstream:
            pieces.append(b"MKSTREAM")
        if entries_read is not None:
            pieces.extend(["ENTRIESREAD", entries_read])

        return self.execute_command(*pieces)

    def xgroup_delconsumer(
        self, name: KeyT, groupname: GroupT, consumername: ConsumerT
    ) -> ResponseT:
        """
        Remove a specific consumer from a consumer group.
        Returns the number of pending messages that the consumer had before it
        was deleted.
        name: name of the stream.
        groupname: name of the consumer group.
        consumername: name of consumer to delete

        For more information see https://redis.io/commands/xgroup-delconsumer
        """
        return self.execute_command("XGROUP DELCONSUMER", name, groupname, consumername)

    def xgroup_destroy(self, name: KeyT, groupname: GroupT) -> ResponseT:
        """
        Destroy a consumer group.
        name: name of the stream.
        groupname: name of the consumer group.

        For more information see https://redis.io/commands/xgroup-destroy
        """
        return self.execute_command("XGROUP DESTROY", name, groupname)

    def xgroup_createconsumer(
        self, name: KeyT, groupname: GroupT, consumername: ConsumerT
    ) -> ResponseT:
        """
        Consumers in a consumer group are auto-created every time a new
        consumer name is mentioned by some command.
        They can be explicitly created by using this command.
        name: name of the stream.
        groupname: name of the consumer group.
        consumername: name of consumer to create.

        See: https://redis.io/commands/xgroup-createconsumer
        """
        return self.execute_command(
            "XGROUP CREATECONSUMER", name, groupname, consumername
        )

    def xgroup_setid(
        self,
        name: KeyT,
        groupname: GroupT,
        id: StreamIdT,
        entries_read: Optional[int] = None,
    ) -> ResponseT:
        """
        Set the consumer group last delivered ID to something else.
        name: name of the stream.
        groupname: name of the consumer group.
        id: ID of the last item in the stream to consider already delivered.

        For more information see https://redis.io/commands/xgroup-setid
        """
        pieces = [name, groupname, id]
        if entries_read is not None:
            pieces.extend(["ENTRIESREAD", entries_read])
        return self.execute_command("XGROUP SETID", *pieces)

    def xinfo_consumers(self, name: KeyT, groupname: GroupT) -> ResponseT:
        """
        Returns general information about the consumers in the group.
        name: name of the stream.
        groupname: name of the consumer group.

        For more information see https://redis.io/commands/xinfo-consumers
        """
        return self.execute_command("XINFO CONSUMERS", name, groupname)

    def xinfo_groups(self, name: KeyT) -> ResponseT:
        """
        Returns general information about the consumer groups of the stream.
        name: name of the stream.

        For more information see https://redis.io/commands/xinfo-groups
        """
        return self.execute_command("XINFO GROUPS", name)

    def xinfo_stream(self, name: KeyT, full: bool = False) -> ResponseT:
        """
        Returns general information about the stream.
        name: name of the stream.
        full: optional boolean, false by default. Return full summary

        For more information see https://redis.io/commands/xinfo-stream
        """
        pieces = [name]
        options = {}
        if full:
            pieces.append(b"FULL")
            options = {"full": full}
        return self.execute_command("XINFO STREAM", *pieces, **options)

    def xlen(self, name: KeyT) -> ResponseT:
        """
        Returns the number of elements in a given stream.

        For more information see https://redis.io/commands/xlen
        """
        return self.execute_command("XLEN", name)

    def xpending(self, name: KeyT, groupname: GroupT) -> ResponseT:
        """
        Returns information about pending messages of a group.
        name: name of the stream.
        groupname: name of the consumer group.

        For more information see https://redis.io/commands/xpending
        """
        return self.execute_command("XPENDING", name, groupname)

    def xpending_range(
        self,
        name: KeyT,
        groupname: GroupT,
        min: StreamIdT,
        max: StreamIdT,
        count: int,
        consumername: Union[ConsumerT, None] = None,
        idle: Union[int, None] = None,
    ) -> ResponseT:
        """
        Returns information about pending messages, in a range.

        name: name of the stream.
        groupname: name of the consumer group.
        idle: available from  version 6.2. filter entries by their
        idle-time, given in milliseconds (optional).
        min: minimum stream ID.
        max: maximum stream ID.
        count: number of messages to return
        consumername: name of a consumer to filter by (optional).
        """
        if {min, max, count} == {None}:
            if idle is not None or consumername is not None:
                raise DataError(
                    "if XPENDING is provided with idle time"
                    " or consumername, it must be provided"
                    " with min, max and count parameters"
                )
            return self.xpending(name, groupname)

        pieces = [name, groupname]
        if min is None or max is None or count is None:
            raise DataError(
                "XPENDING must be provided with min, max "
                "and count parameters, or none of them."
            )
        # idle
        try:
            if int(idle) < 0:
                raise DataError("XPENDING idle must be a integer >= 0")
            pieces.extend(["IDLE", idle])
        except TypeError:
            pass
        # count
        try:
            if int(count) < 0:
                raise DataError("XPENDING count must be a integer >= 0")
            pieces.extend([min, max, count])
        except TypeError:
            pass
        # consumername
        if consumername:
            pieces.append(consumername)

        return self.execute_command("XPENDING", *pieces, parse_detail=True)

    def xrange(
        self,
        name: KeyT,
        min: StreamIdT = "-",
        max: StreamIdT = "+",
        count: Union[int, None] = None,
    ) -> ResponseT:
        """
        Read stream values within an interval.
        name: name of the stream.
        start: first stream ID. defaults to '-',
               meaning the earliest available.
        finish: last stream ID. defaults to '+',
                meaning the latest available.
        count: if set, only return this many items, beginning with the
               earliest available.

        For more information see https://redis.io/commands/xrange
        """
        pieces = [min, max]
        if count is not None:
            if not isinstance(count, int) or count < 1:
                raise DataError("XRANGE count must be a positive integer")
            pieces.append(b"COUNT")
            pieces.append(str(count))

        return self.execute_command("XRANGE", name, *pieces)

    def xread(
        self,
        streams: Dict[KeyT, StreamIdT],
        count: Union[int, None] = None,
        block: Union[int, None] = None,
    ) -> ResponseT:
        """
        Block and monitor multiple streams for new data.
        streams: a dict of stream names to stream IDs, where
                   IDs indicate the last ID already seen.
        count: if set, only return this many items, beginning with the
               earliest available.
        block: number of milliseconds to wait, if nothing already present.

        For more information see https://redis.io/commands/xread
        """
        pieces = []
        if block is not None:
            if not isinstance(block, int) or block < 0:
                raise DataError("XREAD block must be a non-negative integer")
            pieces.append(b"BLOCK")
            pieces.append(str(block))
        if count is not None:
            if not isinstance(count, int) or count < 1:
                raise DataError("XREAD count must be a positive integer")
            pieces.append(b"COUNT")
            pieces.append(str(count))
        if not isinstance(streams, dict) or len(streams) == 0:
            raise DataError("XREAD streams must be a non empty dict")
        pieces.append(b"STREAMS")
        keys, values = zip(*streams.items())
        pieces.extend(keys)
        pieces.extend(values)
        return self.execute_command("XREAD", *pieces)

    def xreadgroup(
        self,
        groupname: str,
        consumername: str,
        streams: Dict[KeyT, StreamIdT],
        count: Union[int, None] = None,
        block: Union[int, None] = None,
        noack: bool = False,
    ) -> ResponseT:
        """
        Read from a stream via a consumer group.
        groupname: name of the consumer group.
        consumername: name of the requesting consumer.
        streams: a dict of stream names to stream IDs, where
               IDs indicate the last ID already seen.
        count: if set, only return this many items, beginning with the
               earliest available.
        block: number of milliseconds to wait, if nothing already present.
        noack: do not add messages to the PEL

        For more information see https://redis.io/commands/xreadgroup
        """
        pieces: list[EncodableT] = [b"GROUP", groupname, consumername]
        if count is not None:
            if not isinstance(count, int) or count < 1:
                raise DataError("XREADGROUP count must be a positive integer")
            pieces.append(b"COUNT")
            pieces.append(str(count))
        if block is not None:
            if not isinstance(block, int) or block < 0:
                raise DataError("XREADGROUP block must be a non-negative integer")
            pieces.append(b"BLOCK")
            pieces.append(str(block))
        if noack:
            pieces.append(b"NOACK")
        if not isinstance(streams, dict) or len(streams) == 0:
            raise DataError("XREADGROUP streams must be a non empty dict")
        pieces.append(b"STREAMS")
        pieces.extend(streams.keys())
        pieces.extend(streams.values())
        return self.execute_command("XREADGROUP", *pieces)

    def xrevrange(
        self,
        name: KeyT,
        max: StreamIdT = "+",
        min: StreamIdT = "-",
        count: Union[int, None] = None,
    ) -> ResponseT:
        """
        Read stream values within an interval, in reverse order.
        name: name of the stream
        start: first stream ID. defaults to '+',
               meaning the latest available.
        finish: last stream ID. defaults to '-',
                meaning the earliest available.
        count: if set, only return this many items, beginning with the
               latest available.

        For more information see https://redis.io/commands/xrevrange
        """
        pieces: list[EncodableT] = [max, min]
        if count is not None:
            if not isinstance(count, int) or count < 1:
                raise DataError("XREVRANGE count must be a positive integer")
            pieces.append(b"COUNT")
            pieces.append(str(count))

        return self.execute_command("XREVRANGE", name, *pieces)

    def xtrim(
        self,
        name: KeyT,
        maxlen: Union[int, None] = None,
        approximate: bool = True,
        minid: Union[StreamIdT, None] = None,
        limit: Union[int, None] = None,
    ) -> ResponseT:
        """
        Trims old messages from a stream.
        name: name of the stream.
        maxlen: truncate old stream messages beyond this size
        Can't be specified with minid.
        approximate: actual stream length may be slightly more than maxlen
        minid: the minimum id in the stream to query
        Can't be specified with maxlen.
        limit: specifies the maximum number of entries to retrieve

        For more information see https://redis.io/commands/xtrim
        """
        pieces: list[EncodableT] = []
        if maxlen is not None and minid is not None:
            raise DataError("Only one of ``maxlen`` or ``minid`` may be specified")

        if maxlen is None and minid is None:
            raise DataError("One of ``maxlen`` or ``minid`` must be specified")

        if maxlen is not None:
            pieces.append(b"MAXLEN")
        if minid is not None:
            pieces.append(b"MINID")
        if approximate:
            pieces.append(b"~")
        if maxlen is not None:
            pieces.append(maxlen)
        if minid is not None:
            pieces.append(minid)
        if limit is not None:
            pieces.append(b"LIMIT")
            pieces.append(limit)

        return self.execute_command("XTRIM", name, *pieces)


AsyncStreamCommands = StreamCommands


class SortedSetCommands(CommandsProtocol):
    """
    Redis commands for Sorted Sets data type.
    see: https://redis.io/topics/data-types-intro#redis-sorted-sets
    """

    def zadd(
        self,
        name: KeyT,
        mapping: Mapping[AnyKeyT, EncodableT],
        nx: bool = False,
        xx: bool = False,
        ch: bool = False,
        incr: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> ResponseT:
        """
        Set any number of element-name, score pairs to the key ``name``. Pairs
        are specified as a dict of element-names keys to score values.

        ``nx`` forces ZADD to only create new elements and not to update
        scores for elements that already exist.

        ``xx`` forces ZADD to only update scores of elements that already
        exist. New elements will not be added.

        ``ch`` modifies the return value to be the numbers of elements changed.
        Changed elements include new elements that were added and elements
        whose scores changed.

        ``incr`` modifies ZADD to behave like ZINCRBY. In this mode only a
        single element/score pair can be specified and the score is the amount
        the existing score will be incremented by. When using this mode the
        return value of ZADD will be the new score of the element.

        ``LT`` Only update existing elements if the new score is less than
        the current score. This flag doesn't prevent adding new elements.

        ``GT`` Only update existing elements if the new score is greater than
        the current score. This flag doesn't prevent adding new elements.

        The return value of ZADD varies based on the mode specified. With no
        options, ZADD returns the number of new elements added to the sorted
        set.

        ``NX``, ``LT``, and ``GT`` are mutually exclusive options.

        See: https://redis.io/commands/ZADD
        """
        if not mapping:
            raise DataError("ZADD requires at least one element/score pair")
        if nx and xx:
            raise DataError("ZADD allows either 'nx' or 'xx', not both")
        if gt and lt:
            raise DataError("ZADD allows either 'gt' or 'lt', not both")
        if incr and len(mapping) != 1:
            raise DataError(
                "ZADD option 'incr' only works when passing a "
                "single element/score pair"
            )
        if nx and (gt or lt):
            raise DataError("Only one of 'nx', 'lt', or 'gr' may be defined.")

        pieces: list[EncodableT] = []
        options = {}
        if nx:
            pieces.append(b"NX")
        if xx:
            pieces.append(b"XX")
        if ch:
            pieces.append(b"CH")
        if incr:
            pieces.append(b"INCR")
            options["as_score"] = True
        if gt:
            pieces.append(b"GT")
        if lt:
            pieces.append(b"LT")
        for pair in mapping.items():
            pieces.append(pair[1])
            pieces.append(pair[0])
        return self.execute_command("ZADD", name, *pieces, **options)

    def zcard(self, name: KeyT) -> ResponseT:
        """
        Return the number of elements in the sorted set ``name``

        For more information see https://redis.io/commands/zcard
        """
        return self.execute_command("ZCARD", name)

    def zcount(self, name: KeyT, min: ZScoreBoundT, max: ZScoreBoundT) -> ResponseT:
        """
        Returns the number of elements in the sorted set at key ``name`` with
        a score between ``min`` and ``max``.

        For more information see https://redis.io/commands/zcount
        """
        return self.execute_command("ZCOUNT", name, min, max)

    def zdiff(self, keys: KeysT, withscores: bool = False) -> ResponseT:
        """
        Returns the difference between the first and all successive input
        sorted sets provided in ``keys``.

        For more information see https://redis.io/commands/zdiff
        """
        pieces = [len(keys), *keys]
        if withscores:
            pieces.append("WITHSCORES")
        return self.execute_command("ZDIFF", *pieces)

    def zdiffstore(self, dest: KeyT, keys: KeysT) -> ResponseT:
        """
        Computes the difference between the first and all successive input
        sorted sets provided in ``keys`` and stores the result in ``dest``.

        For more information see https://redis.io/commands/zdiffstore
        """
        pieces = [len(keys), *keys]
        return self.execute_command("ZDIFFSTORE", dest, *pieces)

    def zincrby(self, name: KeyT, amount: float, value: EncodableT) -> ResponseT:
        """
        Increment the score of ``value`` in sorted set ``name`` by ``amount``

        For more information see https://redis.io/commands/zincrby
        """
        return self.execute_command("ZINCRBY", name, amount, value)

    def zinter(
        self, keys: KeysT, aggregate: Union[str, None] = None, withscores: bool = False
    ) -> ResponseT:
        """
        Return the intersect of multiple sorted sets specified by ``keys``.
        With the ``aggregate`` option, it is possible to specify how the
        results of the union are aggregated. This option defaults to SUM,
        where the score of an element is summed across the inputs where it
        exists. When this option is set to either MIN or MAX, the resulting
        set will contain the minimum or maximum score of an element across
        the inputs where it exists.

        For more information see https://redis.io/commands/zinter
        """
        return self._zaggregate("ZINTER", None, keys, aggregate, withscores=withscores)

    def zinterstore(
        self,
        dest: KeyT,
        keys: Union[Sequence[KeyT], Mapping[AnyKeyT, float]],
        aggregate: Union[str, None] = None,
    ) -> ResponseT:
        """
        Intersect multiple sorted sets specified by ``keys`` into a new
        sorted set, ``dest``. Scores in the destination will be aggregated
        based on the ``aggregate``. This option defaults to SUM, where the
        score of an element is summed across the inputs where it exists.
        When this option is set to either MIN or MAX, the resulting set will
        contain the minimum or maximum score of an element across the inputs
        where it exists.

        For more information see https://redis.io/commands/zinterstore
        """
        return self._zaggregate("ZINTERSTORE", dest, keys, aggregate)

    def zintercard(
        self, numkeys: int, keys: List[str], limit: int = 0
    ) -> Union[Awaitable[int], int]:
        """
        Return the cardinality of the intersect of multiple sorted sets
        specified by ``keys`.
        When LIMIT provided (defaults to 0 and means unlimited), if the intersection
        cardinality reaches limit partway through the computation, the algorithm will
        exit and yield limit as the cardinality

        For more information see https://redis.io/commands/zintercard
        """
        args = [numkeys, *keys, "LIMIT", limit]
        return self.execute_command("ZINTERCARD", *args)

    def zlexcount(self, name, min, max):
        """
        Return the number of items in the sorted set ``name`` between the
        lexicographical range ``min`` and ``max``.

        For more information see https://redis.io/commands/zlexcount
        """
        return self.execute_command("ZLEXCOUNT", name, min, max)

    def zpopmax(self, name: KeyT, count: Union[int, None] = None) -> ResponseT:
        """
        Remove and return up to ``count`` members with the highest scores
        from the sorted set ``name``.

        For more information see https://redis.io/commands/zpopmax
        """
        args = (count is not None) and [count] or []
        options = {"withscores": True}
        return self.execute_command("ZPOPMAX", name, *args, **options)

    def zpopmin(self, name: KeyT, count: Union[int, None] = None) -> ResponseT:
        """
        Remove and return up to ``count`` members with the lowest scores
        from the sorted set ``name``.

        For more information see https://redis.io/commands/zpopmin
        """
        args = (count is not None) and [count] or []
        options = {"withscores": True}
        return self.execute_command("ZPOPMIN", name, *args, **options)

    def zrandmember(
        self, key: KeyT, count: int = None, withscores: bool = False
    ) -> ResponseT:
        """
        Return a random element from the sorted set value stored at key.

        ``count`` if the argument is positive, return an array of distinct
        fields. If called with a negative count, the behavior changes and
        the command is allowed to return the same field multiple times.
        In this case, the number of returned fields is the absolute value
        of the specified count.

        ``withscores`` The optional WITHSCORES modifier changes the reply so it
        includes the respective scores of the randomly selected elements from
        the sorted set.

        For more information see https://redis.io/commands/zrandmember
        """
        params = []
        if count is not None:
            params.append(count)
        if withscores:
            params.append("WITHSCORES")

        return self.execute_command("ZRANDMEMBER", key, *params)

    def bzpopmax(self, keys: KeysT, timeout: TimeoutSecT = 0) -> ResponseT:
        """
        ZPOPMAX a value off of the first non-empty sorted set
        named in the ``keys`` list.

        If none of the sorted sets in ``keys`` has a value to ZPOPMAX,
        then block for ``timeout`` seconds, or until a member gets added
        to one of the sorted sets.

        If timeout is 0, then block indefinitely.

        For more information see https://redis.io/commands/bzpopmax
        """
        if timeout is None:
            timeout = 0
        keys = list_or_args(keys, None)
        keys.append(timeout)
        return self.execute_command("BZPOPMAX", *keys)

    def bzpopmin(self, keys: KeysT, timeout: TimeoutSecT = 0) -> ResponseT:
        """
        ZPOPMIN a value off of the first non-empty sorted set
        named in the ``keys`` list.

        If none of the sorted sets in ``keys`` has a value to ZPOPMIN,
        then block for ``timeout`` seconds, or until a member gets added
        to one of the sorted sets.

        If timeout is 0, then block indefinitely.

        For more information see https://redis.io/commands/bzpopmin
        """
        if timeout is None:
            timeout = 0
        keys: list[EncodableT] = list_or_args(keys, None)
        keys.append(timeout)
        return self.execute_command("BZPOPMIN", *keys)

    def zmpop(
        self,
        num_keys: int,
        keys: List[str],
        min: Optional[bool] = False,
        max: Optional[bool] = False,
        count: Optional[int] = 1,
    ) -> Union[Awaitable[list], list]:
        """
        Pop ``count`` values (default 1) off of the first non-empty sorted set
        named in the ``keys`` list.
        For more information see https://redis.io/commands/zmpop
        """
        args = [num_keys] + keys
        if (min and max) or (not min and not max):
            raise DataError
        elif min:
            args.append("MIN")
        else:
            args.append("MAX")
        if count != 1:
            args.extend(["COUNT", count])

        return self.execute_command("ZMPOP", *args)

    def bzmpop(
        self,
        timeout: float,
        numkeys: int,
        keys: List[str],
        min: Optional[bool] = False,
        max: Optional[bool] = False,
        count: Optional[int] = 1,
    ) -> Optional[list]:
        """
        Pop ``count`` values (default 1) off of the first non-empty sorted set
        named in the ``keys`` list.

        If none of the sorted sets in ``keys`` has a value to pop,
        then block for ``timeout`` seconds, or until a member gets added
        to one of the sorted sets.

        If timeout is 0, then block indefinitely.

        For more information see https://redis.io/commands/bzmpop
        """
        args = [timeout, numkeys, *keys]
        if (min and max) or (not min and not max):
            raise DataError("Either min or max, but not both must be set")
        elif min:
            args.append("MIN")
        else:
            args.append("MAX")
        args.extend(["COUNT", count])

        return self.execute_command("BZMPOP", *args)

    def _zrange(
        self,
        command,
        dest: Union[KeyT, None],
        name: KeyT,
        start: int,
        end: int,
        desc: bool = False,
        byscore: bool = False,
        bylex: bool = False,
        withscores: bool = False,
        score_cast_func: Union[type, Callable, None] = float,
        offset: Union[int, None] = None,
        num: Union[int, None] = None,
    ) -> ResponseT:
        if byscore and bylex:
            raise DataError("``byscore`` and ``bylex`` can not be specified together.")
        if (offset is not None and num is None) or (num is not None and offset is None):
            raise DataError("``offset`` and ``num`` must both be specified.")
        if bylex and withscores:
            raise DataError(
                "``withscores`` not supported in combination with ``bylex``."
            )
        pieces = [command]
        if dest:
            pieces.append(dest)
        pieces.extend([name, start, end])
        if byscore:
            pieces.append("BYSCORE")
        if bylex:
            pieces.append("BYLEX")
        if desc:
            pieces.append("REV")
        if offset is not None and num is not None:
            pieces.extend(["LIMIT", offset, num])
        if withscores:
            pieces.append("WITHSCORES")
        options = {"withscores": withscores, "score_cast_func": score_cast_func}
        return self.execute_command(*pieces, **options)

    def zrange(
        self,
        name: KeyT,
        start: int,
        end: int,
        desc: bool = False,
        withscores: bool = False,
        score_cast_func: Union[type, Callable] = float,
        byscore: bool = False,
        bylex: bool = False,
        offset: int = None,
        num: int = None,
    ) -> ResponseT:
        """
        Return a range of values from sorted set ``name`` between
        ``start`` and ``end`` sorted in ascending order.

        ``start`` and ``end`` can be negative, indicating the end of the range.

        ``desc`` a boolean indicating whether to sort the results in reversed
        order.

        ``withscores`` indicates to return the scores along with the values.
        The return type is a list of (value, score) pairs.

        ``score_cast_func`` a callable used to cast the score return value.

        ``byscore`` when set to True, returns the range of elements from the
        sorted set having scores equal or between ``start`` and ``end``.

        ``bylex`` when set to True, returns the range of elements from the
        sorted set between the ``start`` and ``end`` lexicographical closed
        range intervals.
        Valid ``start`` and ``end`` must start with ( or [, in order to specify
        whether the range interval is exclusive or inclusive, respectively.

        ``offset`` and ``num`` are specified, then return a slice of the range.
        Can't be provided when using ``bylex``.

        For more information see https://redis.io/commands/zrange
        """
        # Need to support ``desc`` also when using old redis version
        # because it was supported in 3.5.3 (of redis-py)
        if not byscore and not bylex and (offset is None and num is None) and desc:
            return self.zrevrange(name, start, end, withscores, score_cast_func)

        return self._zrange(
            "ZRANGE",
            None,
            name,
            start,
            end,
            desc,
            byscore,
            bylex,
            withscores,
            score_cast_func,
            offset,
            num,
        )

    def zrevrange(
        self,
        name: KeyT,
        start: int,
        end: int,
        withscores: bool = False,
        score_cast_func: Union[type, Callable] = float,
    ) -> ResponseT:
        """
        Return a range of values from sorted set ``name`` between
        ``start`` and ``end`` sorted in descending order.

        ``start`` and ``end`` can be negative, indicating the end of the range.

        ``withscores`` indicates to return the scores along with the values
        The return type is a list of (value, score) pairs

        ``score_cast_func`` a callable used to cast the score return value

        For more information see https://redis.io/commands/zrevrange
        """
        pieces = ["ZREVRANGE", name, start, end]
        if withscores:
            pieces.append(b"WITHSCORES")
        options = {"withscores": withscores, "score_cast_func": score_cast_func}
        return self.execute_command(*pieces, **options)

    def zrangestore(
        self,
        dest: KeyT,
        name: KeyT,
        start: int,
        end: int,
        byscore: bool = False,
        bylex: bool = False,
        desc: bool = False,
        offset: Union[int, None] = None,
        num: Union[int, None] = None,
    ) -> ResponseT:
        """
        Stores in ``dest`` the result of a range of values from sorted set
        ``name`` between ``start`` and ``end`` sorted in ascending order.

        ``start`` and ``end`` can be negative, indicating the end of the range.

        ``byscore`` when set to True, returns the range of elements from the
        sorted set having scores equal or between ``start`` and ``end``.

        ``bylex`` when set to True, returns the range of elements from the
        sorted set between the ``start`` and ``end`` lexicographical closed
        range intervals.
        Valid ``start`` and ``end`` must start with ( or [, in order to specify
        whether the range interval is exclusive or inclusive, respectively.

        ``desc`` a boolean indicating whether to sort the results in reversed
        order.

        ``offset`` and ``num`` are specified, then return a slice of the range.
        Can't be provided when using ``bylex``.

        For more information see https://redis.io/commands/zrangestore
        """
        return self._zrange(
            "ZRANGESTORE",
            dest,
            name,
            start,
            end,
            desc,
            byscore,
            bylex,
            False,
            None,
            offset,
            num,
        )

    def zrangebylex(
        self,
        name: KeyT,
        min: EncodableT,
        max: EncodableT,
        start: Union[int, None] = None,
        num: Union[int, None] = None,
    ) -> ResponseT:
        """
        Return the lexicographical range of values from sorted set ``name``
        between ``min`` and ``max``.

        If ``start`` and ``num`` are specified, then return a slice of the
        range.

        For more information see https://redis.io/commands/zrangebylex
        """
        if (start is not None and num is None) or (num is not None and start is None):
            raise DataError("``start`` and ``num`` must both be specified")
        pieces = ["ZRANGEBYLEX", name, min, max]
        if start is not None and num is not None:
            pieces.extend([b"LIMIT", start, num])
        return self.execute_command(*pieces)

    def zrevrangebylex(
        self,
        name: KeyT,
        max: EncodableT,
        min: EncodableT,
        start: Union[int, None] = None,
        num: Union[int, None] = None,
    ) -> ResponseT:
        """
        Return the reversed lexicographical range of values from sorted set
        ``name`` between ``max`` and ``min``.

        If ``start`` and ``num`` are specified, then return a slice of the
        range.

        For more information see https://redis.io/commands/zrevrangebylex
        """
        if (start is not None and num is None) or (num is not None and start is None):
            raise DataError("``start`` and ``num`` must both be specified")
        pieces = ["ZREVRANGEBYLEX", name, max, min]
        if start is not None and num is not None:
            pieces.extend(["LIMIT", start, num])
        return self.execute_command(*pieces)

    def zrangebyscore(
        self,
        name: KeyT,
        min: ZScoreBoundT,
        max: ZScoreBoundT,
        start: Union[int, None] = None,
        num: Union[int, None] = None,
        withscores: bool = False,
        score_cast_func: Union[type, Callable] = float,
    ) -> ResponseT:
        """
        Return a range of values from the sorted set ``name`` with scores
        between ``min`` and ``max``.

        If ``start`` and ``num`` are specified, then return a slice
        of the range.

        ``withscores`` indicates to return the scores along with the values.
        The return type is a list of (value, score) pairs

        `score_cast_func`` a callable used to cast the score return value

        For more information see https://redis.io/commands/zrangebyscore
        """
        if (start is not None and num is None) or (num is not None and start is None):
            raise DataError("``start`` and ``num`` must both be specified")
        pieces = ["ZRANGEBYSCORE", name, min, max]
        if start is not None and num is not None:
            pieces.extend(["LIMIT", start, num])
        if withscores:
            pieces.append("WITHSCORES")
        options = {"withscores": withscores, "score_cast_func": score_cast_func}
        return self.execute_command(*pieces, **options)

    def zrevrangebyscore(
        self,
        name: KeyT,
        max: ZScoreBoundT,
        min: ZScoreBoundT,
        start: Union[int, None] = None,
        num: Union[int, None] = None,
        withscores: bool = False,
        score_cast_func: Union[type, Callable] = float,
    ):
        """
        Return a range of values from the sorted set ``name`` with scores
        between ``min`` and ``max`` in descending order.

        If ``start`` and ``num`` are specified, then return a slice
        of the range.

        ``withscores`` indicates to return the scores along with the values.
        The return type is a list of (value, score) pairs

        ``score_cast_func`` a callable used to cast the score return value

        For more information see https://redis.io/commands/zrevrangebyscore
        """
        if (start is not None and num is None) or (num is not None and start is None):
            raise DataError("``start`` and ``num`` must both be specified")
        pieces = ["ZREVRANGEBYSCORE", name, max, min]
        if start is not None and num is not None:
            pieces.extend(["LIMIT", start, num])
        if withscores:
            pieces.append("WITHSCORES")
        options = {"withscores": withscores, "score_cast_func": score_cast_func}
        return self.execute_command(*pieces, **options)

    def zrank(self, name: KeyT, value: EncodableT) -> ResponseT:
        """
        Returns a 0-based value indicating the rank of ``value`` in sorted set
        ``name``

        For more information see https://redis.io/commands/zrank
        """
        return self.execute_command("ZRANK", name, value)

    def zrem(self, name: KeyT, *values: FieldT) -> ResponseT:
        """
        Remove member ``values`` from sorted set ``name``

        For more information see https://redis.io/commands/zrem
        """
        return self.execute_command("ZREM", name, *values)

    def zremrangebylex(self, name: KeyT, min: EncodableT, max: EncodableT) -> ResponseT:
        """
        Remove all elements in the sorted set ``name`` between the
        lexicographical range specified by ``min`` and ``max``.

        Returns the number of elements removed.

        For more information see https://redis.io/commands/zremrangebylex
        """
        return self.execute_command("ZREMRANGEBYLEX", name, min, max)

    def zremrangebyrank(self, name: KeyT, min: int, max: int) -> ResponseT:
        """
        Remove all elements in the sorted set ``name`` with ranks between
        ``min`` and ``max``. Values are 0-based, ordered from smallest score
        to largest. Values can be negative indicating the highest scores.
        Returns the number of elements removed

        For more information see https://redis.io/commands/zremrangebyrank
        """
        return self.execute_command("ZREMRANGEBYRANK", name, min, max)

    def zremrangebyscore(
        self, name: KeyT, min: ZScoreBoundT, max: ZScoreBoundT
    ) -> ResponseT:
        """
        Remove all elements in the sorted set ``name`` with scores
        between ``min`` and ``max``. Returns the number of elements removed.

        For more information see https://redis.io/commands/zremrangebyscore
        """
        return self.execute_command("ZREMRANGEBYSCORE", name, min, max)

    def zrevrank(self, name: KeyT, value: EncodableT) -> ResponseT:
        """
        Returns a 0-based value indicating the descending rank of
        ``value`` in sorted set ``name``

        For more information see https://redis.io/commands/zrevrank
        """
        return self.execute_command("ZREVRANK", name, value)

    def zscore(self, name: KeyT, value: EncodableT) -> ResponseT:
        """
        Return the score of element ``value`` in sorted set ``name``

        For more information see https://redis.io/commands/zscore
        """
        return self.execute_command("ZSCORE", name, value)

    def zunion(
        self,
        keys: Union[Sequence[KeyT], Mapping[AnyKeyT, float]],
        aggregate: Union[str, None] = None,
        withscores: bool = False,
    ) -> ResponseT:
        """
        Return the union of multiple sorted sets specified by ``keys``.
        ``keys`` can be provided as dictionary of keys and their weights.
        Scores will be aggregated based on the ``aggregate``, or SUM if
        none is provided.

        For more information see https://redis.io/commands/zunion
        """
        return self._zaggregate("ZUNION", None, keys, aggregate, withscores=withscores)

    def zunionstore(
        self,
        dest: KeyT,
        keys: Union[Sequence[KeyT], Mapping[AnyKeyT, float]],
        aggregate: Union[str, None] = None,
    ) -> ResponseT:
        """
        Union multiple sorted sets specified by ``keys`` into
        a new sorted set, ``dest``. Scores in the destination will be
        aggregated based on the ``aggregate``, or SUM if none is provided.

        For more information see https://redis.io/commands/zunionstore
        """
        return self._zaggregate("ZUNIONSTORE", dest, keys, aggregate)

    def zmscore(self, key: KeyT, members: List[str]) -> ResponseT:
        """
        Returns the scores associated with the specified members
        in the sorted set stored at key.
        ``members`` should be a list of the member name.
        Return type is a list of score.
        If the member does not exist, a None will be returned
        in corresponding position.

        For more information see https://redis.io/commands/zmscore
        """
        if not members:
            raise DataError("ZMSCORE members must be a non-empty list")
        pieces = [key] + members
        return self.execute_command("ZMSCORE", *pieces)

    def _zaggregate(
        self,
        command: str,
        dest: Union[KeyT, None],
        keys: Union[Sequence[KeyT], Mapping[AnyKeyT, float]],
        aggregate: Union[str, None] = None,
        **options,
    ) -> ResponseT:
        pieces: list[EncodableT] = [command]
        if dest is not None:
            pieces.append(dest)
        pieces.append(len(keys))
        if isinstance(keys, dict):
            keys, weights = keys.keys(), keys.values()
        else:
            weights = None
        pieces.extend(keys)
        if weights:
            pieces.append(b"WEIGHTS")
            pieces.extend(weights)
        if aggregate:
            if aggregate.upper() in ["SUM", "MIN", "MAX"]:
                pieces.append(b"AGGREGATE")
                pieces.append(aggregate)
            else:
                raise DataError("aggregate can be sum, min or max.")
        if options.get("withscores", False):
            pieces.append(b"WITHSCORES")
        return self.execute_command(*pieces, **options)


AsyncSortedSetCommands = SortedSetCommands


class HyperlogCommands(CommandsProtocol):
    """
    Redis commands of HyperLogLogs data type.
    see: https://redis.io/topics/data-types-intro#hyperloglogs
    """

    def pfadd(self, name: KeyT, *values: FieldT) -> ResponseT:
        """
        Adds the specified elements to the specified HyperLogLog.

        For more information see https://redis.io/commands/pfadd
        """
        return self.execute_command("PFADD", name, *values)

    def pfcount(self, *sources: KeyT) -> ResponseT:
        """
        Return the approximated cardinality of
        the set observed by the HyperLogLog at key(s).

        For more information see https://redis.io/commands/pfcount
        """
        return self.execute_command("PFCOUNT", *sources)

    def pfmerge(self, dest: KeyT, *sources: KeyT) -> ResponseT:
        """
        Merge N different HyperLogLogs into a single one.

        For more information see https://redis.io/commands/pfmerge
        """
        return self.execute_command("PFMERGE", dest, *sources)


AsyncHyperlogCommands = HyperlogCommands


class HashCommands(CommandsProtocol):
    """
    Redis commands for Hash data type.
    see: https://redis.io/topics/data-types-intro#redis-hashes
    """

    def hdel(self, name: str, *keys: List) -> Union[Awaitable[int], int]:
        """
        Delete ``keys`` from hash ``name``

        For more information see https://redis.io/commands/hdel
        """
        return self.execute_command("HDEL", name, *keys)

    def hexists(self, name: str, key: str) -> Union[Awaitable[bool], bool]:
        """
        Returns a boolean indicating if ``key`` exists within hash ``name``

        For more information see https://redis.io/commands/hexists
        """
        return self.execute_command("HEXISTS", name, key)

    def hget(
        self, name: str, key: str
    ) -> Union[Awaitable[Optional[str]], Optional[str]]:
        """
        Return the value of ``key`` within the hash ``name``

        For more information see https://redis.io/commands/hget
        """
        return self.execute_command("HGET", name, key)

    def hgetall(self, name: str) -> Union[Awaitable[dict], dict]:
        """
        Return a Python dict of the hash's name/value pairs

        For more information see https://redis.io/commands/hgetall
        """
        return self.execute_command("HGETALL", name)

    def hincrby(
        self, name: str, key: str, amount: int = 1
    ) -> Union[Awaitable[int], int]:
        """
        Increment the value of ``key`` in hash ``name`` by ``amount``

        For more information see https://redis.io/commands/hincrby
        """
        return self.execute_command("HINCRBY", name, key, amount)

    def hincrbyfloat(
        self, name: str, key: str, amount: float = 1.0
    ) -> Union[Awaitable[float], float]:
        """
        Increment the value of ``key`` in hash ``name`` by floating ``amount``

        For more information see https://redis.io/commands/hincrbyfloat
        """
        return self.execute_command("HINCRBYFLOAT", name, key, amount)

    def hkeys(self, name: str) -> Union[Awaitable[List], List]:
        """
        Return the list of keys within hash ``name``

        For more information see https://redis.io/commands/hkeys
        """
        return self.execute_command("HKEYS", name)

    def hlen(self, name: str) -> Union[Awaitable[int], int]:
        """
        Return the number of elements in hash ``name``

        For more information see https://redis.io/commands/hlen
        """
        return self.execute_command("HLEN", name)

    def hset(
        self,
        name: str,
        key: Optional[str] = None,
        value: Optional[str] = None,
        mapping: Optional[dict] = None,
        items: Optional[list] = None,
    ) -> Union[Awaitable[int], int]:
        """
        Set ``key`` to ``value`` within hash ``name``,
        ``mapping`` accepts a dict of key/value pairs that will be
        added to hash ``name``.
        ``items`` accepts a list of key/value pairs that will be
        added to hash ``name``.
        Returns the number of fields that were added.

        For more information see https://redis.io/commands/hset
        """
        if key is None and not mapping and not items:
            raise DataError("'hset' with no key value pairs")
        items = items or []
        if key is not None:
            items.extend((key, value))
        if mapping:
            for pair in mapping.items():
                items.extend(pair)

        return self.execute_command("HSET", name, *items)

    def hsetnx(self, name: str, key: str, value: str) -> Union[Awaitable[bool], bool]:
        """
        Set ``key`` to ``value`` within hash ``name`` if ``key`` does not
        exist.  Returns 1 if HSETNX created a field, otherwise 0.

        For more information see https://redis.io/commands/hsetnx
        """
        return self.execute_command("HSETNX", name, key, value)

    def hmset(self, name: str, mapping: dict) -> Union[Awaitable[str], str]:
        """
        Set key to value within hash ``name`` for each corresponding
        key and value from the ``mapping`` dict.

        For more information see https://redis.io/commands/hmset
        """
        warnings.warn(
            f"{self.__class__.__name__}.hmset() is deprecated. "
            f"Use {self.__class__.__name__}.hset() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not mapping:
            raise DataError("'hmset' with 'mapping' of length 0")
        items = []
        for pair in mapping.items():
            items.extend(pair)
        return self.execute_command("HMSET", name, *items)

    def hmget(self, name: str, keys: List, *args: List) -> Union[Awaitable[List], List]:
        """
        Returns a list of values ordered identically to ``keys``

        For more information see https://redis.io/commands/hmget
        """
        args = list_or_args(keys, args)
        return self.execute_command("HMGET", name, *args)

    def hvals(self, name: str) -> Union[Awaitable[List], List]:
        """
        Return the list of values within hash ``name``

        For more information see https://redis.io/commands/hvals
        """
        return self.execute_command("HVALS", name)

    def hstrlen(self, name: str, key: str) -> Union[Awaitable[int], int]:
        """
        Return the number of bytes stored in the value of ``key``
        within hash ``name``

        For more information see https://redis.io/commands/hstrlen
        """
        return self.execute_command("HSTRLEN", name, key)


AsyncHashCommands = HashCommands


class Script:
    """
    An executable Lua script object returned by ``register_script``
    """

    def __init__(self, registered_client: "Redis", script: ScriptTextT):
        self.registered_client = registered_client
        self.script = script
        # Precalculate and store the SHA1 hex digest of the script.

        if isinstance(script, str):
            # We need the encoding from the client in order to generate an
            # accurate byte representation of the script
            try:
                encoder = registered_client.connection_pool.get_encoder()
            except AttributeError:
                # Cluster
                encoder = registered_client.get_encoder()
            script = encoder.encode(script)
        self.sha = hashlib.sha1(script).hexdigest()

    def __call__(
        self,
        keys: Union[Sequence[KeyT], None] = None,
        args: Union[Iterable[EncodableT], None] = None,
        client: Union["Redis", None] = None,
    ):
        """Execute the script, passing any required ``args``"""
        keys = keys or []
        args = args or []
        if client is None:
            client = self.registered_client
        args = tuple(keys) + tuple(args)
        # make sure the Redis server knows about the script
        from redis.client import Pipeline

        if isinstance(client, Pipeline):
            # Make sure the pipeline can register the script before executing.
            client.scripts.add(self)
        try:
            return client.evalsha(self.sha, len(keys), *args)
        except NoScriptError:
            # Maybe the client is pointed to a different server than the client
            # that created this instance?
            # Overwrite the sha just in case there was a discrepancy.
            self.sha = client.script_load(self.script)
            return client.evalsha(self.sha, len(keys), *args)


class AsyncScript:
    """
    An executable Lua script object returned by ``register_script``
    """

    def __init__(self, registered_client: "AsyncRedis", script: ScriptTextT):
        self.registered_client = registered_client
        self.script = script
        # Precalculate and store the SHA1 hex digest of the script.

        if isinstance(script, str):
            # We need the encoding from the client in order to generate an
            # accurate byte representation of the script
            try:
                encoder = registered_client.connection_pool.get_encoder()
            except AttributeError:
                # Cluster
                encoder = registered_client.get_encoder()
            script = encoder.encode(script)
        self.sha = hashlib.sha1(script).hexdigest()

    async def __call__(
        self,
        keys: Union[Sequence[KeyT], None] = None,
        args: Union[Iterable[EncodableT], None] = None,
        client: Union["AsyncRedis", None] = None,
    ):
        """Execute the script, passing any required ``args``"""
        keys = keys or []
        args = args or []
        if client is None:
            client = self.registered_client
        args = tuple(keys) + tuple(args)
        # make sure the Redis server knows about the script
        from redis.asyncio.client import Pipeline

        if isinstance(client, Pipeline):
            # Make sure the pipeline can register the script before executing.
            client.scripts.add(self)
        try:
            return await client.evalsha(self.sha, len(keys), *args)
        except NoScriptError:
            # Maybe the client is pointed to a different server than the client
            # that created this instance?
            # Overwrite the sha just in case there was a discrepancy.
            self.sha = await client.script_load(self.script)
            return await client.evalsha(self.sha, len(keys), *args)


class PubSubCommands(CommandsProtocol):
    """
    Redis PubSub commands.
    see https://redis.io/topics/pubsub
    """

    def publish(self, channel: ChannelT, message: EncodableT, **kwargs) -> ResponseT:
        """
        Publish ``message`` on ``channel``.
        Returns the number of subscribers the message was delivered to.

        For more information see https://redis.io/commands/publish
        """
        return self.execute_command("PUBLISH", channel, message, **kwargs)

    def pubsub_channels(self, pattern: PatternT = "*", **kwargs) -> ResponseT:
        """
        Return a list of channels that have at least one subscriber

        For more information see https://redis.io/commands/pubsub-channels
        """
        return self.execute_command("PUBSUB CHANNELS", pattern, **kwargs)

    def pubsub_numpat(self, **kwargs) -> ResponseT:
        """
        Returns the number of subscriptions to patterns

        For more information see https://redis.io/commands/pubsub-numpat
        """
        return self.execute_command("PUBSUB NUMPAT", **kwargs)

    def pubsub_numsub(self, *args: ChannelT, **kwargs) -> ResponseT:
        """
        Return a list of (channel, number of subscribers) tuples
        for each channel given in ``*args``

        For more information see https://redis.io/commands/pubsub-numsub
        """
        return self.execute_command("PUBSUB NUMSUB", *args, **kwargs)


AsyncPubSubCommands = PubSubCommands


class ScriptCommands(CommandsProtocol):
    """
    Redis Lua script commands. see:
    https://redis.com/ebook/part-3-next-steps/chapter-11-scripting-redis-with-lua/
    """

    def _eval(
        self, command: str, script: str, numkeys: int, *keys_and_args: list
    ) -> Union[Awaitable[str], str]:
        return self.execute_command(command, script, numkeys, *keys_and_args)

    def eval(
        self, script: str, numkeys: int, *keys_and_args: list
    ) -> Union[Awaitable[str], str]:
        """
        Execute the Lua ``script``, specifying the ``numkeys`` the script
        will touch and the key names and argument values in ``keys_and_args``.
        Returns the result of the script.

        In practice, use the object returned by ``register_script``. This
        function exists purely for Redis API completion.

        For more information see  https://redis.io/commands/eval
        """
        return self._eval("EVAL", script, numkeys, *keys_and_args)

    def eval_ro(
        self, script: str, numkeys: int, *keys_and_args: list
    ) -> Union[Awaitable[str], str]:
        """
        The read-only variant of the EVAL command

        Execute the read-only Lua ``script`` specifying the ``numkeys`` the script
        will touch and the key names and argument values in ``keys_and_args``.
        Returns the result of the script.

        For more information see  https://redis.io/commands/eval_ro
        """
        return self._eval("EVAL_RO", script, numkeys, *keys_and_args)

    def _evalsha(
        self, command: str, sha: str, numkeys: int, *keys_and_args: list
    ) -> Union[Awaitable[str], str]:
        return self.execute_command(command, sha, numkeys, *keys_and_args)

    def evalsha(
        self, sha: str, numkeys: int, *keys_and_args: list
    ) -> Union[Awaitable[str], str]:
        """
        Use the ``sha`` to execute a Lua script already registered via EVAL
        or SCRIPT LOAD. Specify the ``numkeys`` the script will touch and the
        key names and argument values in ``keys_and_args``. Returns the result
        of the script.

        In practice, use the object returned by ``register_script``. This
        function exists purely for Redis API completion.

        For more information see  https://redis.io/commands/evalsha
        """
        return self._evalsha("EVALSHA", sha, numkeys, *keys_and_args)

    def evalsha_ro(
        self, sha: str, numkeys: int, *keys_and_args: list
    ) -> Union[Awaitable[str], str]:
        """
        The read-only variant of the EVALSHA command

        Use the ``sha`` to execute a read-only Lua script already registered via EVAL
        or SCRIPT LOAD. Specify the ``numkeys`` the script will touch and the
        key names and argument values in ``keys_and_args``. Returns the result
        of the script.

        For more information see  https://redis.io/commands/evalsha_ro
        """
        return self._evalsha("EVALSHA_RO", sha, numkeys, *keys_and_args)

    def script_exists(self, *args: str) -> ResponseT:
        """
        Check if a script exists in the script cache by specifying the SHAs of
        each script as ``args``. Returns a list of boolean values indicating if
        if each already script exists in the cache.

        For more information see  https://redis.io/commands/script-exists
        """
        return self.execute_command("SCRIPT EXISTS", *args)

    def script_debug(self, *args) -> None:
        raise NotImplementedError(
            "SCRIPT DEBUG is intentionally not implemented in the client."
        )

    def script_flush(
        self, sync_type: Union[Literal["SYNC"], Literal["ASYNC"]] = None
    ) -> ResponseT:
        """Flush all scripts from the script cache.
        ``sync_type`` is by default SYNC (synchronous) but it can also be
                      ASYNC.
        For more information see  https://redis.io/commands/script-flush
        """

        # Redis pre 6 had no sync_type.
        if sync_type not in ["SYNC", "ASYNC", None]:
            raise DataError(
                "SCRIPT FLUSH defaults to SYNC in redis > 6.2, or "
                "accepts SYNC/ASYNC. For older versions, "
                "of redis leave as None."
            )
        if sync_type is None:
            pieces = []
        else:
            pieces = [sync_type]
        return self.execute_command("SCRIPT FLUSH", *pieces)

    def script_kill(self) -> ResponseT:
        """
        Kill the currently executing Lua script

        For more information see https://redis.io/commands/script-kill
        """
        return self.execute_command("SCRIPT KILL")

    def script_load(self, script: ScriptTextT) -> ResponseT:
        """
        Load a Lua ``script`` into the script cache. Returns the SHA.

        For more information see https://redis.io/commands/script-load
        """
        return self.execute_command("SCRIPT LOAD", script)

    def register_script(self: "Redis", script: ScriptTextT) -> Script:
        """
        Register a Lua ``script`` specifying the ``keys`` it will touch.
        Returns a Script object that is callable and hides the complexity of
        deal with scripts, keys, and shas. This is the preferred way to work
        with Lua scripts.
        """
        return Script(self, script)


class AsyncScriptCommands(ScriptCommands):
    async def script_debug(self, *args) -> None:
        return super().script_debug()

    def register_script(self: "AsyncRedis", script: ScriptTextT) -> AsyncScript:
        """
        Register a Lua ``script`` specifying the ``keys`` it will touch.
        Returns a Script object that is callable and hides the complexity of
        deal with scripts, keys, and shas. This is the preferred way to work
        with Lua scripts.
        """
        return AsyncScript(self, script)


class GeoCommands(CommandsProtocol):
    """
    Redis Geospatial commands.
    see: https://redis.com/redis-best-practices/indexing-patterns/geospatial/
    """

    def geoadd(
        self,
        name: KeyT,
        values: Sequence[EncodableT],
        nx: bool = False,
        xx: bool = False,
        ch: bool = False,
    ) -> ResponseT:
        """
        Add the specified geospatial items to the specified key identified
        by the ``name`` argument. The Geospatial items are given as ordered
        members of the ``values`` argument, each item or place is formed by
        the triad longitude, latitude and name.

        Note: You can use ZREM to remove elements.

        ``nx`` forces ZADD to only create new elements and not to update
        scores for elements that already exist.

        ``xx`` forces ZADD to only update scores of elements that already
        exist. New elements will not be added.

        ``ch`` modifies the return value to be the numbers of elements changed.
        Changed elements include new elements that were added and elements
        whose scores changed.

        For more information see https://redis.io/commands/geoadd
        """
        if nx and xx:
            raise DataError("GEOADD allows either 'nx' or 'xx', not both")
        if len(values) % 3 != 0:
            raise DataError("GEOADD requires places with lon, lat and name values")
        pieces = [name]
        if nx:
            pieces.append("NX")
        if xx:
            pieces.append("XX")
        if ch:
            pieces.append("CH")
        pieces.extend(values)
        return self.execute_command("GEOADD", *pieces)

    def geodist(
        self, name: KeyT, place1: FieldT, place2: FieldT, unit: Union[str, None] = None
    ) -> ResponseT:
        """
        Return the distance between ``place1`` and ``place2`` members of the
        ``name`` key.
        The units must be one of the following : m, km mi, ft. By default
        meters are used.

        For more information see https://redis.io/commands/geodist
        """
        pieces: list[EncodableT] = [name, place1, place2]
        if unit and unit not in ("m", "km", "mi", "ft"):
            raise DataError("GEODIST invalid unit")
        elif unit:
            pieces.append(unit)
        return self.execute_command("GEODIST", *pieces)

    def geohash(self, name: KeyT, *values: FieldT) -> ResponseT:
        """
        Return the geo hash string for each item of ``values`` members of
        the specified key identified by the ``name`` argument.

        For more information see https://redis.io/commands/geohash
        """
        return self.execute_command("GEOHASH", name, *values)

    def geopos(self, name: KeyT, *values: FieldT) -> ResponseT:
        """
        Return the positions of each item of ``values`` as members of
        the specified key identified by the ``name`` argument. Each position
        is represented by the pairs lon and lat.

        For more information see https://redis.io/commands/geopos
        """
        return self.execute_command("GEOPOS", name, *values)

    def georadius(
        self,
        name: KeyT,
        longitude: float,
        latitude: float,
        radius: float,
        unit: Union[str, None] = None,
        withdist: bool = False,
        withcoord: bool = False,
        withhash: bool = False,
        count: Union[int, None] = None,
        sort: Union[str, None] = None,
        store: Union[KeyT, None] = None,
        store_dist: Union[KeyT, None] = None,
        any: bool = False,
    ) -> ResponseT:
        """
        Return the members of the specified key identified by the
        ``name`` argument which are within the borders of the area specified
        with the ``latitude`` and ``longitude`` location and the maximum
        distance from the center specified by the ``radius`` value.

        The units must be one of the following : m, km mi, ft. By default

        ``withdist`` indicates to return the distances of each place.

        ``withcoord`` indicates to return the latitude and longitude of
        each place.

        ``withhash`` indicates to return the geohash string of each place.

        ``count`` indicates to return the number of elements up to N.

        ``sort`` indicates to return the places in a sorted way, ASC for
        nearest to fairest and DESC for fairest to nearest.

        ``store`` indicates to save the places names in a sorted set named
        with a specific key, each element of the destination sorted set is
        populated with the score got from the original geo sorted set.

        ``store_dist`` indicates to save the places names in a sorted set
        named with a specific key, instead of ``store`` the sorted set
        destination score is set with the distance.

        For more information see https://redis.io/commands/georadius
        """
        return self._georadiusgeneric(
            "GEORADIUS",
            name,
            longitude,
            latitude,
            radius,
            unit=unit,
            withdist=withdist,
            withcoord=withcoord,
            withhash=withhash,
            count=count,
            sort=sort,
            store=store,
            store_dist=store_dist,
            any=any,
        )

    def georadiusbymember(
        self,
        name: KeyT,
        member: FieldT,
        radius: float,
        unit: Union[str, None] = None,
        withdist: bool = False,
        withcoord: bool = False,
        withhash: bool = False,
        count: Union[int, None] = None,
        sort: Union[str, None] = None,
        store: Union[KeyT, None] = None,
        store_dist: Union[KeyT, None] = None,
        any: bool = False,
    ) -> ResponseT:
        """
        This command is exactly like ``georadius`` with the sole difference
        that instead of taking, as the center of the area to query, a longitude
        and latitude value, it takes the name of a member already existing
        inside the geospatial index represented by the sorted set.

        For more information see https://redis.io/commands/georadiusbymember
        """
        return self._georadiusgeneric(
            "GEORADIUSBYMEMBER",
            name,
            member,
            radius,
            unit=unit,
            withdist=withdist,
            withcoord=withcoord,
            withhash=withhash,
            count=count,
            sort=sort,
            store=store,
            store_dist=store_dist,
            any=any,
        )

    def _georadiusgeneric(
        self, command: str, *args: EncodableT, **kwargs: Union[EncodableT, None]
    ) -> ResponseT:
        pieces = list(args)
        if kwargs["unit"] and kwargs["unit"] not in ("m", "km", "mi", "ft"):
            raise DataError("GEORADIUS invalid unit")
        elif kwargs["unit"]:
            pieces.append(kwargs["unit"])
        else:
            pieces.append("m")

        if kwargs["any"] and kwargs["count"] is None:
            raise DataError("``any`` can't be provided without ``count``")

        for arg_name, byte_repr in (
            ("withdist", "WITHDIST"),
            ("withcoord", "WITHCOORD"),
            ("withhash", "WITHHASH"),
        ):
            if kwargs[arg_name]:
                pieces.append(byte_repr)

        if kwargs["count"] is not None:
            pieces.extend(["COUNT", kwargs["count"]])
            if kwargs["any"]:
                pieces.append("ANY")

        if kwargs["sort"]:
            if kwargs["sort"] == "ASC":
                pieces.append("ASC")
            elif kwargs["sort"] == "DESC":
                pieces.append("DESC")
            else:
                raise DataError("GEORADIUS invalid sort")

        if kwargs["store"] and kwargs["store_dist"]:
            raise DataError("GEORADIUS store and store_dist cant be set together")

        if kwargs["store"]:
            pieces.extend([b"STORE", kwargs["store"]])

        if kwargs["store_dist"]:
            pieces.extend([b"STOREDIST", kwargs["store_dist"]])

        return self.execute_command(command, *pieces, **kwargs)

    def geosearch(
        self,
        name: KeyT,
        member: Union[FieldT, None] = None,
        longitude: Union[float, None] = None,
        latitude: Union[float, None] = None,
        unit: str = "m",
        radius: Union[float, None] = None,
        width: Union[float, None] = None,
        height: Union[float, None] = None,
        sort: Union[str, None] = None,
        count: Union[int, None] = None,
        any: bool = False,
        withcoord: bool = False,
        withdist: bool = False,
        withhash: bool = False,
    ) -> ResponseT:
        """
        Return the members of specified key identified by the
        ``name`` argument, which are within the borders of the
        area specified by a given shape. This command extends the
        GEORADIUS command, so in addition to searching within circular
        areas, it supports searching within rectangular areas.
        This command should be used in place of the deprecated
        GEORADIUS and GEORADIUSBYMEMBER commands.
        ``member`` Use the position of the given existing
         member in the sorted set. Can't be given with ``longitude``
         and ``latitude``.
        ``longitude`` and ``latitude`` Use the position given by
        this coordinates. Can't be given with ``member``
        ``radius`` Similar to GEORADIUS, search inside circular
        area according the given radius. Can't be given with
        ``height`` and ``width``.
        ``height`` and ``width`` Search inside an axis-aligned
        rectangle, determined by the given height and width.
        Can't be given with ``radius``
        ``unit`` must be one of the following : m, km, mi, ft.
        `m` for meters (the default value), `km` for kilometers,
        `mi` for miles and `ft` for feet.
        ``sort`` indicates to return the places in a sorted way,
        ASC for nearest to furthest and DESC for furthest to nearest.
        ``count`` limit the results to the first count matching items.
        ``any`` is set to True, the command will return as soon as
        enough matches are found. Can't be provided without ``count``
        ``withdist`` indicates to return the distances of each place.
        ``withcoord`` indicates to return the latitude and longitude of
        each place.
        ``withhash`` indicates to return the geohash string of each place.

        For more information see https://redis.io/commands/geosearch
        """

        return self._geosearchgeneric(
            "GEOSEARCH",
            name,
            member=member,
            longitude=longitude,
            latitude=latitude,
            unit=unit,
            radius=radius,
            width=width,
            height=height,
            sort=sort,
            count=count,
            any=any,
            withcoord=withcoord,
            withdist=withdist,
            withhash=withhash,
            store=None,
            store_dist=None,
        )

    def geosearchstore(
        self,
        dest: KeyT,
        name: KeyT,
        member: Union[FieldT, None] = None,
        longitude: Union[float, None] = None,
        latitude: Union[float, None] = None,
        unit: str = "m",
        radius: Union[float, None] = None,
        width: Union[float, None] = None,
        height: Union[float, None] = None,
        sort: Union[str, None] = None,
        count: Union[int, None] = None,
        any: bool = False,
        storedist: bool = False,
    ) -> ResponseT:
        """
        This command is like GEOSEARCH, but stores the result in
        ``dest``. By default, it stores the results in the destination
        sorted set with their geospatial information.
        if ``store_dist`` set to True, the command will stores the
        items in a sorted set populated with their distance from the
        center of the circle or box, as a floating-point number.

        For more information see https://redis.io/commands/geosearchstore
        """
        return self._geosearchgeneric(
            "GEOSEARCHSTORE",
            dest,
            name,
            member=member,
            longitude=longitude,
            latitude=latitude,
            unit=unit,
            radius=radius,
            width=width,
            height=height,
            sort=sort,
            count=count,
            any=any,
            withcoord=None,
            withdist=None,
            withhash=None,
            store=None,
            store_dist=storedist,
        )

    def _geosearchgeneric(
        self, command: str, *args: EncodableT, **kwargs: Union[EncodableT, None]
    ) -> ResponseT:
        pieces = list(args)

        # FROMMEMBER or FROMLONLAT
        if kwargs["member"] is None:
            if kwargs["longitude"] is None or kwargs["latitude"] is None:
                raise DataError("GEOSEARCH must have member or longitude and latitude")
        if kwargs["member"]:
            if kwargs["longitude"] or kwargs["latitude"]:
                raise DataError(
                    "GEOSEARCH member and longitude or latitude cant be set together"
                )
            pieces.extend([b"FROMMEMBER", kwargs["member"]])
        if kwargs["longitude"] is not None and kwargs["latitude"] is not None:
            pieces.extend([b"FROMLONLAT", kwargs["longitude"], kwargs["latitude"]])

        # BYRADIUS or BYBOX
        if kwargs["radius"] is None:
            if kwargs["width"] is None or kwargs["height"] is None:
                raise DataError("GEOSEARCH must have radius or width and height")
        if kwargs["unit"] is None:
            raise DataError("GEOSEARCH must have unit")
        if kwargs["unit"].lower() not in ("m", "km", "mi", "ft"):
            raise DataError("GEOSEARCH invalid unit")
        if kwargs["radius"]:
            if kwargs["width"] or kwargs["height"]:
                raise DataError(
                    "GEOSEARCH radius and width or height cant be set together"
                )
            pieces.extend([b"BYRADIUS", kwargs["radius"], kwargs["unit"]])
        if kwargs["width"] and kwargs["height"]:
            pieces.extend([b"BYBOX", kwargs["width"], kwargs["height"], kwargs["unit"]])

        # sort
        if kwargs["sort"]:
            if kwargs["sort"].upper() == "ASC":
                pieces.append(b"ASC")
            elif kwargs["sort"].upper() == "DESC":
                pieces.append(b"DESC")
            else:
                raise DataError("GEOSEARCH invalid sort")

        # count any
        if kwargs["count"]:
            pieces.extend([b"COUNT", kwargs["count"]])
            if kwargs["any"]:
                pieces.append(b"ANY")
        elif kwargs["any"]:
            raise DataError("GEOSEARCH ``any`` can't be provided without count")

        # other properties
        for arg_name, byte_repr in (
            ("withdist", b"WITHDIST"),
            ("withcoord", b"WITHCOORD"),
            ("withhash", b"WITHHASH"),
            ("store_dist", b"STOREDIST"),
        ):
            if kwargs[arg_name]:
                pieces.append(byte_repr)

        return self.execute_command(command, *pieces, **kwargs)


AsyncGeoCommands = GeoCommands


class ModuleCommands(CommandsProtocol):
    """
    Redis Module commands.
    see: https://redis.io/topics/modules-intro
    """

    def module_load(self, path, *args) -> ResponseT:
        """
        Loads the module from ``path``.
        Passes all ``*args`` to the module, during loading.
        Raises ``ModuleError`` if a module is not found at ``path``.

        For more information see https://redis.io/commands/module-load
        """
        return self.execute_command("MODULE LOAD", path, *args)

    def module_loadex(
        self,
        path: str,
        options: Optional[List[str]] = None,
        args: Optional[List[str]] = None,
    ) -> ResponseT:
        """
        Loads a module from a dynamic library at runtime with configuration directives.

        For more information see https://redis.io/commands/module-loadex
        """
        pieces = []
        if options is not None:
            pieces.append("CONFIG")
            pieces.extend(options)
        if args is not None:
            pieces.append("ARGS")
            pieces.extend(args)

        return self.execute_command("MODULE LOADEX", path, *pieces)

    def module_unload(self, name) -> ResponseT:
        """
        Unloads the module ``name``.
        Raises ``ModuleError`` if ``name`` is not in loaded modules.

        For more information see https://redis.io/commands/module-unload
        """
        return self.execute_command("MODULE UNLOAD", name)

    def module_list(self) -> ResponseT:
        """
        Returns a list of dictionaries containing the name and version of
        all loaded modules.

        For more information see https://redis.io/commands/module-list
        """
        return self.execute_command("MODULE LIST")

    def command_info(self) -> None:
        raise NotImplementedError(
            "COMMAND INFO is intentionally not implemented in the client."
        )

    def command_count(self) -> ResponseT:
        return self.execute_command("COMMAND COUNT")

    def command_getkeys(self, *args) -> ResponseT:
        return self.execute_command("COMMAND GETKEYS", *args)

    def command(self) -> ResponseT:
        return self.execute_command("COMMAND")


class Script:
    """
    An executable Lua script object returned by ``register_script``
    """

    def __init__(self, registered_client, script):
        self.registered_client = registered_client
        self.script = script
        # Precalculate and store the SHA1 hex digest of the script.

        if isinstance(script, str):
            # We need the encoding from the client in order to generate an
            # accurate byte representation of the script
            encoder = self.get_encoder()
            script = encoder.encode(script)
        self.sha = hashlib.sha1(script).hexdigest()

    def __call__(self, keys=[], args=[], client=None):
        "Execute the script, passing any required ``args``"
        if client is None:
            client = self.registered_client
        args = tuple(keys) + tuple(args)
        # make sure the Redis server knows about the script
        from redis.client import Pipeline

        if isinstance(client, Pipeline):
            # Make sure the pipeline can register the script before executing.
            client.scripts.add(self)
        try:
            return client.evalsha(self.sha, len(keys), *args)
        except NoScriptError:
            # Maybe the client is pointed to a different server than the client
            # that created this instance?
            # Overwrite the sha just in case there was a discrepancy.
            self.sha = client.script_load(self.script)
            return client.evalsha(self.sha, len(keys), *args)

    def get_encoder(self):
        """Get the encoder to encode string scripts into bytes."""
        try:
            return self.registered_client.get_encoder()
        except AttributeError:
            # DEPRECATED
            # In version <=4.1.2, this was the code we used to get the encoder.
            # However, after 4.1.2 we added support for scripting in clustered
            # redis. ClusteredRedis doesn't have a `.connection_pool` attribute
            # so we changed the Script class to use
            # `self.registered_client.get_encoder` (see above).
            # However, that is technically a breaking change, as consumers who
            # use Scripts directly might inject a `registered_client` that
            # doesn't have a `.get_encoder` field. This try/except prevents us
            # from breaking backward-compatibility. Ideally, it would be
            # removed in the next major release.
            return self.registered_client.connection_pool.get_encoder()


class AsyncModuleCommands(ModuleCommands):
    async def command_info(self) -> None:
        return super().command_info()


class ClusterCommands(CommandsProtocol):
    """
    Class for Redis Cluster commands
    """

    def cluster(self, cluster_arg, *args, **kwargs) -> ResponseT:
        return self.execute_command(f"CLUSTER {cluster_arg.upper()}", *args, **kwargs)

    def readwrite(self, **kwargs) -> ResponseT:
        """
        Disables read queries for a connection to a Redis Cluster slave node.

        For more information see https://redis.io/commands/readwrite
        """
        return self.execute_command("READWRITE", **kwargs)

    def readonly(self, **kwargs) -> ResponseT:
        """
        Enables read queries for a connection to a Redis Cluster replica node.

        For more information see https://redis.io/commands/readonly
        """
        return self.execute_command("READONLY", **kwargs)


AsyncClusterCommands = ClusterCommands


class FunctionCommands:
    """
    Redis Function commands
    """

    def function_load(
        self, code: str, replace: Optional[bool] = False
    ) -> Union[Awaitable[str], str]:
        """
        Load a library to Redis.
        :param code: the source code (must start with
        Shebang statement that provides a metadata about the library)
        :param replace: changes the behavior to overwrite the existing library
        with the new contents.
        Return the library name that was loaded.

        For more information see https://redis.io/commands/function-load
        """
        pieces = ["REPLACE"] if replace else []
        pieces.append(code)
        return self.execute_command("FUNCTION LOAD", *pieces)

    def function_delete(self, library: str) -> Union[Awaitable[str], str]:
        """
        Delete the library called ``library`` and all its functions.

        For more information see https://redis.io/commands/function-delete
        """
        return self.execute_command("FUNCTION DELETE", library)

    def function_flush(self, mode: str = "SYNC") -> Union[Awaitable[str], str]:
        """
        Deletes all the libraries.

        For more information see https://redis.io/commands/function-flush
        """
        return self.execute_command("FUNCTION FLUSH", mode)

    def function_list(
        self, library: Optional[str] = "*", withcode: Optional[bool] = False
    ) -> Union[Awaitable[List], List]:
        """
        Return information about the functions and libraries.
        :param library: pecify a pattern for matching library names
        :param withcode: cause the server to include the libraries source
         implementation in the reply
        """
        args = ["LIBRARYNAME", library]
        if withcode:
            args.append("WITHCODE")
        return self.execute_command("FUNCTION LIST", *args)

    def _fcall(
        self, command: str, function, numkeys: int, *keys_and_args: Optional[List]
    ) -> Union[Awaitable[str], str]:
        return self.execute_command(command, function, numkeys, *keys_and_args)

    def fcall(
        self, function, numkeys: int, *keys_and_args: Optional[List]
    ) -> Union[Awaitable[str], str]:
        """
        Invoke a function.

        For more information see https://redis.io/commands/fcall
        """
        return self._fcall("FCALL", function, numkeys, *keys_and_args)

    def fcall_ro(
        self, function, numkeys: int, *keys_and_args: Optional[List]
    ) -> Union[Awaitable[str], str]:
        """
        This is a read-only variant of the FCALL command that cannot
        execute commands that modify data.

        For more information see https://redis.io/commands/fcal_ro
        """
        return self._fcall("FCALL_RO", function, numkeys, *keys_and_args)

    def function_dump(self) -> Union[Awaitable[str], str]:
        """
        Return the serialized payload of loaded libraries.

        For more information see https://redis.io/commands/function-dump
        """
        from redis.client import NEVER_DECODE

        options = {}
        options[NEVER_DECODE] = []

        return self.execute_command("FUNCTION DUMP", **options)

    def function_restore(
        self, payload: str, policy: Optional[str] = "APPEND"
    ) -> Union[Awaitable[str], str]:
        """
        Restore libraries from the serialized ``payload``.
        You can use the optional policy argument to provide a policy
        for handling existing libraries.

        For more information see https://redis.io/commands/function-restore
        """
        return self.execute_command("FUNCTION RESTORE", payload, policy)

    def function_kill(self) -> Union[Awaitable[str], str]:
        """
        Kill a function that is currently executing.

        For more information see https://redis.io/commands/function-kill
        """
        return self.execute_command("FUNCTION KILL")

    def function_stats(self) -> Union[Awaitable[List], List]:
        """
        Return information about the function that's currently running
        and information about the available execution engines.

        For more information see https://redis.io/commands/function-stats
        """
        return self.execute_command("FUNCTION STATS")


AsyncFunctionCommands = FunctionCommands


class DataAccessCommands(
    BasicKeyCommands,
    HyperlogCommands,
    HashCommands,
    GeoCommands,
    ListCommands,
    ScanCommands,
    SetCommands,
    StreamCommands,
    SortedSetCommands,
):
    """
    A class containing all of the implemented data access redis commands.
    This class is to be used as a mixin for synchronous Redis clients.
    """


class AsyncDataAccessCommands(
    AsyncBasicKeyCommands,
    AsyncHyperlogCommands,
    AsyncHashCommands,
    AsyncGeoCommands,
    AsyncListCommands,
    AsyncScanCommands,
    AsyncSetCommands,
    AsyncStreamCommands,
    AsyncSortedSetCommands,
):
    """
    A class containing all of the implemented data access redis commands.
    This class is to be used as a mixin for asynchronous Redis clients.
    """


class CoreCommands(
    ACLCommands,
    ClusterCommands,
    DataAccessCommands,
    ManagementCommands,
    ModuleCommands,
    PubSubCommands,
    ScriptCommands,
    FunctionCommands,
):
    """
    A class containing all of the implemented redis commands. This class is
    to be used as a mixin for synchronous Redis clients.
    """


class AsyncCoreCommands(
    AsyncACLCommands,
    AsyncClusterCommands,
    AsyncDataAccessCommands,
    AsyncManagementCommands,
    AsyncModuleCommands,
    AsyncPubSubCommands,
    AsyncScriptCommands,
    AsyncFunctionCommands,
):
    """
    A class containing all of the implemented redis commands. This class is
    to be used as a mixin for asynchronous Redis clients.
    """
