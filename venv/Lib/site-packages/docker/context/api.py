import json
import os

from docker import errors
from docker.context.config import get_meta_dir
from docker.context.config import METAFILE
from docker.context.config import get_current_context_name
from docker.context.config import write_context_name_to_docker_config
from docker.context import Context


class ContextAPI:
    """Context API.
    Contains methods for context management:
    create, list, remove, get, inspect.
    """
    DEFAULT_CONTEXT = Context("default", "swarm")

    @classmethod
    def create_context(
            cls, name, orchestrator=None, host=None, tls_cfg=None,
            default_namespace=None, skip_tls_verify=False):
        """Creates a new context.
        Returns:
            (Context): a Context object.
        Raises:
            :py:class:`docker.errors.MissingContextParameter`
                If a context name is not provided.
            :py:class:`docker.errors.ContextAlreadyExists`
                If a context with the name already exists.
            :py:class:`docker.errors.ContextException`
                If name is default.

        Example:

        >>> from docker.context import ContextAPI
        >>> ctx = ContextAPI.create_context(name='test')
        >>> print(ctx.Metadata)
        {
            "Name": "test",
            "Metadata": {},
            "Endpoints": {
                "docker": {
                    "Host": "unix:///var/run/docker.sock",
                    "SkipTLSVerify": false
                }
            }
        }
        """
        if not name:
            raise errors.MissingContextParameter("name")
        if name == "default":
            raise errors.ContextException(
                '"default" is a reserved context name')
        ctx = Context.load_context(name)
        if ctx:
            raise errors.ContextAlreadyExists(name)
        endpoint = "docker"
        if orchestrator and orchestrator != "swarm":
            endpoint = orchestrator
        ctx = Context(name, orchestrator)
        ctx.set_endpoint(
            endpoint, host, tls_cfg,
            skip_tls_verify=skip_tls_verify,
            def_namespace=default_namespace)
        ctx.save()
        return ctx

    @classmethod
    def get_context(cls, name=None):
        """Retrieves a context object.
        Args:
            name (str): The name of the context

        Example:

        >>> from docker.context import ContextAPI
        >>> ctx = ContextAPI.get_context(name='test')
        >>> print(ctx.Metadata)
        {
            "Name": "test",
            "Metadata": {},
            "Endpoints": {
                "docker": {
                "Host": "unix:///var/run/docker.sock",
                "SkipTLSVerify": false
                }
            }
        }
        """
        if not name:
            name = get_current_context_name()
        if name == "default":
            return cls.DEFAULT_CONTEXT
        return Context.load_context(name)

    @classmethod
    def contexts(cls):
        """Context list.
        Returns:
            (Context): List of context objects.
        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        names = []
        for dirname, dirnames, fnames in os.walk(get_meta_dir()):
            for filename in fnames + dirnames:
                if filename == METAFILE:
                    try:
                        data = json.load(
                            open(os.path.join(dirname, filename)))
                        names.append(data["Name"])
                    except Exception as e:
                        raise errors.ContextException(
                            "Failed to load metafile {}: {}".format(
                                filename, e))

        contexts = [cls.DEFAULT_CONTEXT]
        for name in names:
            contexts.append(Context.load_context(name))
        return contexts

    @classmethod
    def get_current_context(cls):
        """Get current context.
        Returns:
            (Context): current context object.
        """
        return cls.get_context()

    @classmethod
    def set_current_context(cls, name="default"):
        ctx = cls.get_context(name)
        if not ctx:
            raise errors.ContextNotFound(name)

        err = write_context_name_to_docker_config(name)
        if err:
            raise errors.ContextException(
                f'Failed to set current context: {err}')

    @classmethod
    def remove_context(cls, name):
        """Remove a context. Similar to the ``docker context rm`` command.

        Args:
            name (str): The name of the context

        Raises:
            :py:class:`docker.errors.MissingContextParameter`
                If a context name is not provided.
            :py:class:`docker.errors.ContextNotFound`
                If a context with the name does not exist.
            :py:class:`docker.errors.ContextException`
                If name is default.

        Example:

        >>> from docker.context import ContextAPI
        >>> ContextAPI.remove_context(name='test')
        >>>
        """
        if not name:
            raise errors.MissingContextParameter("name")
        if name == "default":
            raise errors.ContextException(
                'context "default" cannot be removed')
        ctx = Context.load_context(name)
        if not ctx:
            raise errors.ContextNotFound(name)
        if name == get_current_context_name():
            write_context_name_to_docker_config(None)
        ctx.remove()

    @classmethod
    def inspect_context(cls, name="default"):
        """Remove a context. Similar to the ``docker context inspect`` command.

        Args:
            name (str): The name of the context

        Raises:
            :py:class:`docker.errors.MissingContextParameter`
                If a context name is not provided.
            :py:class:`docker.errors.ContextNotFound`
                If a context with the name does not exist.

        Example:

        >>> from docker.context import ContextAPI
        >>> ContextAPI.remove_context(name='test')
        >>>
        """
        if not name:
            raise errors.MissingContextParameter("name")
        if name == "default":
            return cls.DEFAULT_CONTEXT()
        ctx = Context.load_context(name)
        if not ctx:
            raise errors.ContextNotFound(name)

        return ctx()
