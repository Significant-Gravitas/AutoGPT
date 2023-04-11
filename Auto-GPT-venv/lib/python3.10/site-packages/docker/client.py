from .api.client import APIClient
from .constants import (DEFAULT_TIMEOUT_SECONDS, DEFAULT_MAX_POOL_SIZE)
from .models.configs import ConfigCollection
from .models.containers import ContainerCollection
from .models.images import ImageCollection
from .models.networks import NetworkCollection
from .models.nodes import NodeCollection
from .models.plugins import PluginCollection
from .models.secrets import SecretCollection
from .models.services import ServiceCollection
from .models.swarm import Swarm
from .models.volumes import VolumeCollection
from .utils import kwargs_from_env


class DockerClient:
    """
    A client for communicating with a Docker server.

    Example:

        >>> import docker
        >>> client = docker.DockerClient(base_url='unix://var/run/docker.sock')

    Args:
        base_url (str): URL to the Docker server. For example,
            ``unix:///var/run/docker.sock`` or ``tcp://127.0.0.1:1234``.
        version (str): The version of the API to use. Set to ``auto`` to
            automatically detect the server's version. Default: ``1.35``
        timeout (int): Default timeout for API calls, in seconds.
        tls (bool or :py:class:`~docker.tls.TLSConfig`): Enable TLS. Pass
            ``True`` to enable it with default options, or pass a
            :py:class:`~docker.tls.TLSConfig` object to use custom
            configuration.
        user_agent (str): Set a custom user agent for requests to the server.
        credstore_env (dict): Override environment variables when calling the
            credential store process.
        use_ssh_client (bool): If set to `True`, an ssh connection is made
            via shelling out to the ssh client. Ensure the ssh client is
            installed and configured on the host.
        max_pool_size (int): The maximum number of connections
            to save in the pool.
    """
    def __init__(self, *args, **kwargs):
        self.api = APIClient(*args, **kwargs)

    @classmethod
    def from_env(cls, **kwargs):
        """
        Return a client configured from environment variables.

        The environment variables used are the same as those used by the
        Docker command-line client. They are:

        .. envvar:: DOCKER_HOST

            The URL to the Docker host.

        .. envvar:: DOCKER_TLS_VERIFY

            Verify the host against a CA certificate.

        .. envvar:: DOCKER_CERT_PATH

            A path to a directory containing TLS certificates to use when
            connecting to the Docker host.

        Args:
            version (str): The version of the API to use. Set to ``auto`` to
                automatically detect the server's version. Default: ``auto``
            timeout (int): Default timeout for API calls, in seconds.
            max_pool_size (int): The maximum number of connections
                to save in the pool.
            ssl_version (int): A valid `SSL version`_.
            assert_hostname (bool): Verify the hostname of the server.
            environment (dict): The environment to read environment variables
                from. Default: the value of ``os.environ``
            credstore_env (dict): Override environment variables when calling
                the credential store process.
            use_ssh_client (bool): If set to `True`, an ssh connection is
                made via shelling out to the ssh client. Ensure the ssh
                client is installed and configured on the host.

        Example:

            >>> import docker
            >>> client = docker.from_env()

        .. _`SSL version`:
            https://docs.python.org/3.5/library/ssl.html#ssl.PROTOCOL_TLSv1
        """
        timeout = kwargs.pop('timeout', DEFAULT_TIMEOUT_SECONDS)
        max_pool_size = kwargs.pop('max_pool_size', DEFAULT_MAX_POOL_SIZE)
        version = kwargs.pop('version', None)
        use_ssh_client = kwargs.pop('use_ssh_client', False)
        return cls(
            timeout=timeout,
            max_pool_size=max_pool_size,
            version=version,
            use_ssh_client=use_ssh_client,
            **kwargs_from_env(**kwargs)
        )

    # Resources
    @property
    def configs(self):
        """
        An object for managing configs on the server. See the
        :doc:`configs documentation <configs>` for full details.
        """
        return ConfigCollection(client=self)

    @property
    def containers(self):
        """
        An object for managing containers on the server. See the
        :doc:`containers documentation <containers>` for full details.
        """
        return ContainerCollection(client=self)

    @property
    def images(self):
        """
        An object for managing images on the server. See the
        :doc:`images documentation <images>` for full details.
        """
        return ImageCollection(client=self)

    @property
    def networks(self):
        """
        An object for managing networks on the server. See the
        :doc:`networks documentation <networks>` for full details.
        """
        return NetworkCollection(client=self)

    @property
    def nodes(self):
        """
        An object for managing nodes on the server. See the
        :doc:`nodes documentation <nodes>` for full details.
        """
        return NodeCollection(client=self)

    @property
    def plugins(self):
        """
        An object for managing plugins on the server. See the
        :doc:`plugins documentation <plugins>` for full details.
        """
        return PluginCollection(client=self)

    @property
    def secrets(self):
        """
        An object for managing secrets on the server. See the
        :doc:`secrets documentation <secrets>` for full details.
        """
        return SecretCollection(client=self)

    @property
    def services(self):
        """
        An object for managing services on the server. See the
        :doc:`services documentation <services>` for full details.
        """
        return ServiceCollection(client=self)

    @property
    def swarm(self):
        """
        An object for managing a swarm on the server. See the
        :doc:`swarm documentation <swarm>` for full details.
        """
        return Swarm(client=self)

    @property
    def volumes(self):
        """
        An object for managing volumes on the server. See the
        :doc:`volumes documentation <volumes>` for full details.
        """
        return VolumeCollection(client=self)

    # Top-level methods
    def events(self, *args, **kwargs):
        return self.api.events(*args, **kwargs)
    events.__doc__ = APIClient.events.__doc__

    def df(self):
        return self.api.df()
    df.__doc__ = APIClient.df.__doc__

    def info(self, *args, **kwargs):
        return self.api.info(*args, **kwargs)
    info.__doc__ = APIClient.info.__doc__

    def login(self, *args, **kwargs):
        return self.api.login(*args, **kwargs)
    login.__doc__ = APIClient.login.__doc__

    def ping(self, *args, **kwargs):
        return self.api.ping(*args, **kwargs)
    ping.__doc__ = APIClient.ping.__doc__

    def version(self, *args, **kwargs):
        return self.api.version(*args, **kwargs)
    version.__doc__ = APIClient.version.__doc__

    def close(self):
        return self.api.close()
    close.__doc__ = APIClient.close.__doc__

    def __getattr__(self, name):
        s = [f"'DockerClient' object has no attribute '{name}'"]
        # If a user calls a method on APIClient, they
        if hasattr(APIClient, name):
            s.append("In Docker SDK for Python 2.0, this method is now on the "
                     "object APIClient. See the low-level API section of the "
                     "documentation for more details.")
        raise AttributeError(' '.join(s))


from_env = DockerClient.from_env
