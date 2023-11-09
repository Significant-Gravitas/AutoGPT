import os
import json
from shutil import copyfile, rmtree
from docker.tls import TLSConfig
from docker.errors import ContextException
from docker.context.config import get_meta_dir
from docker.context.config import get_meta_file
from docker.context.config import get_tls_dir
from docker.context.config import get_context_host


class Context:
    """A context."""

    def __init__(self, name, orchestrator=None, host=None, endpoints=None,
                 tls=False):
        if not name:
            raise Exception("Name not provided")
        self.name = name
        self.context_type = None
        self.orchestrator = orchestrator
        self.endpoints = {}
        self.tls_cfg = {}
        self.meta_path = "IN MEMORY"
        self.tls_path = "IN MEMORY"

        if not endpoints:
            # set default docker endpoint if no endpoint is set
            default_endpoint = "docker" if (
                not orchestrator or orchestrator == "swarm"
                ) else orchestrator

            self.endpoints = {
                default_endpoint: {
                    "Host": get_context_host(host, tls),
                    "SkipTLSVerify": not tls
                }
            }
            return

        # check docker endpoints
        for k, v in endpoints.items():
            if not isinstance(v, dict):
                # unknown format
                raise ContextException("""Unknown endpoint format for
                    context {}: {}""".format(name, v))

            self.endpoints[k] = v
            if k != "docker":
                continue

            self.endpoints[k]["Host"] = v.get("Host", get_context_host(
                host, tls))
            self.endpoints[k]["SkipTLSVerify"] = bool(v.get(
                "SkipTLSVerify", not tls))

    def set_endpoint(
            self, name="docker", host=None, tls_cfg=None,
            skip_tls_verify=False, def_namespace=None):
        self.endpoints[name] = {
            "Host": get_context_host(host, not skip_tls_verify),
            "SkipTLSVerify": skip_tls_verify
        }
        if def_namespace:
            self.endpoints[name]["DefaultNamespace"] = def_namespace

        if tls_cfg:
            self.tls_cfg[name] = tls_cfg

    def inspect(self):
        return self.__call__()

    @classmethod
    def load_context(cls, name):
        meta = Context._load_meta(name)
        if meta:
            instance = cls(
                meta["Name"],
                orchestrator=meta["Metadata"].get("StackOrchestrator", None),
                endpoints=meta.get("Endpoints", None))
            instance.context_type = meta["Metadata"].get("Type", None)
            instance._load_certs()
            instance.meta_path = get_meta_dir(name)
            return instance
        return None

    @classmethod
    def _load_meta(cls, name):
        meta_file = get_meta_file(name)
        if not os.path.isfile(meta_file):
            return None

        metadata = {}
        try:
            with open(meta_file) as f:
                metadata = json.load(f)
        except (OSError, KeyError, ValueError) as e:
            # unknown format
            raise Exception("""Detected corrupted meta file for
                context {} : {}""".format(name, e))

        # for docker endpoints, set defaults for
        # Host and SkipTLSVerify fields
        for k, v in metadata["Endpoints"].items():
            if k != "docker":
                continue
            metadata["Endpoints"][k]["Host"] = v.get(
                "Host", get_context_host(None, False))
            metadata["Endpoints"][k]["SkipTLSVerify"] = bool(
                v.get("SkipTLSVerify", True))

        return metadata

    def _load_certs(self):
        certs = {}
        tls_dir = get_tls_dir(self.name)
        for endpoint in self.endpoints.keys():
            if not os.path.isdir(os.path.join(tls_dir, endpoint)):
                continue
            ca_cert = None
            cert = None
            key = None
            for filename in os.listdir(os.path.join(tls_dir, endpoint)):
                if filename.startswith("ca"):
                    ca_cert = os.path.join(tls_dir, endpoint, filename)
                elif filename.startswith("cert"):
                    cert = os.path.join(tls_dir, endpoint, filename)
                elif filename.startswith("key"):
                    key = os.path.join(tls_dir, endpoint, filename)
            if all([ca_cert, cert, key]):
                verify = None
                if endpoint == "docker" and not self.endpoints["docker"].get(
                        "SkipTLSVerify", False):
                    verify = True
                certs[endpoint] = TLSConfig(
                    client_cert=(cert, key), ca_cert=ca_cert, verify=verify)
        self.tls_cfg = certs
        self.tls_path = tls_dir

    def save(self):
        meta_dir = get_meta_dir(self.name)
        if not os.path.isdir(meta_dir):
            os.makedirs(meta_dir)
        with open(get_meta_file(self.name), "w") as f:
            f.write(json.dumps(self.Metadata))

        tls_dir = get_tls_dir(self.name)
        for endpoint, tls in self.tls_cfg.items():
            if not os.path.isdir(os.path.join(tls_dir, endpoint)):
                os.makedirs(os.path.join(tls_dir, endpoint))

            ca_file = tls.ca_cert
            if ca_file:
                copyfile(ca_file, os.path.join(
                    tls_dir, endpoint, os.path.basename(ca_file)))

            if tls.cert:
                cert_file, key_file = tls.cert
                copyfile(cert_file, os.path.join(
                    tls_dir, endpoint, os.path.basename(cert_file)))
                copyfile(key_file, os.path.join(
                    tls_dir, endpoint, os.path.basename(key_file)))

        self.meta_path = get_meta_dir(self.name)
        self.tls_path = get_tls_dir(self.name)

    def remove(self):
        if os.path.isdir(self.meta_path):
            rmtree(self.meta_path)
        if os.path.isdir(self.tls_path):
            rmtree(self.tls_path)

    def __repr__(self):
        return f"<{self.__class__.__name__}: '{self.name}'>"

    def __str__(self):
        return json.dumps(self.__call__(), indent=2)

    def __call__(self):
        result = self.Metadata
        result.update(self.TLSMaterial)
        result.update(self.Storage)
        return result

    def is_docker_host(self):
        return self.context_type is None

    @property
    def Name(self):
        return self.name

    @property
    def Host(self):
        if not self.orchestrator or self.orchestrator == "swarm":
            endpoint = self.endpoints.get("docker", None)
            if endpoint:
                return endpoint.get("Host", None)
            return None

        return self.endpoints[self.orchestrator].get("Host", None)

    @property
    def Orchestrator(self):
        return self.orchestrator

    @property
    def Metadata(self):
        meta = {}
        if self.orchestrator:
            meta = {"StackOrchestrator": self.orchestrator}
        return {
            "Name": self.name,
            "Metadata": meta,
            "Endpoints": self.endpoints
        }

    @property
    def TLSConfig(self):
        key = self.orchestrator
        if not key or key == "swarm":
            key = "docker"
        if key in self.tls_cfg.keys():
            return self.tls_cfg[key]
        return None

    @property
    def TLSMaterial(self):
        certs = {}
        for endpoint, tls in self.tls_cfg.items():
            cert, key = tls.cert
            certs[endpoint] = list(
                map(os.path.basename, [tls.ca_cert, cert, key]))
        return {
            "TLSMaterial": certs
        }

    @property
    def Storage(self):
        return {
            "Storage": {
                "MetadataPath": self.meta_path,
                "TLSPath": self.tls_path
            }}
