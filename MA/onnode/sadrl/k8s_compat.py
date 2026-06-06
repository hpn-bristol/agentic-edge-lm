"""Workaround for python kubernetes lib's broken load_incluster_config()
on this RKE cluster.

Symptom: pods using `kubernetes.config.load_incluster_config()` get 401
Unauthorized from the apiserver, even though the same projected SA token
file at /var/run/secrets/kubernetes.io/serviceaccount/token works when
sent via raw HTTPS requests (returns 403 = valid token, just no perms).

Workaround: build a Configuration object manually with explicit BearerToken
auth key (the lib looks up under identifier "BearerToken", not "authorization"),
and resolve well-known cluster services (e.g. mongodb-service) via raw
requests so they don't depend on the lib working at all.

Use:
    from sadrl.k8s_compat import get_api_client, get_mongo_ip
    api_client = get_api_client()
    v1 = client.CoreV1Api(api_client=api_client)
    apps = client.AppsV1Api(api_client=api_client)
    mongo_ip = get_mongo_ip()
"""
import os

import requests
from kubernetes import client
from kubernetes.client.configuration import Configuration

_K8S_TOKEN_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/token"
_K8S_CA_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"


def _read_token() -> str:
    return open(_K8S_TOKEN_FILE, "r").read().strip()


def _k8s_base() -> str:
    host = os.environ["KUBERNETES_SERVICE_HOST"]
    port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
    return f"https://{host}:{port}"


def get_api_client() -> client.ApiClient:
    """Returns an ApiClient with explicit Bearer token configuration.

    Replaces config.load_incluster_config() which sends a malformed token
    in this cluster.
    """
    cfg = Configuration()
    cfg.host = _k8s_base()
    cfg.verify_ssl = True
    cfg.ssl_ca_cert = _K8S_CA_FILE
    cfg.api_key["BearerToken"] = _read_token()
    cfg.api_key_prefix["BearerToken"] = "Bearer"
    return client.ApiClient(configuration=cfg)


def get_mongo_ip() -> str:
    """Resolves mongodb-service.mongodb cluster IP via raw HTTPS.

    Avoids going through the kubernetes lib's auth code path.
    """
    token = _read_token()
    r = requests.get(
        f"{_k8s_base()}/api/v1/namespaces/mongodb/services/mongodb-service",
        headers={"Authorization": f"Bearer {token}"},
        verify=_K8S_CA_FILE,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["spec"]["clusterIP"]


def patch_load_incluster_config() -> None:
    """Monkey-patch kubernetes.config.load_incluster_config in place.

    After calling this once at module entry, every subsequent
    `config.load_incluster_config()` and `client.CoreV1Api()` call
    in any module (including 3rd-party code) will pick up our working
    Configuration without code changes.
    """
    from kubernetes import config as _orig_config

    def _patched(client_configuration=None):
        cfg = get_api_client().configuration
        if client_configuration is None:
            Configuration.set_default(cfg)
        else:
            client_configuration.host = cfg.host
            client_configuration.verify_ssl = cfg.verify_ssl
            client_configuration.ssl_ca_cert = cfg.ssl_ca_cert
            client_configuration.api_key = dict(cfg.api_key)
            client_configuration.api_key_prefix = dict(cfg.api_key_prefix)

    _orig_config.load_incluster_config = _patched
    # Also set the default Configuration so client.CoreV1Api() without
    # explicit api_client uses our working setup.
    Configuration.set_default(get_api_client().configuration)
