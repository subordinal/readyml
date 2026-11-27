"""Read model metadata from the Hugging Face Hub without downloading weights."""
import json
import os
import urllib.error
import urllib.request

API = "https://huggingface.co/api/models"
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".pth")


class HubError(Exception):
    pass


def fetch_model(model_id, revision=None, token=None):
    url = f"{API}/{model_id}"
    if revision:
        url += f"/revision/{revision}"
    req = urllib.request.Request(url + "?blobs=true", headers={"User-Agent": "readyml"})
    token = token or os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        hint = " (gated or private: set HF_TOKEN)" if e.code in (401, 403) else ""
        raise HubError(f"Hub returned {e.code} for {model_id}{hint}") from e
    except urllib.error.URLError as e:
        raise HubError(f"cannot reach the Hub: {e.reason}") from e


def weight_files(info):
    """Weight files to load. Safetensors wins when a repo ships several formats."""
    files = [(s["rfilename"], s.get("size") or 0) for s in info.get("siblings", [])]
    for suffix in WEIGHT_SUFFIXES:
        hit = [f for f in files if f[0].endswith(suffix)]
        if hit:
            return hit
    return []
