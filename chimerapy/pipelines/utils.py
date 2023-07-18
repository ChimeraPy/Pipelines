import importlib
from typing import Callable

import requests
from tqdm import tqdm


class UnmetDependencyError(ImportError):
    pass


def requires_packages(*packages: str) -> Callable:
    """Decorator to check if the required packages are installed."""

    def decorator(func_or_class):
        nonlocal packages

        failed = []

        for package_name in packages:
            try:
                importlib.import_module(package_name)
            except ImportError:
                failed.append(package_name)

        if any(failed):
            raise UnmetDependencyError(
                f"The code at {func_or_class.__name__} requires the following "
                f"packages to be installed: {', '.join(packages)}. "
                f"The following packages are missing: {', '.join(failed)}. "
                "Please install the missing packages and try again. "
            )

        return func_or_class

    return decorator


def download_file(
    url: str, fname: str, chunk_size: int = 2048, desc: str = "Downloading File"
) -> str:
    """Download a file from a url."""

    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))

    with open(fname, "wb") as file, tqdm(
        desc=desc,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

    return fname
