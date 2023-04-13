import importlib
from typing import Callable


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
