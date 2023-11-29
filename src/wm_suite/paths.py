"""
paths.py is a small module contiaingin get_paths() function
It reads contents from pathconfig.yaml file which contains paths to various directories used by wm_suite

Examples
--------
pathsconfig.toml can look like:
```yaml
src: /path/to/project/root/src
```

```python
# assuming pathconfig.yaml exists in wm_suite directory
python paths.py

```

Or it can be used as a module:
```
from wm_suite.paths import get_paths
p = get_paths()
print(p.src)           # evaluates to /path/to/project/root/src

```
"""
import logging
import sys
import os
from types import SimpleNamespace
import yaml
import logging
from pathlib import Path

logger = logging.getLogger("wm_suite.utils")


DATA_PATH = Path(__file__).parent.parent.parent / "data"
NOUNS_PATH = DATA_PATH / "noun_lists"

def add_pathsconfig_to_environ():
    if "PATHSCONFIG" not in os.environ:
        path_to_config_toml = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "pathconfig.yaml"
        )

        logger.info(
            f"Adding {path_to_config_toml} to os.environ['PATHSCONFIG'] variable"
        )
        os.environ["PATHSCONFIG"] = path_to_config_toml


def get_paths(path_to_config: str = None) -> SimpleNamespace:
    """
    A wrapper to read path values from pathconfig.yaml

    Parameters
    ----------
    path_to_config : str
        path to pathconfig.yaml file

    Returns
    -------
    SimpleNamespace
        a namespace object with paths as attributes
    """

    if path_to_config is None:
        add_pathsconfig_to_environ()
        path_to_config = os.environ["PATHSCONFIG"]

    with open(path_to_config, "r") as f:
        cfg = yaml.safe_load(f)

    return SimpleNamespace(**cfg)


def add_data_to_syspath():
    paths = get_paths()

    if paths.data not in sys.path:
        logger.info(f"Adding {paths.data} to pythonpath")
        sys.path.append(paths.data)


def main():
    paths = get_paths()

    if paths.src not in sys.path:
        logger.info(f"Adding {paths.src} to pythonpath")
        sys.path.append(paths.src)

    add_data_to_syspath()

    if paths.wm_suite not in sys.path:
        logger.info(f"Adding {paths.wm_suite} to pythonpath")
        sys.path.append(paths.wm_suite)


if __name__ == "__main__":
    main()
