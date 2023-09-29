"""
paths.py is a small helper function can be used to set project paths at the start of a python session.
It reads contents from pathsconfig.toml file. pathsconfig.toml should be placed in the same directory as paths.py (root)

Examples
--------
pathsconfig.toml can look like:
```toml
[paths]
src = /path/to/project/root/src
```

```
# assuming ./pathsconfig.toml exists
python paths.py   # it will add /path/to/project/root/src to sys.path
```

Or it can be used as a module:
```
from paths import PATHS

PATHS.src  # evaluates to /path/to/project/root/src
```
"""
import logging
import sys
import os
from types import SimpleNamespace
import yaml
import logging

logger = logging.getLogger("wm_suite.utils")

def add_pathsconfig_to_environ():

    if "PATHSCONFIG" not in os.environ:

        path_to_config_toml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pathconfig.yaml")

        logger.info(f"Adding {path_to_config_toml} to os.environ['PATHSCONFIG'] variable")
        os.environ['PATHSCONFIG'] = path_to_config_toml

#config = configparser.ConfigParser()
#config.read(os.environ['PATHSCONFIG'])
#PATHS = SimpleNamespace(**config["paths"])


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
        path_to_config = os.environ['PATHSCONFIG']

    with open(path_to_config, "r") as f:
        logger.info(f"Found {path_to_config}")
        cfg = yaml.safe_load(f)

    return SimpleNamespace(**cfg)


def main():

    paths = get_paths()

    if paths.src not in sys.path:
        logger.info(f"Adding {paths.src} to pythonpath")
        sys.path.append(paths.src)

    if paths.wm_suite not in sys.path:
        logger.info(f"Adding {paths.wm_suite} to pythonpath")
        sys.path.append(paths.wm_suite)


if __name__ == "__main__":

    main()
