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
import configparser
import logging
import sys
import os
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO)

if "PATHSCONFIG" not in os.environ:

    path_to_config_toml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pathsconfig.toml")

    logging.info(f"Adding {path_to_config_toml} to os.environ['PATHSCONFIG'] variable")
    os.environ['PATHSCONFIG'] = path_to_config_toml

config = configparser.ConfigParser()
config.read(os.environ['PATHSCONFIG'])
PATHS = SimpleNamespace(**config["paths"])


if __name__ == "__main__":

    logging.info(f"Adding {PATHS.src} to pythonpath")
    sys.path.append(PATHS.src)
