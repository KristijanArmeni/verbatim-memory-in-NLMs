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
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO)

config = configparser.ConfigParser()
config.read("./config.toml")

PATHS = SimpleNamespace(**config["paths"])

if __name__ == "__main__":
    logging.info(f"Adding {PATHS.src} to pythonpath")
    sys.path.append(PATHS.src)
