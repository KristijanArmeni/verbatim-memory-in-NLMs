import os
from src.wm_suite import wm_test_suite
from src.wm_suite.utils import get_paths
from paths import PATHS as p

def test_wm_test_suite():

    wm_test_suite.main(["--test_run", 
                        "--device", "cuda",
                        "--output_filename", "test_run.csv", 
                        "--output_dir", os.path.join(p.root, "src", "tests", "data")])