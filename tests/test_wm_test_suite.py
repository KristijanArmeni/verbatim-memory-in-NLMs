from src.wm_suite import wm_test_suite
from wm_suite.paths import get_paths

p = get_paths()

def test_wm_test_suite():

    wm_test_suite.main(["--test_run", 
                        "--device", "cuda",
                        "--output_filename", "test_run.csv", 
                        "--output_dir", "./data"])