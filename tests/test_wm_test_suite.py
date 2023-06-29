import os
from src.wm_suite import wm_test_suite

def test_wm_test_suite():


    wm_test_suite.main(["--test_run", 
                        "--device", "cuda",
                        "--output_filename", "test_run.csv", 
                        "--output_dir", os.path.join(os.environ['PROJ_ROOT'], "src", "tests", "data")])