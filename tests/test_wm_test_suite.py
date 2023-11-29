from src.wm_suite import wm_test_suite
from wm_suite.paths import get_paths

p = get_paths()

def test_wm_test_suite():

    # test gpt2
    wm_test_suite.main(["--test_run", 
                        "--tokenizer", "gpt2", 
                        "--checkpoint", "gpt2", 
                        "--aggregate_output", 
                        "--aggregate_positions", "[0, 1, 2]"])
    
    # test a different checkpoint
    wm_test_suite.main(["--test_run", 
                        "--tokenizer", "EleutherAI/pythia-14m", 
                        "--checkpoint", "EleutherAI/pythia-14m", 
                        "--aggregate_output", 
                        "--aggregate_positions", "[0, 1, 2]"])
    