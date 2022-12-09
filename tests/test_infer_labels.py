
from src.wm_suite.preprocess_and_merge_csvfiles import infer_labels_from_filebasename

def test_infer_labels():

    filename = "surprisal_gpt2_a-10_sce1_repeat_random.csv"

    arc, model_id, scenario, second_list, list_type = infer_labels_from_filebasename(filename)

    assert arc == "gpt2"
    assert model_id == "a-10"
    assert scenario == "sce1"
    assert second_list == "repeat"
    assert list_type == "random"