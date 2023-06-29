
from src.wm_suite.preprocess_and_merge_csvfiles import infer_labels_from_filebasename

def test_infer_labels():

    filename = "wms_gpt2_a-10_sce1_1_n3_random_repeat.csv"

    labels = infer_labels_from_filebasename(filename)

    arc, model_id, scenario, list_len, prompt_len, list_type, condition = labels

    assert len(labels) == 7
    assert arc == "gpt2"
    assert model_id == "a-10"
    assert scenario == "sce1"
    assert list_len == "n3"
    assert prompt_len == "1"
    assert condition == "repeat"
    assert list_type == "random"