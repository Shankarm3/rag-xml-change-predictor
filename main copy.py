import os
import glob
import json
from utils.xml_diff import compute_diff, read_xml
from app.rag_predictor import XMLRAGPredictor

def extract_and_save_diffs(v1_folder, v2_folder, output_file):
    v1_files = sorted(glob.glob(os.path.join(v1_folder, '*.xml')))
    v2_files = sorted(glob.glob(os.path.join(v2_folder, '*.xml')))
    with open(output_file, 'w', encoding='utf-8') as out:
        for v1, v2 in zip(v1_files, v2_files):
            try:
                diff = compute_diff(v1, v2)
                sample = {
                    "v1": read_xml(v1),
                    "v2": read_xml(v2),
                    "diff": diff
                }
                out.write(json.dumps(sample) + "\n")
            except Exception as e:
                print(f"Error diffing {v1}: {e}")

def run_pipeline():
    extract_and_save_diffs("data/v1", "data/v2", "processed/diffs.jsonl")

    rag = XMLRAGPredictor()
    rag.train_from_diffs("processed/diffs.jsonl")

    new_v1 = read_xml("data/v1/sample_test.xml")
    prediction = rag.predict_changes(new_v1)
    print("Predicted Changes:\n", prediction)

if __name__ == "__main__":
    run_pipeline()