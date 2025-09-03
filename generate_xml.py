import os
import shutil
import random
from glob import glob

def duplicate_xmls_with_random(src_folder, dest_folder, total_files=10):
    os.makedirs(dest_folder, exist_ok=True)
    xml_files = sorted(glob(os.path.join(src_folder, "*.xml")))
    if not xml_files:
        print(f"No XML files found in {src_folder}")
        return

    used_names = set()
    count = len(xml_files)
    for i in range(total_files):
        src_file = xml_files[i % count]
        base = os.path.splitext(os.path.basename(src_file))[0]
        # Generate a unique random 4-digit number for the new file
        while True:
            rand_digits = f"{random.randint(1000, 9999)}"
            new_name = f"{base}_{rand_digits}.xml"
            if new_name not in used_names:
                used_names.add(new_name)
                break
        dest_path = os.path.join(dest_folder, new_name)
        shutil.copy2(src_file, dest_path)
        print(f"Copied {src_file} -> {dest_path}")

if __name__ == "__main__":
    duplicate_xmls_with_random("data/v1", "data/v1", total_files=10)
    duplicate_xmls_with_random("data/v2", "data/v2", total_files=10)