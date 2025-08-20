from xmldiff import main
from lxml import etree

def compute_diff(xml_path_1, xml_path_2):
    return main.diff_files(xml_path_1, xml_path_2)

def read_xml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()