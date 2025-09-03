from xmldiff import main
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from xmldiff import main, formatting
import logging
from lxml import etree
import ollama
import json

logger = logging.getLogger(__name__)

@dataclass
class XMLChange:
    action: str
    node: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    xpath: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'action': self.action,
            'node': self.node,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'xpath': self.xpath
        }

class XMLDiff:
    """Enhanced XML diff utility with better change tracking and analysis."""
    def __init__(self, v1_content: str, v2_content: str):
        self.v1_content = v1_content
        self.v2_content = v2_content
        self.diff = []
        self._parse_diffs()
    
    def _parse_diffs(self) -> None:
        """Parse the XML diffs using xmldiff library."""
        try:
            formatter = formatting.DiffFormatter()
            diff_objects = main.diff_texts(
                self.v1_content, 
                self.v2_content,
                formatter=formatter
            )
            
            for change in diff_objects:
                xml_change = XMLChange(
                    action=getattr(change, 'action', 'unknown'),
                    node=getattr(change, 'node', ''),
                    old_value=getattr(change, 'old_value', None),
                    new_value=getattr(change, 'new_value', None),
                    xpath=getattr(change, 'xpath', None)
                )
                self.diff.append(xml_change)
                
        except Exception as e:
            logger.error(f"Error parsing XML diffs: {e}")
            raise
    
    def get_changes_by_tag(self, tag_name: str) -> List[XMLChange]:
        """Get all changes for a specific XML tag."""
        return [change for change in self.diff 
                if tag_name.lower() in change.node.lower()]
    
    def get_changes_by_type(self, change_type: str) -> List[XMLChange]:
        """Get all changes of a specific type (insert, delete, update, etc.)."""
        return [change for change in self.diff 
                if change.action.lower() == change_type.lower()]
    
    def get_most_changed_tag(self) -> Tuple[Optional[str], int]:
        """Get the tag with the most changes."""
        tag_counts = {}
        for change in self.diff:
            if change.node:
                tag = change.node.split('/')[-1].split('[')[0]
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        if not tag_counts:
            return None, 0
            
        return max(tag_counts.items(), key=lambda x: x[1])
    
    def get_summary(self) -> Dict:
        """Get a summary of changes."""
        tag_counts = {}
        change_type_counts = {}
        
        for change in self.diff:
            if change.node:
                tag = change.node.split('/')[-1].split('[')[0]
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            change_type = change.action
            change_type_counts[change_type] = change_type_counts.get(change_type, 0) + 1
        
        most_changed_tag, max_changes = self.get_most_changed_tag()
        
        return {
            'total_changes': len(self.diff),
            'tag_changes': tag_counts,
            'change_types': change_type_counts,
            'most_changed_tag': most_changed_tag,
            'most_changes_count': max_changes
        }


def get_xpath(element: etree.Element) -> str:
    """Generate full XPath for an element including indexes for siblings."""
    path = []
    while element is not None:
        parent = element.getparent()
        if parent is not None:
            same_tag_siblings = [sib for sib in parent if sib.tag == element.tag]
            if len(same_tag_siblings) > 1:
                index = same_tag_siblings.index(element) + 1
                path.insert(0, f"{element.tag}[{index}]")
            else:
                path.insert(0, f"{element.tag}")
        else:
            path.insert(0, f"{element.tag}")
        element = parent
    return '/' + '/'.join(path)

def get_text_elements(xml_str: str) -> Dict[str, str]:
    """Extract all elements with text content and return XPath -> text."""
    try:
        root = etree.fromstring(xml_str)
        elements = {}
        for elem in root.iter():
            text = (elem.text or '').strip()
            if text:
                path = get_xpath(elem)
                elements[path] = text
        return elements
    except etree.XMLSyntaxError as e:
        logger.error(f"XML parsing error: {e}")
        return {}

def compute_diff_deep(xml_path_1: str, xml_path_2: str, model: str = "mistral") -> list:
    import ollama
    import json
    try:
        with open(xml_path_1, 'r', encoding='utf-8') as f1, open(xml_path_2, 'r', encoding='utf-8') as f2:
            # v1_content = f1.read() 
            # v2_content = f2.read()

            v1_content = f1.read()[:2000] 
            v2_content = f2.read()[:2000]

        prompt = f"""You are an expert in XML. Compare these two short XML files and return a JSON list of changes: ...
XML v1:
{v1_content}

XML v2:
{v2_content}
"""
        print("Prompt ready, starting Ollama call...")
        response = ollama.generate(model=model, prompt=prompt)
        response_text = response['response'].strip()
        print("Ollama call finished!")
        print("Ollama response:", repr(response_text))

        if response_text.startswith("```") and "json" in response_text[:10].lower():
            response_text = response_text.split("```")[-2].strip()

        if not response_text.strip():
            logger.warning("Ollama returned empty output.")
            return compute_diff(xml_path_1, xml_path_2)

        try:
            changes = json.loads(response_text)
            if not isinstance(changes, list):
                raise ValueError("Ollama output is not a list")
            return changes
        except Exception as e:
            logger.warning(f"Ollama output could not be parsed as JSON:{response_text!r}")
            return compute_diff(xml_path_1, xml_path_2)

    except Exception as exc:
        import traceback
        logger.warning(
            "Ollama deep XML diff failed (fallback to Python core diff): %s\n%s",
            exc, traceback.format_exc()
        )
        return compute_diff(xml_path_1, xml_path_2)

def compute_diff(xml_path_1: str, xml_path_2: str) -> List[Dict]:
    try:
        with open(xml_path_1, 'r', encoding='utf-8') as f1, \
             open(xml_path_2, 'r', encoding='utf-8') as f2:
            
            v1_content = f1.read()
            v2_content = f2.read()

            v1_elements = get_text_elements(v1_content)
            v2_elements = get_text_elements(v2_content)

            changes = []
            all_paths = set(v1_elements.keys()).union(v2_elements.keys())

            for path in all_paths:
                v1_text = v1_elements.get(path)
                v2_text = v2_elements.get(path)

                if v1_text is None:
                    changes.append({
                        'action': 'insert',
                        'node': path,
                        'old_value': None,
                        'new_value': v2_text,
                        'xpath': path
                    })
                elif v2_text is None:
                    changes.append({
                        'action': 'delete',
                        'node': path,
                        'old_value': v1_text,
                        'new_value': None,
                        'xpath': path
                    })
                elif v1_text != v2_text:
                    changes.append({
                        'action': 'update',
                        'node': path,
                        'old_value': v1_text,
                        'new_value': v2_text,
                        'xpath': path
                    })

            return changes

    except Exception as e:
        logger.error(f"Error computing diff: {e}")
        raise

def read_xml(path: str) -> str:
    """Read XML file and return its content as a string."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading XML file {path}: {e}")
        raise

def compare_xml_files(v1_path: str, v2_path: str) -> XMLDiff:
    """Compare two XML files and return an XMLDiff object."""
    v1_content = read_xml(v1_path)
    v2_content = read_xml(v2_path)
    return XMLDiff(v1_content, v2_content)