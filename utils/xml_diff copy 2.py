from xmldiff import main
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from xmldiff import main, formatting
import logging

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
                # Convert the diff object to our XMLChange object
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
            # Count by tag
            if change.node:
                tag = change.node.split('/')[-1].split('[')[0]
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Count by change type
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

def get_xpath(element) -> str:
    """
    Generate XPath from an XML element using standard xml.etree.ElementTree.
    
    Args:
        element: XML element to generate XPath for
        
    Returns:
        str: XPath string for the element
    """
    def _get_path(element, path=None):
        if path is None:
            path = []
        
        # Get parent element if it exists
        parent = element.find('..')
        if parent is not None:
            # Find all siblings with the same tag
            siblings = [e for e in parent if e.tag == element.tag]
            if len(siblings) > 1:
                # If there are multiple siblings with same tag, add index
                index = siblings.index(element) + 1
                path.insert(0, f"{element.tag}[{index}]")
            else:
                path.insert(0, element.tag)
            return _get_path(parent, path)
        else:
            # Reached the root element
            path.insert(0, element.tag)
            return path
    
    path = _get_path(element)
    return '/' + '/'.join(path)

def compute_diff(xml_path_1: str, xml_path_2: str) -> List[Dict]:
    """Compute differences between two XML files with detailed text changes."""
    try:
        with open(xml_path_1, 'r', encoding='utf-8') as f1, \
             open(xml_path_2, 'r', encoding='utf-8') as f2:
            
            v1_content = f1.read()
            v2_content = f2.read()
            
            # Parse XML to get structured data
            from xml.etree import ElementTree as ET
            
            def get_text_elements(xml_str):
                """Extract all elements with text content."""
                try:
                    root = ET.fromstring(xml_str)
                    elements = {}
                    for elem in root.iter():
                        if elem.text and elem.text.strip():
                            path = get_xpath(elem)
                            elements[path] = elem.text.strip()
                    return elements
                except ET.ParseError:
                    return {}
            
            # Get text elements from both versions
            v1_elements = get_text_elements(v1_content)
            v2_elements = get_text_elements(v2_content)
            
            # Find changes
            changes = []
            all_paths = set(v1_elements.keys()) | set(v2_elements.keys())
            
            for path in all_paths:
                v1_text = v1_elements.get(path)
                v2_text = v2_elements.get(path)
                
                if v1_text is None:
                    # New element in v2
                    changes.append({
                        'action': 'insert',
                        'node': path,
                        'old_value': None,
                        'new_value': v2_text,
                        'xpath': path
                    })
                elif v2_text is None:
                    # Deleted in v2
                    changes.append({
                        'action': 'delete',
                        'node': path,
                        'old_value': v1_text,
                        'new_value': None,
                        'xpath': path
                    })
                elif v1_text != v2_text:
                    # Updated text
                    changes.append({
                        'action': 'update',
                        'node': path,
                        'old_value': v1_text,
                        'new_value': v2_text,
                        'xpath': path
                    })
            
            return changes
            
    except Exception as e:
        logger.error(f"Error computing diff between {xml_path_1} and {xml_path_2}: {e}")
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