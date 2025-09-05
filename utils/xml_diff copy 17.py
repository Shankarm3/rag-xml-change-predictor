from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
import json
from lxml import etree

logger = logging.getLogger(__name__)

def pretty_print_xml(xml_str: str) -> str:
    """Return pretty-printed XML string to avoid giant single-line diffs."""
    try:
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.fromstring(xml_str.encode(), parser=parser)
        return etree.tostring(tree, pretty_print=True, encoding='unicode')
    except Exception:
        return xml_str  # fallback if parse fails


def _safe_value(value, max_len: int = 120) -> str:
    """Convert value safely to string and truncate long values."""
    try:
        s = str(value)
    except Exception:
        s = repr(value)
    return s if len(s) <= max_len else s[:max_len] + "..."


def normalize_text(text: str) -> str:
    """Normalize text for comparison by removing extra whitespace and converting to lowercase."""
    if not text or not isinstance(text, str):
        return ''
    return ' '.join(text.split()).lower()


def get_xpath(element: etree.Element) -> str:
    """Generate stable XPath for an element including indexes for siblings."""
    def get_sibling_index(elem):
        """Get 1-based index of element among its siblings with same tag."""
        if elem.getparent() is None:
            return 1
        siblings = [sib for sib in elem.getparent().iterchildren(elem.tag)]
        return siblings.index(elem) + 1 if len(siblings) > 1 else None
    
    path = []
    current = element
    
    while current is not None:
        if not etree.iselement(current):
            break
            
        # Safely get tag as string
        try:
            tag = str(current.tag)
        except Exception:
            tag = "unknown"
            
        idx = get_sibling_index(current)
        
        # Add ID attribute to path if available for better stability
        elem_id = None
        try:
            elem_id = current.get('id') or current.get('name')
        except Exception:
            pass
            
        if elem_id:
            path.append(f"{tag}[@id='{elem_id}']")
        elif idx is not None:
            path.append(f"{tag}[{idx}]")
        else:
            path.append(tag)
            
        current = current.getparent()
    
    # Ensure all path components are strings and filter out any problematic ones
    safe_path = []
    for part in reversed(path):
        try:
            if part is not None:
                safe_path.append(str(part))
        except Exception:
            continue
    
    return '/' + '/'.join(safe_path) if safe_path else ''

def get_text_elements(xml_str: str) -> Dict[str, str]:
    """Extract all elements with text content and return XPath -> text."""
    try:
        root = etree.fromstring(xml_str.encode())
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
    """Enhanced XML diff utility with better error handling and validation."""
    
    def __init__(self, v1_content: str, v2_content: str):
        self.v1_content = self._preprocess_xml(v1_content or '')
        self.v2_content = self._preprocess_xml(v2_content or '')
        self.diff: List[XMLChange] = []
        self._parse_diffs()
        self._validate_diffs()
    
    def _preprocess_xml(self, content: str) -> str:
        """Preprocess XML content to handle common issues."""
        try:
            # Remove BOM if present
            if content.startswith('\ufeff'):
                content = content[1:]
            
            # Parse and re-serialize to ensure well-formed XML
            parser = etree.XMLParser(remove_blank_text=True, recover=True)
            root = etree.fromstring(content.encode('utf-8'), parser=parser)
            return etree.tostring(root, encoding='unicode', pretty_print=True)
        except Exception as e:
            logger.debug(f"XML preprocessing failed: {e}")
            return content  # Return original if preprocessing fails

    def _parse_diffs(self) -> None:
        """Try xmldiff first; fallback to element-based diff."""
        try:
            try:
                from xmldiff import main, formatting
                formatter = formatting.DiffFormatter()
                diff_objects = main.diff_texts(
                    self.v1_content,
                    self.v2_content,
                    formatter=formatter,
                    diff_options={'F': 0.5, 'uniqueattrs': ['id', 'name']}
                )
                self._process_diff_objects(diff_objects)
                return
            except Exception as e:
                logger.debug(f"xmldiff failed, using element-based diff: {e}")
                self._fallback_element_diff()
        except Exception as e:
            logger.error(f"Error in XML diff: {e}")
            self.diff = [XMLChange(
                action='replace',
                node='document',
                old_value=_safe_value(self.v1_content),
                new_value=_safe_value(self.v2_content),
                xpath='/')
            ]
    
    def _serialize_diff_object(self, obj):
        """Safely convert xmldiff object fields to strings, handling functions and objects."""
        if obj is None:
            return None
        try:
            if callable(obj):
                return f"<callable:{obj.__name__}>"
            return _safe_value(obj)
        except Exception:
            return repr(obj)

    def _process_diff_objects(self, diff_objects) -> None:
        """Convert xmldiff results into XMLChange objects with validation."""
        def safe_get_attr(obj, attr, default=''):
            try:
                value = getattr(obj, attr, default)
                return str(value) if value is not None else default
            except Exception:
                return str(default)

        for change in diff_objects:
            try:
                if isinstance(change, tuple):
                    action = str(change[0]) if len(change) > 0 else 'unknown'
                    node = str(change[1]) if len(change) > 1 else ''
                    old_value = str(change[2]) if len(change) > 2 else ''
                    new_value = str(change[3]) if len(change) > 3 else ''
                    xpath = node  # For tuple format, node contains the XPath
                else:
                    action = safe_get_attr(change, 'action', 'unknown')
                    node = safe_get_attr(change, 'node', '')
                    old_value = safe_get_attr(change, 'old_value', '')
                    new_value = safe_get_attr(change, 'new_value', '')
                    xpath = safe_get_attr(change, 'xpath', node)
                
                # Skip if values are effectively the same after normalization
                if action == 'update' and normalize_text(old_value) == normalize_text(new_value):
                    logger.debug(f"Skipping semantically equal values: {old_value} -> {new_value}")
                    continue
                
                xml_change = XMLChange(
                    action=action,
                    node=node,
                    old_value=old_value,
                    new_value=new_value,
                    xpath=xpath
                )
                self.diff.append(xml_change)
                
            except Exception as e:
                logger.warning(f"Error processing diff object: {e}", exc_info=True)
                continue
    
    def _fallback_element_diff(self) -> None:
        """Fallback to element-level diff when xmldiff fails."""
        def safe_str(value):
            """Safely convert any value to string, handling Cython functions."""
            if value is None:
                return ''
            try:
                if callable(value):
                    return f"<{value.__class__.__name__}>"
                return str(value)
            except Exception:
                return str(type(value))
                
        try:
            v1_elements = get_text_elements(self.v1_content)
            v2_elements = get_text_elements(self.v2_content)
            all_paths = set(v1_elements.keys()).union(v2_elements.keys())

            for path in all_paths:
                v1_text = v1_elements.get(path)
                v2_text = v2_elements.get(path)

                if v1_text is None:
                    self.diff.append(XMLChange(
                        action='insert',
                        node=safe_str(path),
                        new_value=safe_str(v2_text),
                        xpath=safe_str(path)
                    ))
                elif v2_text is None:
                    self.diff.append(XMLChange(
                        action='delete',
                        node=safe_str(path),
                        old_value=safe_str(v1_text),
                        xpath=safe_str(path)
                    ))
                elif v1_text != v2_text:
                    self.diff.append(XMLChange(
                        action='update',
                        node=safe_str(path),
                        old_value=safe_str(v1_text),
                        new_value=safe_str(v2_text),
                        xpath=safe_str(path)
                    ))
        except Exception as e:
            logger.error(f"Error in element-based diff: {e}", exc_info=True)
            self.diff = [XMLChange(
                action='error',
                node='document',
                old_value=safe_str(self.v1_content[:1000]),
                new_value=safe_str(self.v2_content[:1000]),
                xpath='/',
                error=str(e)
            )]

    def _validate_diffs(self):
        """Validate generated diffs against actual file contents."""
        valid_diffs = []
        
        try:
            v1_root = etree.fromstring(self.v1_content.encode())
            v2_root = etree.fromstring(self.v2_content.encode())
            
            for change in self.diff:
                xpath = change.xpath
                if not xpath:
                    continue
                    
                try:
                    # Check if elements exist in both documents
                    v1_elems = v1_root.xpath(xpath)
                    v2_elems = v2_root.xpath(xpath)
                    
                    if not v1_elems or not v2_elems:
                        logger.debug(f"Skipping diff - elements not found in both docs: {xpath}")
                        continue
                        
                    # For updates, verify the values are actually different
                    if change.action == 'update':
                        v1_text = normalize_text(''.join(v1_elems[0].itertext()))
                        v2_text = normalize_text(''.join(v2_elems[0].itertext()))
                        
                        if v1_text == v2_text:
                            logger.debug(f"Skipping false positive diff - values match: {xpath}")
                            continue
                            
                    valid_diffs.append(change)
                    
                except Exception as e:
                    logger.warning(f"Error validating diff for {xpath}: {e}")
                    valid_diffs.append(change)  # Keep the diff if validation fails
                    
            self.diff = valid_diffs
            
        except Exception as e:
            logger.error(f"Error during diff validation: {e}")
            # Keep original diffs if validation fails


def read_xml(path: str) -> str:
    """Read and validate XML file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            try:
                etree.fromstring(content.encode())
            except etree.XMLSyntaxError as e:
                logger.warning(f"XML syntax error in {path}: {e}")
            return content
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {path}") from e
    except Exception as e:
        raise Exception(f"Error reading XML file {path}: {str(e)}") from e


def compare_xml_files(v1_path: str, v2_path: str) -> List[Dict]:
    """Compare two XML files and return structured diffs."""
    try:
        v1_content = read_xml(v1_path)
        v2_content = read_xml(v2_path)
        diff = XMLDiff(v1_content, v2_content)
        return [change.to_dict() for change in diff.diff]
    except Exception as e:
        logger.error(f"Error comparing files {v1_path} and {v2_path}: {e}")
        return [{
            'action': 'error',
            'node': 'document',
            'error': str(e),
            'xpath': '/'
        }]

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

def print_pretty_diff(diffs: List[Dict]) -> None:
    """Pretty print diffs in an indented format."""
    print(json.dumps(diffs, indent=4, ensure_ascii=False))
