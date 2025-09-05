from typing import List, Dict, Optional, Any, Callable, Tuple
from dataclasses import dataclass
import logging
import json
import os
import hashlib
import time
from functools import wraps
from datetime import datetime
from lxml import etree
import sys

# Configure logging
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_ENABLED = True
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.cache', 'xml_diff')
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL = 3600  # 1 hour cache TTL

# Metrics collection
class DiffMetrics:
    def __init__(self):
        self.comparisons = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.comparison_times = []
        self.file_sizes = []
        self.errors = 0

    def record_comparison(self, duration: float, file_size: int):
        self.comparisons += 1
        self.comparison_times.append(duration)
        self.file_sizes.append(file_size)

    def record_cache_hit(self):
        self.cache_hits += 1

    def record_cache_miss(self):
        self.cache_misses += 1

    def record_error(self):
        self.errors += 1

    def get_metrics(self) -> Dict[str, Any]:
        return {
            'total_comparisons': self.comparisons,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_ratio': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'avg_comparison_time': sum(self.comparison_times) / len(self.comparison_times) if self.comparison_times else 0,
            'total_errors': self.errors,
            'avg_file_size': sum(self.file_sizes) / len(self.file_sizes) if self.file_sizes else 0,
            'timestamp': datetime.utcnow().isoformat()
        }

# Global metrics instance
metrics = DiffMetrics()

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


def compare_xml_files(v1_path: str, v2_path: str, 
                    ignore_whitespace: bool = True,
                    normalize_case: bool = False,
                    ignore_comments: bool = True) -> List[Dict]:
    """
    Compare two XML files and return structured diffs with caching.
    
    Args:
        v1_path: Path to first XML file
        v2_path: Path to second XML file
        ignore_whitespace: If True, normalize whitespace in text content
        normalize_case: If True, perform case-insensitive comparison
        ignore_comments: If True, exclude comments from comparison
        
    Returns:
        List of dictionaries containing the differences
    """
    start_time = time.time()
    
    # Generate cache key based on file paths and comparison parameters
    cache_key = _get_cache_key(
        v1_path, v2_path,
        ignore_whitespace=ignore_whitespace,
        normalize_case=normalize_case,
        ignore_comments=ignore_comments
    )
    
    # Try to get cached result
    cached_result = _get_cached_result(cache_key)
    if cached_result is not None:
        metrics.record_cache_hit()
        return cached_result
        
    metrics.record_cache_miss()
    
    try:
        # Read and validate files
        v1_content = read_xml(v1_path)
        v2_content = read_xml(v2_path)
        
        # Generate diff using XMLDiff
        diff = XMLDiff(v1_content, v2_content)
        result = [change.to_dict() for change in diff.diff]
        
        # Record metrics
        duration = time.time() - start_time
        file_size = os.path.getsize(v1_path) + os.path.getsize(v2_path)
        metrics.record_comparison(duration, file_size)
        
        # Cache the result
        _save_to_cache(cache_key, result)
        
        return result
        
    except Exception as e:
        metrics.record_error()
        logger.error(f"Error comparing files {v1_path} and {v2_path}: {e}", 
                    exc_info=True)
        return [{
            'action': 'error',
            'error_type': 'comparison_error',
            'message': f'Error comparing files: {str(e)}',
            'xpath': '/',
            'timestamp': datetime.utcnow().isoformat()
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
                        'new_value': _safe_value(v2_text),
                        'xpath': path
                    })
                elif v2_text is None:
                    changes.append({
                        'action': 'delete',
                        'node': path,
                        'old_value': _safe_value(v1_text),
                        'new_value': None,
                        'xpath': path
                    })
                elif v1_text != v2_text:
                    changes.append({
                        'action': 'update',
                        'node': path,
                        'old_value': _safe_value(v1_text),
                        'new_value': _safe_value(v2_text),
                        'xpath': path
                    })

            return changes

    except Exception as e:
        logger.error(f"Error computing diff: {e}")
        raise

def print_pretty_diff(diffs: List[Dict]) -> None:
    """Pretty print diffs in an indented format."""
    print(json.dumps(diffs, indent=4, ensure_ascii=False))

def _get_cache_key(file1: str, file2: str, **kwargs) -> str:
    """Generate a cache key based on file paths and comparison parameters."""
    key_parts = [
        os.path.abspath(file1),
        os.path.abspath(file2),
        str(sorted(kwargs.items()))
    ]
    return hashlib.md5('|'.join(key_parts).encode('utf-8')).hexdigest()

def _get_cached_result(cache_key: str) -> Optional[Dict]:
    """Retrieve a cached comparison result if it exists and is fresh."""
    if not CACHE_ENABLED:
        return None
        
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        if os.path.exists(cache_file):
            mtime = os.path.getmtime(cache_file)
            if (time.time() - mtime) < CACHE_TTL:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
    except Exception as e:
        logger.warning(f"Cache read error: {e}")
    return None

def _save_to_cache(cache_key: str, result: Dict) -> None:
    """Save comparison result to cache."""
    if not CACHE_ENABLED:
        return
        
    try:
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(result, f)
    except Exception as e:
        logger.warning(f"Cache write error: {e}")

def clear_cache() -> None:
    """Clear all cached comparison results."""
    if os.path.exists(CACHE_DIR):
        for filename in os.listdir(CACHE_DIR):
            if filename.endswith('.json'):
                try:
                    os.remove(os.path.join(CACHE_DIR, filename))
                except Exception as e:
                    logger.warning(f"Error clearing cache file {filename}: {e}")

def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the cache."""
    if not os.path.exists(CACHE_DIR):
        return {
            'cache_enabled': CACHE_ENABLED,
            'cache_dir': CACHE_DIR,
            'cache_ttl_seconds': CACHE_TTL,
            'cached_items': 0,
            'cache_size_bytes': 0
        }
        
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
    total_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in cache_files)
    
    return {
        'cache_enabled': CACHE_ENABLED,
        'cache_dir': CACHE_DIR,
        'cache_ttl_seconds': CACHE_TTL,
        'cached_items': len(cache_files),
        'cache_size_bytes': total_size,
        'cache_size_mb': round(total_size / (1024 * 1024), 2)
    }

def get_metrics() -> Dict[str, Any]:
    """Get current metrics about XML comparisons."""
    cache_stats = get_cache_stats()
    metrics_data = metrics.get_metrics()
    
    return {
        'comparison_metrics': metrics_data,
        'cache_metrics': cache_stats,
        'system': {
            'python_version': '.'.join(map(str, sys.version_info[:3])),
            'platform': sys.platform,
            'timestamp': datetime.utcnow().isoformat()
        }
    }

def _safe_value(value: Any, max_length: int = 100) -> str:
    """
    Safely convert a value to string and truncate if needed.
    
    Args:
        value: The value to convert to string
        max_length: Maximum length of the output string
        
    Returns:
        String representation of the value, truncated if needed
    """
    if value is None:
        return "[None]"
        
    text = str(value).strip()
    if not text:
        return "[Empty]"
        
    # Truncate long text and add ellipsis
    if len(text) > max_length:
        return text[:max_length] + "..."
        
    return text
