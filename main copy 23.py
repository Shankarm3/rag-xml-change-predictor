#!/usr/bin/env python3
"""
RAG XML Change Predictor - Main entry point
"""
import os
import sys
import asyncio
import argparse
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import local modules
from utils.xml_diff import compare_xml_files
import glob
import json
from collections import defaultdict
from typing import List, Dict, Tuple
from utils.xml_diff import clear_cache, compute_diff, get_metrics, read_xml, get_xpath
from app.rag_predictor import XMLRAGPredictor
import logging
import re

def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by:
    1. Converting to lowercase
    2. Replacing multiple whitespace characters with a single space
    3. Stripping leading/trailing whitespace
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text string
    """
    if not text or not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Replace multiple whitespace with single space and strip
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Set up logging with better formatting
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Set the root logger level to DEBUG to capture all logs
    logging.getLogger().setLevel(logging.DEBUG)

setup_logging()
logger = logging.getLogger(__name__)

class ChangeAnalyzer:
    def __init__(self):
        self.change_patterns = defaultdict(lambda: defaultdict(int))
        self.tag_changes = defaultdict(int)
        self.change_types = defaultdict(int)
        self.changed_paths = set()
        self.false_positives = 0
        self.valid_changes = 0

    def _is_semantic_change(self, old_val: str, new_val: str) -> bool:
        """Check if there's a meaningful change between values."""
        if not old_val or not new_val:
            return True
        return normalize_text(old_val) != normalize_text(new_val)

    def analyze_diff(self, diff: List[Dict], v1_content: str, v2_content: str):
        """Analyze the diff and update change patterns with detailed information."""
        if not diff:
            return
            
        logger.debug(f"Analyzing {len(diff)} potential changes...")
        
        for change in diff:
            action = change.get('action')
            node = change.get('node', '')
            old_val = change.get('old_value', '')
            new_val = change.get('new_value', '')
            xpath = change.get('xpath', node)
            
            if not node or not xpath:
                logger.debug(f"Skipping change with missing node/xpath: {change}")
                continue
                
            tag = node.split('/')[-1].split('[')[0] if node else 'unknown'
            
            # Skip if values are semantically equivalent
            if action == 'update' and not self._is_semantic_change(old_val, new_val):
                self.false_positives += 1
                logger.debug(f"Skipping semantically equivalent change: {tag} {old_val} -> {new_val}")
                continue
                
            self.valid_changes += 1
            self.changed_paths.add(node)
            
            if tag and tag != 'unknown':
                self.tag_changes[tag] += 1

            self.change_types[action] += 1

            # Track change patterns
            if action == 'update' and old_val and new_val:
                pattern = f"{tag}: {old_val} -> {new_val}"
                self.change_patterns[tag][pattern] += 1
            elif action == 'insert' and new_val:
                pattern = f"Add {tag}: {new_val}"
                self.change_patterns[tag][pattern] += 1
            elif action == 'delete' and old_val:
                pattern = f"Remove {tag}: {old_val}"
                self.change_patterns[tag][pattern] += 1
                
        logger.info(f"Processed {self.valid_changes} valid changes, filtered {self.false_positives} false positives")

    def get_most_common_changes(self, top_n: int = 5) -> List[Tuple[str, int]]:
        """Get the most common change patterns across all tags."""
        all_changes = []
        for tag, patterns in self.change_patterns.items():
            for pattern, count in patterns.items():
                all_changes.append((pattern, count))

        return sorted(all_changes, key=lambda x: x[1], reverse=True)[:top_n]

    def get_changes_by_tag(self, tag: str, top_n: int = 3) -> List[Tuple[str, int]]:
        """Get the most common changes for a specific tag."""
        if tag in self.change_patterns:
            return sorted(
                self.change_patterns[tag].items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
        return []

    def get_most_changed_tags(self, top_n: int = 3, min_count: int = 2) -> List[Tuple[str, int]]:
        """Get the most frequently changed tags with a minimum change threshold."""
        filtered = [(tag, count) for tag, count in self.tag_changes.items() if count >= min_count]
        return sorted(filtered, key=lambda x: x[1], reverse=True)[:top_n]

def extract_and_save_diffs(v1_folder: str, v2_folder: str, output_dir: str, journal: str, progress_step: int = 500, max_lines_per_file: int = 5000):
    """
    Extract diffs between v1 and v2 XML files with enhanced validation.
    """
    analyzer = ChangeAnalyzer()
    v1_files = sorted(glob.glob(os.path.join(v1_folder, '*.xml')))
    total = len(v1_files)
    file_count = 1
    lines_written = 0
    
    journal_output_dir = os.path.join(output_dir, journal)
    os.makedirs(journal_output_dir, exist_ok=True)
    
    base_output = os.path.join(journal_output_dir, 'diffs.jsonl')
    current_output = f"{os.path.splitext(base_output)[0]}_{file_count}.jsonl"
    
    out = open(current_output, 'w', encoding='utf-8')
    processed_files = 0
    skipped_files = 0
    
    try:
        for idx, v1_path in enumerate(v1_files, 1):
            filename = os.path.basename(v1_path)
            v2_path = os.path.join(v2_folder, filename)
            
            if not os.path.exists(v2_path):
                logger.warning(f"Skipping {filename}: v2 file not found at {v2_path}")
                skipped_files += 1
                continue
                
            try:
                logger.debug(f"Processing {filename}...")
                v1_content = read_xml(v1_path)
                v2_content = read_xml(v2_path)

                if not v1_content or not v2_content:
                    logger.warning(f"Skipping {filename}: Empty content in v1 or v2 file")
                    skipped_files += 1
                    continue


                # Please do not remove this comment block
                # Perform the diff with enhanced validation
                # diff = compare_xml_files(v1_path, v2_path)

                diff = compare_xml_files(v1_path, v2_path)
                
                metrics_data = get_metrics()
                log_metrics(metrics_data)
                
                if diff:
                    analyzer.analyze_diff(diff, v1_content, v2_content)
                    
                    valid_diffs = [d for d in diff if d.get('xpath')]
                    if valid_diffs:
                        diff_entry = {
                            'filename': filename,
                            'journal': journal,
                            'diff': valid_diffs
                        }
                        out.write(json.dumps(diff_entry) + '\n')
                        lines_written += 1
                        
                        if lines_written >= max_lines_per_file:
                            out.close()
                            file_count += 1
                            current_output = f"{os.path.splitext(base_output)[0]}_{file_count}.jsonl"
                            out = open(current_output, 'w', encoding='utf-8')
                            lines_written = 0

                processed_files += 1
                
                if idx % progress_step == 0 or idx == total:
                    logger.info(f"Processed {idx}/{total} files ({idx/total*100:.1f}%)")
                    logger.info(f"  - Valid changes: {analyzer.valid_changes}")
                    logger.info(f"  - False positives filtered: {analyzer.false_positives}")
                    logger.info(f"  - Output file: {current_output}")
            
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)
                skipped_files += 1
                continue
                
    except Exception as e:
        logger.error(f"Fatal error in processing: {str(e)}", exc_info=True)
        raise
        
    finally:
        out.close()
        
    logger.info(f"Processing complete. Processed: {processed_files}, Skipped: {skipped_files}")
    logger.info(f"Total valid changes: {analyzer.valid_changes}")
    logger.info(f"Total false positives filtered: {analyzer.false_positives}")
    
    return analyzer

def log_metrics(metrics_data: Dict):
    """Log metrics in a structured and readable format."""
    if not metrics_data:
        return
    
    logger.info("\n=== XML Comparison Metrics ===")
    
    # Comparison metrics
    comp_metrics = metrics_data.get('comparison_metrics', {})
    if comp_metrics:
        logger.info("\nComparison Statistics:")
        logger.info(f"  - Total comparisons: {comp_metrics.get('total_comparisons', 0)}")
        logger.info(f"  - Cache hits: {comp_metrics.get('cache_hits', 0)}")
        logger.info(f"  - Cache misses: {comp_metrics.get('cache_misses', 0)}")
        logger.info(f"  - Cache hit ratio: {comp_metrics.get('cache_hit_ratio', 0):.1%}")
        logger.info(f"  - Average comparison time: {comp_metrics.get('avg_comparison_time', 0):.3f}s")
        logger.info(f"  - Total errors: {comp_metrics.get('total_errors', 0)}")
        logger.info(f"  - Average file size: {comp_metrics.get('avg_file_size', 0) / 1024:.2f} KB")
    
    # Cache metrics
    cache_metrics = metrics_data.get('cache_metrics', {})
    if cache_metrics:
        logger.info("\nCache Statistics:")
        logger.info(f"  - Cache enabled: {cache_metrics.get('cache_enabled', False)}")
        logger.info(f"  - Cache directory: {cache_metrics.get('cache_dir', 'N/A')}")
        logger.info(f"  - Cached items: {cache_metrics.get('cached_items', 0)}")
        logger.info(f"  - Cache size: {cache_metrics.get('cache_size_mb', 0):.2f} MB")
        logger.info(f"  - Cache TTL: {cache_metrics.get('cache_ttl_seconds', 0) / 3600:.1f} hours")
    
    # System info
    sys_info = metrics_data.get('system', {})
    if sys_info:
        logger.info("\nSystem Information:")
        logger.info(f"  - Python version: {sys_info.get('python_version', 'N/A')}")
        logger.info(f"  - Platform: {sys_info.get('platform', 'N/A')}")
        logger.info(f"  - Timestamp: {sys_info.get('timestamp', 'N/A')}")
    
    logger.info("\n" + "="*50 + "\n")

def load_analyzer_from_diffs(journal: str) -> ChangeAnalyzer:
    """
    Load and aggregate processed diffs for analysis from a specific journal.
    
    Args:
        journal: Journal name to load diffs for (required)
        
    Returns:
        ChangeAnalyzer: An instance of ChangeAnalyzer with loaded diff data
        
    Raises:
        ValueError: If journal is not provided or is empty
    """
    if not journal:
        raise ValueError("Journal name is required")
        
    analyzer = ChangeAnalyzer()
    processed_dir = "processed"
    
    if not os.path.exists(processed_dir):
        logger.warning(f"No processed directory found at {processed_dir}")
        return analyzer
    
    journal_dir = os.path.join(processed_dir, journal)
    if not os.path.exists(journal_dir):
        logger.warning(f"No directory found for journal: {journal}")
        return analyzer
        
    journal_dirs = [journal_dir]
    
    total_files = 0
    lines = 0
    
    for journal_dir in journal_dirs:
        if not os.path.isdir(journal_dir):
            logger.warning(f"Skipping non-directory: {journal_dir}")
            continue
            
        # Find all diff files for this journal
        diff_files = sorted(glob.glob(os.path.join(journal_dir, 'diffs_*.jsonl'))) or \
                    sorted(glob.glob(os.path.join(journal_dir, 'diffs.jsonl*')))
        
        for diff_file in diff_files:
            try:
                with open(diff_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            analyzer.analyze_diff(
                                entry['diff'],
                                "",  # Original content not needed for analysis
                                ""   # New content not needed for analysis
                            )
                            lines += 1
                        except json.JSONDecodeError as e:
                            logger.warning(f"Error parsing JSON in {diff_file}: {e}")
                            continue
                total_files += 1
            except Exception as e:
                logger.error(f"Error reading {diff_file}: {e}")
                continue
    
    logger.info(f"Loaded {lines} diffs from {total_files} files for journal: {journal}")
    return analyzer

def generate_change_prediction(analyzer: ChangeAnalyzer, new_xml: str) -> Dict:
    """Generate change predictions based on learned patterns."""
    from lxml import etree as ET

    try:
        if isinstance(new_xml, str) and new_xml.strip().startswith('<?xml'):
            new_xml = new_xml.encode('utf-8')
                
        parser = ET.XMLParser(recover=True)
        root = ET.fromstring(new_xml, parser=parser)
        if root is None:
            return {
                'error': 'Invalid or empty XML file. No root element found.',
                'suggested_changes': [],
                'potential_improvements': []
            }
            
        elements = {}
        for elem in root.iter():
            if elem.text and elem.text.strip():
                path = get_xpath(elem)
                tag = elem.tag
                elements[path] = {
                    'tag': tag,
                    'text': elem.text.strip(),
                    'element': elem
                }

        predictions = {
            'suggested_changes': [],
            'potential_improvements': [],
            'change_patterns': []
        }

        for path, elem_info in elements.items():
            tag = elem_info['tag']
            current_text = elem_info['text']

            # Check if any parent path is in changed_paths
            path_parts = path.split('/')
            is_changed = any(
                '/'.join(path_parts[:i+1]) in analyzer.changed_paths 
                for i in range(1, len(path_parts))
            )
            
            if not is_changed:
                continue

            # Get common changes for this tag
            common_changes = analyzer.get_changes_by_tag(tag)

            for pattern, frequency in common_changes:
                if '->' in pattern:
                    try:
                        old_part = pattern.split('->')[0].split(':', 1)[1].strip()
                        if current_text == old_part and frequency >= 1:
                            new_text = pattern.split('->', 1)[1].strip()
                            confidence = min(95, 10 + frequency * 30)
                            predictions['suggested_changes'].append({
                                'xpath': path,
                                'tag': tag,
                                'current_value': current_text,
                                'suggested_change': {
                                    'from': old_part,
                                    'to': new_text,
                                    'confidence': confidence,
                                    'pattern': pattern,
                                }
                            })
                    except Exception as e:
                        logger.warning(f"Error processing pattern '{pattern}': {e}")
                        continue

        # Add potential improvements based on frequently changed tags
        for tag, count in analyzer.get_most_changed_tags():
            # Only consider tags that exist in the current document
            matching_elements = [(p, e) for p, e in elements.items() if e['tag'] == tag]
            
            if not matching_elements:
                continue
                
            # Get the first matching element for this tag
            path, elem_info = matching_elements[0]
            
            # Get sample changes for this tag
            sample_changes = analyzer.get_changes_by_tag(tag, top_n=3)
            if not sample_changes:
                continue
                
            # Format examples
            examples = "; ".join([
                f"{p.split('->')[0].strip()} -> {p.split('->')[1].strip() if '->' in p else '...'}" 
                for p, _ in sample_changes
            ])
            
            # Calculate confidence based on frequency
            confidence = min(80, 10 + count * 10)
            
            predictions['potential_improvements'].append({
                'tag': tag,
                'xpath': path,
                'current_value': elem_info['text'],
                'suggestion': (
                    f"This <{tag}> element was frequently modified in previous versions. "
                    f"Examples: {examples}" if examples else
                    f"This <{tag}> element was frequently modified in previous versions."
                ),
                'confidence': confidence
            })

        return predictions

    except ET.XMLSyntaxError as e:
        logger.error(f"XML syntax error: {e}")
        return {
            'error': f'Invalid XML: {str(e)}',
            'suggested_changes': [],
            'potential_improvements': []
        }
    except Exception as e:
        logger.error(f"Error generating predictions: {e}", exc_info=True)
        return {
            'error': f'Unexpected error: {str(e)}',
            'suggested_changes': [],
            'potential_improvements': []
        }

async def run_pipeline(analyzer: ChangeAnalyzer = None, file_path: str = None, journal: str = "mnras"):
    # Clear cache at the start of processing
    clear_cache()
    logger.info("Cleared XML diff cache at the start of processing")
    
    result = {
        'status': 'error',
        'message': 'No files processed',
        'suggested_changes': [],
        'potential_improvements': []
    }
    try:
        v1_folder = os.path.join("data", journal, "v1")
        v2_folder = os.path.join("data", journal, "v2")

        # Only extract diffs if analyzer is not provided
        if analyzer is None:
            print(f"Analyzing changes between {v1_folder} and {v2_folder} files...")
            analyzer = extract_and_save_diffs(v1_folder, v2_folder, "processed", journal, progress_step=500)
        
        predictor = XMLRAGPredictor(persist_dir="vectorstore")
        try:
            print("\nTraining RAG model (this may take a few minutes)...")
            # Please do not remove this comment block
            # await asyncio.wait_for(
            #     predictor.train_from_diffs(
            #         "processed/diffs.jsonl",
            #         chunk_size=500,
            #         chunk_overlap=50
            #     ),
            #     timeout=3 
            # )
            print("\nRAG model training completed")
        except asyncio.TimeoutError:
            print("\nTraining took too long, continuing with partial results")
        except Exception as e:
            print(f"\nError during training: {str(e)}")
        print("\n=== Change Analysis ===")
        print("\nMost common change patterns:")
        for pattern, count in analyzer.get_most_common_changes(5):
            print(f"- {pattern} (x{count})")

        print("\nMost frequently changed tags:")
        for tag, count in analyzer.get_most_changed_tags(5):
            print(f"- {tag} (changed {count} times)")

        input_files = []
        if file_path is None:
            input_dir = Path(os.path.join("data", journal, "input"))
            if input_dir.exists() and any(input_dir.iterdir()):
                input_files = list(input_dir.glob("*.xml"))
            else:
                print(f"\nNo XML files found in {input_dir} directory.")
                print(f"Please place new version 1 XML files in {input_dir}/ for analysis.")
                return result
        else:
            input_files = [Path(file_path)]
        for test_file in input_files:
            print(f"\nAnalyzing {test_file} for potential improvements...")
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    test_xml = f.read()
                # Please do not remove this code block
                # Get RAG predictions (commented out for now)
                # print("\n=== RAG Predictions ===")
                # rag_predictions = await predictor.predict_changes(test_xml)

                predictions = generate_change_prediction(analyzer, test_xml)

                if predictions.get("suggested_changes"):
                    print("\n=== Suggested Changes ===")
                    for i, change in enumerate(predictions["suggested_changes"], 1):
                        print(f"\n{i}. {change['suggested_change']['from']} -> {change['suggested_change']['to']} (Confidence: {change['suggested_change']['confidence']}%)")
                        print(f"   XPath: {change['xpath']}")
                        print(f"   Current: {change['current_value']}")
                        print(f"   Suggest: {change['suggested_change']['to']}")

                if predictions.get("potential_improvements"):
                    print("\n=== Potential Improvements ===")
                    for i, imp in enumerate(predictions["potential_improvements"], 1):
                        print(f"\n{i}. {imp['suggestion']} (Confidence: {imp['confidence']}%)")
                        print(f"   XPath: {imp['xpath']}")
                        print(f"   Current: {imp['current_value']}")
                        print(f"   Suggestion: {imp['suggestion']}")

                output_dir = Path("predictions")
                output_dir.mkdir(exist_ok=True)
                output_file = output_dir / f"suggestions_{test_file.stem}.json"
                output_data = {
                    'suggested_changes': predictions.get("suggested_changes", []),
                    'potential_improvements': predictions.get("potential_improvements", []),
                    'change_patterns': [
                        { 'pattern': pattern, 'frequency': freq }
                        for pattern, freq in analyzer.get_most_common_changes(10)
                    ],
                    'file_analyzed': str(test_file),
                    'status': 'completed'
                }
                if 'rag_predictions' in locals():
                    output_data['rag_predictions'] = rag_predictions

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2)

                print(f"\nPredictions saved to: {output_file}")
                result = output_data
            except Exception as e:
                print(f"Error processing {test_file}: {str(e)}")
                result['message'] = f"Error processing {test_file}: {str(e)}"
        result['status'] = 'success'
        result['message'] = 'Processing completed successfully'
        
        cache_stats = get_metrics()
        logger.info(f"Cache statistics: {json.dumps(cache_stats, indent=2)}")
        
        clear_cache()
        logger.info("Cleared XML diff cache after processing")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in run_pipeline: {str(e)}", exc_info=True)
        result['message'] = f'Error: {str(e)}'
        return result
    finally:
        clear_cache()
        logger.info("Ensured XML diff cache is cleared")
        if 'predictor' in locals():
            del predictor
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

def run_tests():
    """Run test suite for the application."""
    print("Running test suite...")
    try:
        # Import here to avoid circular imports
        from tests.test_xml_diff import run_xml_diff_tests
        return run_xml_diff_tests()
    except ImportError as e:
        print(f"Error importing test module: {e}")
        print("Make sure you're running from the project root directory.")
        return 1

def main():
    """Main entry point for the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG XML Change Predictor')
    parser.add_argument('--test', action='store_true', help='Run test suite')
    parser.add_argument('--file', type=str, help='Process a single XML file')
    parser.add_argument('--dir', type=str, help='Process all XML files in a directory')
    
    args = parser.parse_args()
    
    if args.test:
        return run_tests()
    elif args.file:
        return run_pipeline(file_path=args.file)
    elif args.dir:
        return run_pipeline(journal=args.dir)
    else:
        parser.print_help()
        return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
