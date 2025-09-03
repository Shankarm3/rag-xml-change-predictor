import os
import glob
import json
from collections import defaultdict
from typing import List, Dict, Tuple
from utils.xml_diff import compute_diff, read_xml, get_xpath
from app.rag_predictor import XMLRAGPredictor
import asyncio
from pathlib import Path
import logging
import sys
import argparse

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

    def analyze_diff(self, diff: List[Dict], v1_content: str, v2_content: str):
        """Analyze the diff and update change patterns with detailed information."""
        for change in diff:
            action = change.get('action')
            node = change.get('node', '')
            old_val = change.get('old_value', '')
            new_val = change.get('new_value', '')
            if node:
                self.changed_paths.add(node)
            tag = node.split('/')[-1].split('[')[0] if node else 'unknown'

            if tag and tag != 'unknown':
                self.tag_changes[tag] += 1

            self.change_types[action] += 1

            if action == 'update' and old_val and new_val:
                pattern = f"{tag}: {old_val} -> {new_val}"
                self.change_patterns[tag][pattern] += 1
            elif action == 'insert' and new_val:
                pattern = f"Add {tag}: {new_val}"
                self.change_patterns[tag][pattern] += 1
            elif action == 'delete' and old_val:
                pattern = f"Remove {tag}: {old_val}"
                self.change_patterns[tag][pattern] += 1

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
    Extract diffs between v1 and v2 XML files and analyze changes.
    Processes files in batches, writes progress, and splits output if needed.
    Skips files that cannot be processed and continues with the next file.
    
    Args:
        v1_folder: Path to folder containing v1 XML files
        v2_folder: Path to folder containing v2 XML files
        output_dir: Base output directory for diff files
        journal: Journal name for organizing diff files
        progress_step: Print progress every N files
        max_lines_per_file: Maximum number of lines per output file before splitting
    """
    analyzer = ChangeAnalyzer()
    v1_files = sorted(glob.glob(os.path.join(v1_folder, '*.xml')))[:1000]
    total = len(v1_files)
    file_count = 1
    lines_written = 0
    
    # Create journal-specific output directory
    journal_output_dir = os.path.join(output_dir, journal)
    os.makedirs(journal_output_dir, exist_ok=True)
    
    # Base output file path with journal name
    base_output = os.path.join(journal_output_dir, 'diffs.jsonl')
    current_output = f"{os.path.splitext(base_output)[0]}_{file_count}.jsonl"
    
    out = open(current_output, 'w', encoding='utf-8')
    processed_files = 0
    skipped_files = 0
    
    try:
        for idx, v1_path in enumerate(v1_files, 1):
            filename = os.path.basename(v1_path)
            v2_path = os.path.join(v2_folder, filename)
            
            # Skip if v2 file doesn't exist
            if not os.path.exists(v2_path):
                logger.warning(f"Skipping {filename}: v2 file not found at {v2_path}")
                skipped_files += 1
                continue
                
            try:
                # Process the file
                v1_content = read_xml(v1_path)
                v2_content = read_xml(v2_path)

                if not v1_content or not v2_content:
                    logger.warning(f"Skipping {filename}: Empty content in v1 or v2 file")
                    skipped_files += 1
                    continue

                # Please do not remove this comment
                # Try deep diff (if model/API available), else fallback to classic
                # from utils.xml_diff import compute_diff_deep
                # diff = compute_diff_deep(v1_path, v2_path)
                
                diff = compute_diff(v1_path, v2_path)
                
                if diff:
                    analyzer.analyze_diff(diff, v1_content, v2_content)
                    diff_entry = {
                        'filename': filename,
                        'journal': journal,
                        'diff': [d if isinstance(d, dict) else d.to_dict() for d in diff]
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
                    logger.info(f"Processed {idx}/{total} files ({idx/total*100:.1f}%), "
                              f"Skipped: {skipped_files}, Output file: {current_output}")
            
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
    return analyzer

def load_analyzer_from_diffs(journal: str = None) -> ChangeAnalyzer:
    """
    Load and aggregate processed diffs for analysis.
    If journal is specified, only load diffs for that journal.
    
    Args:
        journal: Optional journal name to filter diffs
    """
    analyzer = ChangeAnalyzer()
    processed_dir = "processed"
    
    if not os.path.exists(processed_dir):
        logger.warning(f"No processed directory found at {processed_dir}")
        return analyzer
    
    # If journal is specified, only process that journal's directory
    if journal:
        journal_dirs = [os.path.join(processed_dir, journal)]
    else:
        # Otherwise process all journals
        journal_dirs = [
            os.path.join(processed_dir, d) 
            for d in os.listdir(processed_dir)
            if os.path.isdir(os.path.join(processed_dir, d))
        ]
    
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
                            # Only process entries for the specified journal if filtering
                            if journal is None or entry.get('journal') == journal:
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
    
    logger.info(f"Loaded {lines} diffs from {total_files} files for journal: {journal or 'all'}")
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

            if path not in analyzer.changed_paths:
                continue

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
                                    # 'occurrence': frequency
                                }
                            })
                    except (IndexError, ValueError):
                        continue

        for pattern, frequency in analyzer.get_most_common_changes():
            predictions['change_patterns'].append({
                'pattern': pattern,
                'frequency': frequency
            })

        for tag, count in analyzer.get_most_changed_tags():
            if tag in [e['tag'] for e in elements.values()]:
                for path, elem_info in elements.items():
                    if elem_info['tag'] == tag:
                        confidence = min(80, count * 15)
                        sample_changes = analyzer.get_changes_by_tag(tag, top_n=3)
                        examples = "; ".join([p for p, _ in sample_changes])
                        suggestion = (
                            f"This <{tag}> element was frequently modified in previous versions. "
                            f"Examples: {examples}" if examples else
                            f"This <{tag}> element was frequently modified in previous versions."
                        )
                        predictions['potential_improvements'].append({
                            'tag': tag,
                            'xpath': path,
                            'current_value': elem_info['text'],
                            'suggestion': suggestion,
                            'confidence': confidence
                        })
                        break

        return predictions

    except ET.ParseError as e:
        return {
            'error': f'Error parsing XML: {str(e)}',
            'suggested_changes': [],
            'potential_improvements': []
        }


async def run_pipeline(analyzer: ChangeAnalyzer = None, file_path: str = None, journal: str = "mnras"):
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
        return result
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        result['message'] = str(e)
        return result
    finally:
        if 'predictor' in locals():
            del predictor
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

# Usage example (do NOT replace main workflow):
# analyzer = load_analyzer_from_diffs(journal="mnras") use this whenever we need to skip the prediction generation
# ... use 'analyzer' for pattern predictions ...

def load_analyzer_from_diffs(journal: str = None) -> ChangeAnalyzer:
    """
    Load and aggregate processed diffs for analysis.
    If journal is specified, only load diffs for that journal.
    
    Args:
        journal: Optional journal name to filter diffs
    """
    analyzer = ChangeAnalyzer()
    processed_dir = "processed"
    
    if not os.path.exists(processed_dir):
        logger.warning(f"No processed directory found at {processed_dir}")
        return analyzer
    
    # If journal is specified, only process that journal's directory
    if journal:
        journal_dirs = [os.path.join(processed_dir, journal)]
    else:
        # Otherwise process all journals
        journal_dirs = [
            os.path.join(processed_dir, d) 
            for d in os.listdir(processed_dir)
            if os.path.isdir(os.path.join(processed_dir, d))
        ]
    
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
                            # Only process entries for the specified journal if filtering
                            if journal is None or entry.get('journal') == journal:
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
    
    logger.info(f"Loaded {lines} diffs from {total_files} files for journal: {journal or 'all'}")
    return analyzer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XML Change Predictor with RAG - journal support")
    parser.add_argument("--journal", type=str, required=True, help="Journal name for subdirectory under data/<journal>/v1,v2")
    parser.add_argument("--file", type=str, default=None, help="Specific XML file to process (optional)")
    args = parser.parse_args()

    # Immediate exit if --journal is empty or just whitespace
    if not args.journal or not args.journal.strip():
        print("Error: --journal argument must be a non-empty value. Exiting.")
        sys.exit(1)

    # Immediate exit if data/<journal> directory does not exist
    journal_dir = os.path.join("data", args.journal)
    if not os.path.isdir(journal_dir):
        print(f"Error: Journal data directory '{journal_dir}' does not exist. Exiting.")
        sys.exit(1)
        
    # Set up paths with journal name
    v1_dir = os.path.join(journal_dir, "v1")
    v2_dir = os.path.join(journal_dir, "v2")
    output_dir = "processed"
    
    # Option 1: Generate new diffs (default)
    # Comment this block to use existing diffs instead
    print(f"Analyzing changes between {v1_dir} and {v2_dir} files...")
    analyzer = extract_and_save_diffs(
        v1_folder=v1_dir,
        v2_folder=v2_dir,
        output_dir=output_dir,
        journal=args.journal
    )

    # Option 2: Use existing diffs
    # Uncomment below to use pre-generated diffs instead of processing files
    # analyzer = load_analyzer_from_diffs(journal=args.journal)
    try:
        # Pass the analyzer to run_pipeline to avoid duplicate processing
        asyncio.run(run_pipeline(analyzer=analyzer, file_path=args.file, journal=args.journal))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.stop()
                loop.close()
        except:
            pass
        print("\nCleanup complete. Exiting...")
