import os
import glob
import json
from collections import defaultdict
from typing import List, Dict, Tuple
from utils.xml_diff import compute_diff, read_xml, get_xpath
from app.rag_predictor import XMLRAGPredictor
import asyncio

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

def extract_and_save_diffs(v1_folder: str, v2_folder: str, output_file: str) -> ChangeAnalyzer:
    """Extract diffs between v1 and v2 XML files and analyze changes."""
    analyzer = ChangeAnalyzer()
    v1_files = sorted(glob.glob(os.path.join(v1_folder, '*.xml')))

    with open(output_file, 'w', encoding='utf-8') as out:
        for v1_path in v1_files:
            filename = os.path.basename(v1_path)
            v2_path = os.path.join(v2_folder, filename)

            if not os.path.exists(v2_path):
                print(f"Missing v2 file for {filename}")
                continue

            try:
                v1_content = read_xml(v1_path)
                v2_content = read_xml(v2_path)

                diff = compute_diff(v1_path, v2_path)

                analyzer.analyze_diff(diff, v1_content, v2_content)

                if diff:
                    sample = {
                        "file": filename,
                        "diff": diff
                    }
                    out.write(json.dumps(sample) + "\n")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return analyzer

async def generate_change_prediction(analyzer: ChangeAnalyzer, new_xml: str, predictor = None) -> Dict:
    """Generate change predictions using pattern matching and optionally RAG model."""
    from lxml import etree as ET

    try:
        if isinstance(new_xml, str) and new_xml.strip().startswith('<?xml'):
            new_xml = new_xml.encode('utf-8')
            
        parser = ET.XMLParser(recover=True)
        root = ET.fromstring(new_xml, parser=parser)
        
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

        # First, get pattern-based predictions
        for path, elem_info in elements.items():
            tag = elem_info['tag']
            current_text = elem_info['text']

            if path not in analyzer.changed_paths:
                continue

            common_changes = analyzer.get_changes_by_tag(tag)
            
            # Add pattern-based predictions
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
                                    'occurrences': frequency,
                                    'source': 'pattern'
                                }
                            })
                    except (IndexError, ValueError):
                        continue

            # Add change patterns for reference (only those with frequency >= 20)
            predictions['change_patterns'].extend([
                {'pattern': p, 'frequency': f, 'source': 'pattern'} 
                for p, f in common_changes if f >= 20
            ])

        # Then try to get RAG-based predictions if predictor is provided
        if predictor:
            try:
                # Set a timeout for RAG predictions
                rag_predictions = await asyncio.wait_for(
                    predictor.predict_changes(new_xml),
                    timeout=30  # 30 seconds timeout
                )
                
                if rag_predictions and 'suggested_changes' in rag_predictions:
                    # Add source information to RAG predictions
                    for change in rag_predictions['suggested_changes']:
                        change['suggested_change']['source'] = 'rag'
                    predictions['suggested_changes'].extend(rag_predictions['suggested_changes'])
                
                if rag_predictions and 'potential_improvements' in rag_predictions:
                    predictions['potential_improvements'].extend(rag_predictions['potential_improvements'])
                    
            except asyncio.TimeoutError:
                print("RAG prediction timed out, using pattern-based predictions only")
            except Exception as e:
                print(f"Warning: RAG prediction failed: {str(e)}")

        return predictions

    except ET.ParseError as e:
        return {
            'error': f'Error parsing XML: {str(e)}',
            'suggested_changes': [],
            'potential_improvements': []
        }

async def run_pipeline(file_path=None):
    print("Analyzing changes between v1 and v2 files...")
    analyzer = extract_and_save_diffs("data/oldvone", "data/oldvtwo", "processed/diffs.jsonl")
    
    predictor = XMLRAGPredictor(persist_dir="vectorstore")
    print("\nTraining RAG model...")
    predictor.train_from_diffs("processed/diffs.jsonl")

    print("\n=== Change Analysis ===")
    print("\nMost common change patterns:")
    for pattern, count in analyzer.get_most_common_changes(5):
        print(f"- {pattern} (x{count})")

    print("\nMost frequently changed tags:")
    for tag, count in analyzer.get_most_changed_tags(5):
        print(f"- {tag} (changed {count} times)")

    os.makedirs("data/input", exist_ok=True)
    
    if file_path is None:
        input_files = glob.glob("data/input/*.xml")
        if not input_files:
            print("\nNo XML files found in data/input/ directory.")
            print("Please place new version 1 XML files in data/input/ for analysis.")
            return {"error": "No XML files found in data/input/ directory."}
    else:
        input_files = [file_path]
        
    for test_file in input_files:
        print(f"\nAnalyzing {test_file} for potential improvements...")
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                test_xml = f.read()

            predictions = await generate_change_prediction(analyzer, test_xml, predictor)

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
                    print(f"\n{i}. Potential improvement in <{imp['tag']}> (Confidence: {imp['confidence']}%)")
                    print(f"   XPath: {imp['xpath']}")
                    print(f"   Current: '{imp['current_value']}'")
                    print(f"   Suggestion: {imp['suggestion']}")

            os.makedirs("predictions", exist_ok=True)
            output_file = os.path.join("predictions", f"suggestions_{os.path.basename(test_file)}.json")
            
            output_data = {
                'analyzer_predictions': predictions,
                'file_analyzed': test_file,
                'status': 'completed'
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)

            print(f"\nPredictions saved to: {output_file}")
            return output_data
            
        except Exception as e:
            error_msg = f"\nError processing {test_file}: {e}"
            print(error_msg)
            return {"error": str(e), "file": test_file, "status": "error"}
    
    return {"error": "No files processed", "status": "error"}

if __name__ == "__main__":
    asyncio.run(run_pipeline())
