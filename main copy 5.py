import os
import glob
import json
from collections import defaultdict
from typing import List, Dict, Tuple
from utils.xml_diff import compute_diff, read_xml, XMLDiff, get_xpath
from app.rag_predictor import XMLRAGPredictor

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

                if diff:  # only if there's a difference
                    sample = {
                        "file": filename,
                        "diff": diff
                    }
                    out.write(json.dumps(sample) + "\n")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return analyzer

def generate_change_prediction(analyzer: ChangeAnalyzer, new_xml: str) -> Dict:
    """Generate change predictions based on learned patterns."""
    from lxml import etree as ET

    try:
        root = ET.fromstring(new_xml)

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
                                    'occurrences': frequency
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
                            'change_count': count,
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


def run_pipeline():
    print("Analyzing changes between v1 and v2 files...")
    # analyzer = extract_and_save_diffs("data/v1", "data/v2", "processed/diffs.jsonl")
    analyzer = extract_and_save_diffs("data/oldvone", "data/oldvtwo", "processed/diffs.jsonl")
    
    # predictor = XMLRAGPredictor(persist_dir="vectorstore")
    # print("\nTraining RAG model...")
    # predictor.train_from_diffs("processed/diffs.jsonl")

    print("\n=== Change Analysis ===")
    print("\nMost common change patterns:")
    for pattern, count in analyzer.get_most_common_changes(5):
        print(f"- {pattern} (x{count})")

    print("\nMost frequently changed tags:")
    for tag, count in analyzer.get_most_changed_tags(5):
        print(f"- {tag} (changed {count} times)")

    # test_file = "data/v1/sample_test.xml"
    test_file = "data/oldvone/sample-staf001.xml"
    if os.path.exists(test_file):
        print(f"\nAnalyzing {test_file} for potential improvements...")
        with open(test_file, 'r', encoding='utf-8') as f:
            test_xml = f.read()

        # print("\n=== RAG Predictions ===")
        # rag_predictions = predictor.predict_changes(test_xml)
        predictions = generate_change_prediction(analyzer, test_xml)

        if predictions.get("suggested_changes"):
            print("\n=== Suggested Changes ===")
            for i, change in enumerate(predictions["suggested_changes"], 1):
                print(f"\n{i}. {change['suggested_change']['from']} -> {change['suggested_change']['to']} suggestion (Confidence: {change['suggested_change']['confidence']}%)")
                print(f"   XPath: {change['xpath']}")
                print(f"   Current: {change['current_value']}")
                print(f"   Suggest: {change['suggested_change']['to']}")

        else:
            print("\nNo specific change suggestions based on patterns.")

        if predictions.get("potential_improvements"):
            print("\n=== Potential Improvements ===")
            for i, imp in enumerate(predictions["potential_improvements"], 1):
                print(f"\n{i}. Potential improvement in <{imp['tag']}> (Confidence: {imp['confidence']}%)")
                print(f"   XPath: {imp['xpath']}")
                print(f"   Current: '{imp['current_value']}'")
                print(f"   Suggestion: {imp['suggestion']}")

        os.makedirs("predictions", exist_ok=True)
        output_file = os.path.join("predictions", f"suggestions_{os.path.basename(test_file)}.json")
        
        # Prepare output data
        output_data = {
            'analyzer_predictions': predictions
        }
        
        # Only include RAG predictions if they were generated
        if 'rag_predictions' in locals():
            output_data['rag_predictions'] = rag_predictions
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nPredictions saved to: {output_file}")
    else:
        print(f"\nTest file not found: {test_file}")
        print("Please place your test XML file in the data/oldvone/ directory")

if __name__ == "__main__":
    run_pipeline()
