import os
import json
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
from utils.xml_diff import compare_xml_files

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """Represents a single test case with expected changes."""
    name: str
    v1_path: str
    v2_path: str
    expected_changes: List[Dict]

class AccuracyEvaluator:
    """Evaluates the accuracy of the XML diff tool."""
    
    def __init__(self, test_cases: List[TestCase]):
        self.test_cases = test_cases
        self.results = []
    
    def run_tests(self) -> Dict:
        """Run all test cases and collect results."""
        logger.info(f"Running evaluation on {len(self.test_cases)} test cases")
        
        for test_case in self.test_cases:
            try:
                logger.info(f"Testing: {test_case.name}")
                
                # Get actual changes from the tool
                actual_changes = compare_xml_files(test_case.v1_path, test_case.v2_path)
                
                # Compare with expected changes
                metrics = self._calculate_metrics(test_case.expected_changes, actual_changes)
                
                self.results.append({
                    'test_case': test_case.name,
                    'metrics': metrics,
                    'expected_count': len(test_case.expected_changes),
                    'actual_count': len(actual_changes) if actual_changes else 0,
                    'success': metrics['f1_score'] >= 0.9  # 90% threshold for success
                })
                
            except Exception as e:
                logger.error(f"Error in test case {test_case.name}: {str(e)}")
                self.results.append({
                    'test_case': test_case.name,
                    'error': str(e),
                    'success': False
                })
        
        return self._generate_summary()
    
    def _calculate_metrics(self, expected: List[Dict], actual: List[Dict]) -> Dict:
        """Calculate precision, recall, and F1-score."""
        if not expected and not actual:
            return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0}
            
        if not expected or not actual:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        # Convert to sets of tuples for comparison
        expected_set = {self._change_to_tuple(change) for change in expected}
        actual_set = {self._change_to_tuple(change) for change in actual}
        
        # Calculate true positives, false positives, and false negatives
        true_positives = len(expected_set.intersection(actual_set))
        false_positives = len(actual_set - expected_set)
        false_negatives = len(expected_set - actual_set)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    def _change_to_tuple(self, change: Dict) -> tuple:
        """Convert a change dictionary to a hashable tuple."""
        return (
            change.get('action'),
            change.get('node'),
            change.get('xpath'),
            change.get('old_value', ''),
            change.get('new_value', '')
        )
    
    def _generate_summary(self) -> Dict:
        """Generate a summary of test results."""
        if not self.results:
            return {}
            
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.get('success', False))
        success_rate = (passed_tests / total_tests) * 100
        
        # Calculate average metrics
        metrics = ['precision', 'recall', 'f1_score']
        avg_metrics = {}
        
        for metric in metrics:
            values = [r['metrics'][metric] for r in self.results if 'metrics' in r and metric in r['metrics']]
            avg_metrics[f'avg_{metric}'] = sum(values) / len(values) if values else 0
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            **avg_metrics,
            'detailed_results': self.results
        }

def create_test_cases(test_data_dir: str) -> List[TestCase]:
    """Create test cases from the test data directory."""
    test_cases = []
    
    # Create test data directory if it doesn't exist
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Create sample XML files
    sample_v1 = """<?xml version="1.0"?>
<doc>
    <bdy>
        <eqn id="equ6">
            <latex>E = mc^2</latex>
        </eqn>
        <fig id="fig1">
            <ti>Figure 1</ti>
        </fig>
    </bdy>
</doc>"""

    sample_v2 = """<?xml version="1.0"?>
<doc>
    <bdy>
        <eqn id="equ6">
            <latex>E = mc^2 + E_0</latex>
        </eqn>
        <fig id="fig1">
            <ti>Updated Figure 1</ti>
        </fig>
    </bdy>
</doc>"""
    
    # Define file paths
    v1_path = os.path.join(test_data_dir, "sample_v1.xml")
    v2_path = os.path.join(test_data_dir, "sample_v2.xml")
    
    # Write test files
    with open(v1_path, 'w', encoding='utf-8') as f:
        f.write(sample_v1)
    with open(v2_path, 'w', encoding='utf-8') as f:
        f.write(sample_v2)
    
    # Define expected changes
    test_case = TestCase(
        name="sample_test",
        v1_path=v1_path,
        v2_path=v2_path,
        expected_changes=[
            {
                'action': 'update',
                'node': 'latex',
                'xpath': '/doc[1]/bdy/eqn[@id="equ6"]/latex',
                'old_value': 'E = mc^2',
                'new_value': 'E = mc^2 + E_0'
            },
            {
                'action': 'update',
                'node': 'ti',
                'xpath': '/doc[1]/bdy/fig[@id="fig1"]/ti',
                'old_value': 'Figure 1',
                'new_value': 'Updated Figure 1'
            }
        ]
    )
    test_cases.append(test_case)
    
    return test_cases

def main():
    # Set up test data directory
    test_data_dir = os.path.join("data", "test_cases")
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Create or load test cases
    test_cases = create_test_cases(test_data_dir)
    
    # Run evaluation
    evaluator = AccuracyEvaluator(test_cases)
    results = evaluator.run_tests()
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed Tests: {results['passed_tests']}")
    print(f"Success Rate: {results['success_rate']:.2f}%")
    print(f"Average Precision: {results['avg_precision']:.4f}")
    print(f"Average Recall: {results['avg_recall']:.4f}")
    print(f"Average F1-Score: {results['avg_f1_score']:.4f}")
    
    # Save detailed results
    results_file = os.path.join("results", "evaluation_results.json")
    os.makedirs("results", exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    main()
