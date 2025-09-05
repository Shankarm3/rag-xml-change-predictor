# RAG XML Change Predictor

A tool for analyzing and predicting changes between XML document versions using RAG (Retrieval-Augmented Generation).

## Project Structure

```
project_root/
├── data/                   # Input XML files (v1/v2 versions)
├── results/                # Output results and reports
├── test_data/              # Test XML files (automatically generated during tests)
├── tests/                  # Test suite
│   ├── __init__.py
│   └── test_xml_diff.py    # XML diff test cases
├── utils/                  # Utility modules
├── main.py                 # Main application script
└── README.md               # This file
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Tests

```bash
python main.py --test
```

### Processing Files

Process a single XML file:
```bash
python main.py --file path/to/your/file.xml
```

Process all XML files in a directory:
```bash
python main.py --dir path/to/xml/files
```

## Development

### Adding Tests

1. Add new test cases to `tests/test_xml_diff.py`
2. Run tests with: `python -m pytest tests/`

### Code Style

This project follows PEP 8 style guidelines. Use the following tools:

```bash
# Format code with black
black .

# Check for style issues
flake8 .
```

## License

MIT
