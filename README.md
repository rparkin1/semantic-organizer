# Semantic File Organizer

Automatically organize files and folders from an input directory into themed categories in an output directory using semantic similarity analysis. The application intelligently detects themes from both the input files and existing output directory structure, preserving folder hierarchies during organization.

## 🚀 Features

- **Semantic Analysis**: Uses AI embeddings to understand file content and organize by meaning, not just file type
- **Multi-format Support**: Extracts content from `.txt`, `.md`, `.docx`, `.pdf`, `.xlsx`, `.pptx` files
- **Intelligent Theme Detection**: Automatically detects existing themes and creates new ones as needed
- **Folder Structure Preservation**: Maintains folder hierarchies when moving directories
- **Conflict Resolution**: Multiple strategies for handling naming conflicts (rename, skip, overwrite, timestamp, interactive)
- **Dry Run Mode**: Preview organization without actually moving files
- **Progress Tracking**: Real-time progress indicators for long operations
- **Comprehensive Logging**: Detailed logs of all operations and conflicts

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Clone or download the project
cd semantic-file-organizer

# Install required packages
pip install -r requirements.txt
```

### Required Libraries

The application uses several key libraries:

- `sentence-transformers`: For semantic embeddings
- `scikit-learn`: For clustering algorithms
- `python-docx`: For Word document processing
- `pdfplumber`: For PDF text extraction
- `openpyxl`: For Excel spreadsheet processing
- `python-pptx`: For PowerPoint processing
- `chardet`: For text encoding detection
- `tqdm`: For progress bars

## 🚀 Quick Start

### Basic Usage

```bash
# Organize files from 'input_folder' into 'output_folder'
python -m semantic_organizer input_folder output_folder
```

### Preview Mode (Recommended First Run)

```bash
# See what would be organized without moving files
python -m semantic_organizer input_folder output_folder --dry-run
```

### Common Usage Patterns

```bash
# Organize with verbose output
python -m semantic_organizer ~/Downloads ~/Documents/Organized --verbose

# Use timestamp-based conflict resolution
python -m semantic_organizer ./messy_folder ./organized --conflict-resolution timestamp

# Skip certain file types
python -m semantic_organizer ./input ./output --skip-extensions .exe .dll .tmp

# Adjust similarity threshold (0.0-1.0)
python -m semantic_organizer ./input ./output --similarity-threshold 0.8
```

## 📋 Command Line Options

### Required Arguments

- `input_directory`: Source folder containing files/folders to organize
- `output_directory`: Destination folder where themed subdirectories will be created

### Optional Arguments

| Option | Default | Description |
|--------|---------|-------------|
| `--similarity-threshold` | 0.7 | Minimum similarity for theme matching (0.0-1.0) |
| `--conflict-resolution` | rename | How to handle naming conflicts |
| `--dry-run` | False | Preview organization without moving files |
| `--verbose` | False | Enable detailed logging output |
| `--embedding-model` | intfloat/e5-base | Sentence transformer model to use |
| `--min-cluster-size` | 2 | Minimum items needed to form a new theme |
| `--max-file-size` | 100 | Maximum file size to process in MB |
| `--sample-limit` | 50000 | Maximum characters to extract from large files |
| `--skip-extensions` | None | File extensions to skip entirely |
| `--content-weight` | 0.8 | Weight of content vs filename in similarity |
| `--filename-weight` | 0.2 | Weight of filename vs content in similarity |
| `--log-file` | None | Optional file to write detailed logs to |

### Conflict Resolution Modes

1. **`rename` (default)**: Automatically rename with counter
   ```bash
   document.pdf → document_1.pdf
   ```

2. **`skip`**: Don't move conflicting files, just log them
   - Useful for incremental organization
   - Logs all skipped files in report

3. **`overwrite`**: Replace existing files
   - ⚠️ **Warning**: This is destructive!
   - Requires `--force` flag or interactive confirmation

4. **`timestamp`**: Use timestamp instead of counter
   ```bash
   document.pdf → document_20241229_143052.pdf
   ```

5. **`prompt`**: Ask user for each conflict (interactive mode)
   - Shows file details for comparison
   - Options: (r)ename, (s)kip, (o)verwrite, (a)uto-rename remaining

## 🎯 How It Works

The semantic file organizer follows a systematic 7-phase process:

### Phase 1: Setup & Validation
- Validates input/output directories
- Initializes AI models and components
- Sets up logging and configuration

### Phase 2: Discovery
- Scans input directory for all files and folders
- Identifies existing themes in output directory
- Generates embeddings for existing theme names

### Phase 3: Analysis
- Extracts text content from supported file types
- Generates semantic embeddings for all items
- Handles extraction errors gracefully

### Phase 4: Theme Matching
- Compares each item against existing themes
- Uses similarity threshold to determine matches
- Separates matched vs unmatched items

### Phase 5: Clustering & Theme Creation
- Groups unmatched items using clustering algorithms
- Generates descriptive names for new themes
- Creates theme directories as needed

### Phase 6: Organization
- Moves items to their designated theme directories
- Applies conflict resolution strategies
- Preserves folder structures for directories

### Phase 7: Reporting
- Generates comprehensive operation report
- Lists all conflicts and resolutions
- Provides performance metrics

## 📁 File Type Support

### Supported Formats with Content Extraction

| File Type | Extensions | Extraction Method |
|-----------|------------|-------------------|
| Text Files | `.txt`, `.md` | Direct text reading with encoding detection |
| Word Documents | `.docx` | python-docx (paragraphs, tables, headers) |
| PDF Files | `.pdf` | pdfplumber or PyPDF2 (text layer extraction) |
| Excel Spreadsheets | `.xlsx` | openpyxl (sheet names, cell contents, headers) |
| PowerPoint | `.pptx` | python-pptx (slide text, titles, notes) |

### Fallback Processing

- **Unsupported formats**: Uses filename and metadata analysis
- **Corrupted files**: Gracefully falls back to filename-based processing
- **Large files**: Automatically samples content to respect size limits
- **Empty files**: Handled without errors

### Content Extraction Best Practices

- **Size Limits**: Large files are sampled (first 50,000 characters by default)
- **Encoding Detection**: Automatic encoding detection for text files
- **Error Recovery**: Robust error handling with graceful fallbacks
- **Multi-language Support**: Unicode and UTF-8 text handling

## 🔧 Configuration Examples

### High Accuracy Mode
```bash
python -m semantic_organizer input/ output/ \
  --similarity-threshold 0.9 \
  --min-cluster-size 3 \
  --content-weight 0.9 \
  --filename-weight 0.1
```

### Fast Processing Mode
```bash
python -m semantic_organizer input/ output/ \
  --max-file-size 50 \
  --sample-limit 25000 \
  --skip-extensions .mp4 .avi .mkv
```

### Safe Exploration Mode
```bash
python -m semantic_organizer input/ output/ \
  --dry-run \
  --verbose \
  --conflict-resolution prompt
```

## 📊 Understanding the Output

### Example Organization Result

```
INPUT DIRECTORY:
├── machine_learning_paper.pdf
├── ai_research_notes.txt
├── vacation_photos/
│   ├── beach.jpg
│   └── mountains.jpg
├── budget_2024.xlsx
└── presentation.pptx

OUTPUT DIRECTORY (After Organization):
├── Research_Documents/
│   ├── machine_learning_paper.pdf
│   └── ai_research_notes.txt
├── Images/
│   └── vacation_photos/      # Folder structure preserved
│       ├── beach.jpg
│       └── mountains.jpg
├── Spreadsheets/
│   └── budget_2024.xlsx
└── Presentations/
    └── presentation.pptx
```

### Report Example

```
==================================================
SEMANTIC FILE ORGANIZATION REPORT
==================================================
Input Directory: /home/user/Downloads
Output Directory: /home/user/Documents/Organized
Processing Time: 12.3 seconds
Dry Run Mode: No

STATISTICS:
  Total Items Found: 47
  Files: 42
  Folders: 5
  Items Moved: 45
  Items Skipped: 2
  Existing Themes Used: 3
  New Themes Created: 4
  Conflicts Encountered: 8
  Errors: 1

CONFLICT RESOLUTION SUMMARY:
  renamed: 7
  skipped: 1

SUCCESS RATE: 95.7%
==================================================
```

## 🔍 Troubleshooting

### Common Issues

**1. "No module named 'sentence_transformers'"**
```bash
pip install sentence-transformers
```

**2. "Permission denied" errors**
- Ensure you have read access to input directory
- Ensure you have write access to output directory
- Run with appropriate permissions

**3. "Out of memory" errors**
- Reduce `--max-file-size` limit
- Decrease `--sample-limit`
- Process smaller batches

**4. Poor theme detection**
- Adjust `--similarity-threshold` (try 0.6 or 0.8)
- Increase `--content-weight` for content-heavy analysis
- Decrease `--min-cluster-size` for more granular themes

**5. Large file processing is slow**
- Increase `--max-file-size` limit if you have sufficient RAM
- Use `--skip-extensions` to exclude large media files
- Consider processing in smaller batches

### Debug Mode

For detailed debugging information:

```bash
python -m semantic_organizer input/ output/ \
  --verbose \
  --log-file debug.log \
  --dry-run
```

### Performance Tips

- **First run**: Always use `--dry-run` to preview results
- **Large directories**: Process in smaller batches
- **Network storage**: Copy files locally first for better performance
- **SSD storage**: Significantly faster than traditional hard drives

## ❓ FAQ

### General Questions

**Q: Can I undo the organization?**
A: There's no built-in undo, but the detailed logs show all moves. You can manually reverse them, or keep backups of important directories.

**Q: What happens if a file already exists in the destination?**
A: The conflict resolution strategy determines this:
- `rename` (default): Adds a counter (`file_1.txt`)
- `skip`: Leaves the original file in place
- `overwrite`: Replaces the existing file
- `timestamp`: Adds timestamp to filename
- `prompt`: Asks you what to do

**Q: How are folders treated differently from files?**
A: Folders are moved as complete units. Their internal structure is never modified, and files within folders are never separated into different themes.

**Q: Can I organize the same directory multiple times?**
A: Yes! The system detects existing themes and can incrementally organize new files into them.

### Technical Questions

**Q: Which AI model is used for semantic analysis?**
A: By default, `intfloat/e5-base` from sentence-transformers. You can specify different models with `--embedding-model`.

**Q: How much disk space and RAM is needed?**
A: Minimal disk space (just for organized files). RAM usage depends on file sizes and number of files being processed simultaneously. Typical usage: 500MB-2GB RAM.

**Q: Does it work offline?**
A: Yes, after the first run downloads the AI model. No internet connection needed for subsequent runs.

**Q: Can I customize the theme names?**
A: Theme names are auto-generated based on file content and naming patterns. You can manually rename theme directories after organization.

### Privacy and Security

**Q: Is my data sent anywhere?**
A: No, all processing is done locally on your machine. No data is transmitted over the internet.

**Q: Are file contents modified?**
A: Never. The application only reads files and moves them. File contents are never altered.

## 📈 Advanced Usage

### Custom Embedding Models

You can use different sentence-transformer models:

```bash
# Multilingual model
python -m semantic_organizer input/ output/ \
  --embedding-model paraphrase-multilingual-MiniLM-L12-v2

# Fast alternative model
python -m semantic_organizer input/ output/ \
  --embedding-model all-MiniLM-L6-v2
```

### Batch Processing Large Directories

For very large directories (1000+ files):

```bash
# Process in smaller chunks with higher thresholds
python -m semantic_organizer input/ output/ \
  --similarity-threshold 0.8 \
  --min-cluster-size 5 \
  --max-file-size 50
```

### Content vs Filename Weight Tuning

Adjust the balance between content analysis and filename analysis:

```bash
# Prefer content over filenames
python -m semantic_organizer input/ output/ \
  --content-weight 0.9 \
  --filename-weight 0.1

# Prefer filenames (faster processing)
python -m semantic_organizer input/ output/ \
  --content-weight 0.3 \
  --filename-weight 0.7
```

## 🧪 Testing

### Run the Test Suite

```bash
# Run all tests
python tests/run_tests.py

# Run specific test modules
python -m unittest tests.test_file_extractors
python -m unittest tests.test_analyzer
python -m unittest tests.test_organizer
python -m unittest tests.test_integration
```

### Test with Sample Data

Create a test directory with various file types to see how the organizer works:

```bash
mkdir test_input
echo "Machine learning research paper content" > test_input/ml_paper.txt
echo "Artificial intelligence findings" > test_input/ai_study.txt
echo "Recipe for chocolate cake" > test_input/recipe.txt
echo "Financial budget for 2024" > test_input/budget.txt

# Run with dry-run first
python -m semantic_organizer test_input test_output --dry-run --verbose
```

## 🤝 Contributing

This project follows clean code principles and comprehensive testing practices. Key areas for contribution:

- Additional file format support
- Performance optimizations
- UI/GUI development
- Advanced clustering algorithms
- Multi-language support

## 📄 License

This project is intended for educational and personal use. Please ensure you comply with the licenses of all dependencies, particularly the sentence-transformers library.

## 🙏 Acknowledgments

- **sentence-transformers**: For providing excellent semantic embedding models
- **scikit-learn**: For robust clustering algorithms
- **pdfplumber**: For superior PDF text extraction
- **python-docx, openpyxl, python-pptx**: For Microsoft Office format support

---

**Need help?** Run `python -m semantic_organizer --help` for quick reference, or create an issue in the project repository.