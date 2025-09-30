# Semantic File Organizer

## Project Overview
Build a Python application that automatically organizes files and folders from an input directory into themed categories in an output directory using semantic similarity analysis. The application should intelligently detect themes from both the input files and existing output directory structure, preserving folder hierarchies during organization.

## Core Requirements

### 1. Input/Output Handling
- Accept two command-line arguments or interactive prompts:
  - `input_directory`: Source folder containing files/folders to organize
  - `output_directory`: Destination folder where themed subdirectories will be created
- Validate that both directories exist before processing
- Support both absolute and relative paths

### 2. Semantic Analysis
- Analyze file contents and/or filenames to determine semantic themes
- Use embeddings-based similarity (e.g., sentence-transformers, OpenAI embeddings, or similar)
- **Support multiple file types with content extraction**:
  - `.txt`, `.md`: Direct text reading
  - `.docx`: Extract text using python-docx
  - `.pdf`: Extract text using PyPDF2 or pdfplumber
  - `.xlsx`: Extract cell contents and sheet names using openpyxl
  - `.pptx`: Extract slide text and titles using python-pptx
  - Other files: Fall back to filename and metadata analysis
- Cluster similar items together to identify common themes
- Handle files that cannot be read (corrupted, password-protected, etc.) gracefully

### 3. Theme Detection & Matching
- **Existing Themes**: Scan the output directory for existing subdirectories to use as potential theme categories
- **New Themes**: When files don't match existing themes, create new theme categories
- Generate descriptive, human-readable folder names for themes (e.g., "Financial_Documents", "Project_Photos", "Python_Scripts")
- Use semantic similarity to match input items against existing output directory themes first

### 4. Folder Structure Preservation
- **Critical**: When a folder is detected in the input directory, move the entire folder as a unit
- Do NOT split up folders by moving individual files to different theme directories
- The folder structure within moved folders should remain intact
- Treat each top-level folder in the input directory as an atomic unit

### 5. Organization Logic
```
If item is a folder:
    - Analyze the folder name and/or sample of contents
    - Find best matching theme (existing or new)
    - Move entire folder to that theme directory
    
If item is a file:
    - Analyze the file (content/name/metadata)
    - Find best matching theme (existing or new)
    - Move file to that theme directory
```

### 6. Conflict Resolution
- **Default Behavior**: When a file/folder with the same name already exists in the destination, automatically rename the incoming file
- **Renaming Strategy**:
  - For files: Insert a counter before the file extension: `document.pdf` → `document_1.pdf`, `document_2.pdf`, etc.
  - For folders: Append counter to folder name: `ProjectFolder` → `ProjectFolder_1`, `ProjectFolder_2`, etc.
  - Use intelligent numbering: Check for highest existing number and increment (if `file_1.pdf` and `file_3.pdf` exist, next is `file_4.pdf`)
- **Alternative Behaviors** (configurable via command-line flag):
  - `skip`: Skip the file and log it (don't move)
  - `overwrite`: Replace the existing file (use with caution)
  - `timestamp`: Append timestamp instead of counter: `document_20250929_143022.pdf`
  - `prompt`: Ask user interactively what to do (for small batches)
- **Collision Detection**:
  - Check for filename conflicts before moving any file
  - Handle case-insensitive filesystems (Windows, macOS) appropriately
  - Preserve original file extension in renamed files
- **Logging**: Always log when files are renamed due to conflicts, showing both original and new names

## Technical Specifications

### Required Libraries

#### Core Functionality
- **sentence-transformers** or **openai**: For semantic embeddings
- **scikit-learn**: For clustering (DBSCAN, KMeans, or hierarchical clustering)
- **pathlib**: For robust path handling
- **argparse**: For command-line interface
- **logging**: For detailed operation logs
- **numpy**: For efficient numerical operations and similarity calculations

#### File Type Handling
- **python-docx**: Extract text from .docx files
- **PyPDF2** or **pdfplumber**: Extract text from .pdf files (pdfplumber recommended for better accuracy)
- **openpyxl**: Read .xlsx files and extract cell contents
- **python-pptx**: Extract text from .pptx presentations
- **markdown**: Parse .md files (optional, can use plain text reading)
- **chardet**: Detect text file encodings for robust text file reading

#### Optional but Recommended
- **python-magic**: For reliable file type detection
- **tqdm**: Progress bars for long operations

### Architecture
```
semantic_organizer/
├── main.py                 # Entry point, CLI argument parsing
├── analyzer.py             # Semantic analysis and embedding generation
├── file_extractors.py      # File type-specific content extraction
├── clusterer.py            # Theme detection and clustering logic
├── organizer.py            # File/folder moving logic
├── theme_matcher.py        # Match items to existing/new themes
├── utils.py                # Helper functions (validation, logging)
├── requirements.txt        # Dependencies
└── README.md              # Usage documentation
```

### Key Classes/Functions

#### `FileExtractor`
- `extract_text(file_path)`: Main dispatcher that routes to appropriate extractor
- `extract_from_docx(file_path)`: Extract text from Word documents
- `extract_from_pdf(file_path)`: Extract text from PDFs
- `extract_from_xlsx(file_path)`: Extract cell contents and sheet names from Excel files
- `extract_from_pptx(file_path)`: Extract slide text and titles from PowerPoint
- `extract_from_text(file_path)`: Read plain text and markdown files with encoding detection
- `get_file_type(file_path)`: Determine file type from extension
- `handle_extraction_error(file_path, error)`: Log and handle files that can't be read

#### `SemanticAnalyzer`
- `analyze_item(path)`: Generate semantic embedding for file/folder
- `get_text_content(file_path)`: Extract text using FileExtractor
- `analyze_folder(folder_path)`: Summarize folder contents by sampling files
- `generate_embedding(text)`: Convert text to semantic vector
- `preprocess_text(text)`: Clean and normalize text before embedding

#### `ThemeMatcher`
- `load_existing_themes(output_dir)`: Scan output directory for theme folders
- `match_to_theme(embedding, existing_themes)`: Find best matching theme
- `create_new_theme(items)`: Generate new theme name from clustered items
- `calculate_similarity(embedding1, embedding2)`: Compute cosine similarity

#### `FileOrganizer`
- `move_item(source, destination, conflict_mode)`: Safely move file/folder with conflict handling
- `handle_conflict(source, destination, mode)`: Resolve naming conflicts based on mode
- `generate_unique_filename(destination, original_name)`: Create numbered filename that doesn't conflict
- `get_next_available_number(base_path, base_name, extension)`: Find next available counter number
- `apply_timestamp_suffix(filename)`: Generate timestamp-based unique name
- `check_case_insensitive_conflict(destination)`: Handle case-insensitive filesystem issues
- `create_theme_folder(output_dir, theme_name)`: Create new theme directory
- `preserve_structure(folder_path)`: Ensure folder hierarchy is maintained
- `validate_move_operation(source, destination)`: Pre-check for issues before moving

#### `Clusterer`
- `cluster_items(embeddings, items)`: Group similar items
- `determine_optimal_clusters(embeddings)`: Auto-detect number of themes
- `assign_theme_names(clusters)`: Generate descriptive names

## File Type Handling Details

### Text Extraction Strategies

#### `.txt` and `.md` Files
```python
# Use chardet to detect encoding
# Read entire file content
# For .md files, optionally strip markdown syntax or keep it for context
```

#### `.docx` Files
```python
# Use python-docx
# Extract from document.paragraphs
# Include text from tables (document.tables)
# Extract headers and footers if significant
# Concatenate all text sections
```

#### `.pdf` Files
```python
# Use pdfplumber (preferred) or PyPDF2
# Extract text from all pages
# Handle multi-column layouts
# Skip pages that are purely images (no text layer)
# Set max page limit for very large PDFs (e.g., first 50 pages)
# Log warning if PDF is password-protected or corrupted
```

#### `.xlsx` Files
```python
# Use openpyxl
# Extract sheet names (these often indicate content themes)
# Sample cells from each sheet (e.g., first 100 rows)
# Include headers from each sheet
# Concatenate all text with sheet names as context
# Skip empty sheets
```

#### `.pptx` Files
```python
# Use python-pptx
# Extract text from all slides
# Include slide titles (often most semantically meaningful)
# Extract text from shapes and text boxes
# Include notes if present
# Preserve order to maintain context
```

### Content Extraction Best Practices

1. **Size Limits**: For very large files, extract only a representative sample (e.g., first 10,000 words)
2. **Error Handling**: If extraction fails for any file, fall back to filename-based analysis
3. **Encoding Detection**: Use chardet for text files to handle various encodings
4. **Text Cleaning**: Remove excessive whitespace, special characters that don't add semantic value
5. **Metadata**: When available, include file metadata (author, title, keywords) in analysis
6. **Combining Signals**: Combine filename, content, and metadata for robust analysis

### Example Extraction Function Flow
```
1. Detect file type from extension
2. Call appropriate extractor
3. If extraction succeeds:
   - Clean and normalize text
   - Combine with filename for context
   - Return combined text
4. If extraction fails:
   - Log error with details
   - Return filename only
   - Mark file as "content_unavailable"
```

## Conflict Resolution Implementation

### Detailed Renaming Logic

When moving a file to the output directory, the program must check if a file with the same name already exists in the target theme folder. The **default behavior is to rename** the incoming file to avoid conflicts.

#### Algorithm for Generating Unique Filenames

```python
def generate_unique_filename(destination_dir, original_filename):
    """
    Generate a unique filename by appending a counter.
    
    Examples:
        report.pdf → report_1.pdf (if report.pdf exists)
        report.pdf → report_2.pdf (if report.pdf and report_1.pdf exist)
        data.xlsx → data_1.xlsx
        folder → folder_1 (for directories)
    """
    1. Parse filename into base_name and extension
    2. Check if destination_dir/original_filename exists
    3. If no conflict, return original_filename
    4. If conflict exists:
        a. Find all existing files matching pattern: base_name_N.extension
        b. Extract all counter numbers (N)
        c. Find maximum N
        d. Return base_name_{max_N + 1}.extension
```

#### Special Cases

**Case 1: Files already numbered**
```
Input: report_5.pdf
Existing files: report.pdf, report_1.pdf, report_5.pdf
Output: report_6.pdf (not report_5_1.pdf)
```

**Case 2: Folders with same name**
```
Input folder: ProjectData
Existing: ProjectData, ProjectData_1
Output: ProjectData_2
```

**Case 3: Multiple dots in filename**
```
Input: my.data.backup.xlsx
Existing: my.data.backup.xlsx
Output: my.data.backup_1.xlsx (counter before final extension only)
```

**Case 4: Case-insensitive filesystems**
```
On Windows/macOS:
Input: Document.PDF
Existing: document.pdf
Result: Treated as conflict → document_1.PDF
```

### Configuration Flag Behavior

The `--conflict-resolution` flag supports these modes:

1. **`rename` (default)**: Automatically rename with counter
   ```bash
   python main.py input/ output/ --conflict-resolution rename
   ```

2. **`skip`**: Don't move conflicting files, just log them
   ```bash
   python main.py input/ output/ --conflict-resolution skip
   ```
   - Useful for incremental organization
   - Logs all skipped files in report

3. **`overwrite`**: Replace existing files
   ```bash
   python main.py input/ output/ --conflict-resolution overwrite
   ```
   - **Warning**: This is destructive!
   - Should prompt for confirmation unless --force flag is used

4. **`timestamp`**: Use timestamp instead of counter
   ```bash
   python main.py input/ output/ --conflict-resolution timestamp
   ```
   - Format: `filename_YYYYMMDD_HHMMSS.ext`
   - Example: `report_20250929_143052.pdf`

5. **`prompt`**: Ask user for each conflict (interactive mode)
   ```bash
   python main.py input/ output/ --conflict-resolution prompt
   ```
   - Show both files (size, modified date)
   - Options: (r)ename, (s)kip, (o)verwrite, (a)uto-rename all remaining

### Conflict Logging

All conflicts should be logged with details:
```
[CONFLICT] File: report.pdf
  Source: /input/documents/report.pdf
  Destination: /output/Financial_Documents/report.pdf (already exists)
  Action: Renamed to report_1.pdf
  Reason: Conflict resolution mode = rename
```

### Performance Considerations

For directories with many files:
- Cache list of existing filenames in destination to avoid repeated filesystem checks
- Use set-based lookups for O(1) conflict detection
- Pre-scan destination directory once rather than checking per-file

## Implementation Guidelines

### Phase 1: Setup & Validation
1. Parse command-line arguments
2. Validate input and output directories
3. Initialize logging
4. Load or initialize embedding model

### Phase 2: Discovery
1. Scan output directory for existing themes
2. Generate embeddings for existing theme folder names
3. Recursively discover all files/folders in input directory
4. Separate top-level folders from individual files

### Phase 3: Analysis
1. Generate embeddings for all items in input directory
2. For folders: create composite embedding from folder name + sample contents
3. For files: 
   - Extract text content based on file type (.docx, .pdf, .xlsx, .pptx, .md, .txt)
   - Generate embedding from extracted content + filename
   - Handle extraction errors gracefully (fall back to filename-only analysis)
   - Respect file size limits and sampling for large files

### Phase 4: Theme Matching
1. Compare each input item's embedding to existing output themes
2. Set a similarity threshold (e.g., 0.7) for matching
3. Items above threshold → matched to existing theme
4. Items below threshold → candidates for new theme creation

### Phase 5: Clustering & Theme Creation
1. Cluster unmatched items together
2. Generate descriptive theme names for new clusters
3. Create new theme folders in output directory

### Phase 6: Organization
1. Move folders as complete units to their theme directories
2. Move individual files to their theme directories
3. **For each move operation**:
   - Check if destination filename already exists
   - If conflict exists, apply conflict resolution strategy (default: rename with counter)
   - Generate unique filename if needed using intelligent numbering
   - Perform the move operation
4. Preserve file permissions and timestamps during moves
5. Log all operations, especially renames due to conflicts
6. Update internal tracking of moved items

### Phase 7: Reporting
1. Print summary with key metrics:
   - Total files/folders processed
   - Items moved successfully
   - New themes created vs existing themes used
   - Conflicts encountered and resolution actions taken
   - Files skipped (due to errors, conflicts with skip mode, etc.)
   - Files renamed due to conflicts (with count)
2. Save detailed log file in output directory
3. If conflicts occurred, list all renamed files with original and new names
4. Show processing time and performance metrics

## Configuration Options

Provide command-line flags for:
- `--similarity-threshold`: Minimum similarity for theme matching (default: 0.7)
- `--conflict-resolution`: How to handle naming conflicts - `rename` (default), `skip`, `overwrite`, `timestamp`, `prompt`
- `--dry-run`: Preview organization without moving files
- `--verbose`: Detailed logging output
- `--embedding-model`: Which model to use for embeddings
- `--min-cluster-size`: Minimum items needed to form a new theme (default: 2)
- `--max-file-size`: Maximum file size to process in MB (default: 100)
- `--sample-limit`: Maximum characters to extract from large files (default: 50000)
- `--skip-extensions`: File extensions to skip entirely (e.g., .exe, .dll)
- `--content-weight`: Weight of content vs filename in similarity (default: 0.8 for content, 0.2 for filename)
- `--force`: Skip confirmation prompts (use with overwrite mode)

## Error Handling

- Gracefully handle permission errors
- Skip files that cannot be read/analyzed
- Provide clear error messages for all failure scenarios
- Continue processing even if individual items fail
- Maintain a log of all errors and warnings
- **Conflict handling errors**:
  - Unable to rename file after multiple attempts
  - Filename too long after adding counter
  - Destination directory becomes unavailable mid-operation
- **Rollback considerations**: In case of critical errors, log all moves made for potential manual rollback
- Exit codes: 0 for success, 1 for partial success with errors, 2 for complete failure

## Testing Considerations

Create test scenarios for:
1. Empty input directory
2. Input directory with only files (no folders)
3. Input directory with only folders
4. Mixed files and folders
5. Output directory with existing themes
6. Empty output directory
7. Files with same names in different input folders
8. **All supported file types (.docx, .pdf, .xlsx, .pptx, .md, .txt)**
9. **Mixed file types in same directory**
10. **Corrupted or password-protected files**
11. **Very large files (100+ pages PDF, large spreadsheets)**
12. **Files with non-English text (UTF-8, Unicode handling)**
13. Large directory structures
14. Permission errors
15. **Empty or content-free files (blank PDFs, empty documents)**
16. **Conflict resolution scenarios**:
    - File with same name already in output directory
    - Multiple files with same name from different input locations
    - Files already numbered (e.g., report_3.pdf)
    - Folders with same name as existing folders
    - Files with multiple dots in filename
    - Case-insensitive conflicts (Document.pdf vs document.PDF)
    - Sequential conflicts (moving 10 files named "document.pdf")
17. **Different conflict resolution modes** (rename, skip, overwrite, timestamp, prompt)
18. **Dry run mode** - verify no files actually moved

## Performance Considerations

- For large directories (1000+ files), implement batch processing
- Cache embeddings to avoid recomputation
- Use efficient similarity computation (cosine similarity with numpy)
- Consider parallel processing for embedding generation
- Provide progress indicators for long operations

## Documentation Requirements

### README.md should include:
- Installation instructions
- Quick start guide
- Detailed usage examples
- Configuration options with examples of each conflict resolution mode
- **Conflict resolution behavior** with clear examples of how files are renamed
- Troubleshooting section
- Limitations and known issues
- FAQ section addressing common questions:
  - "What happens if a file already exists?"
  - "Can I undo the organization?"
  - "How are folders vs files treated?"

### Code Documentation:
- Docstrings for all classes and functions
- Type hints throughout
- Inline comments for complex logic
- Examples in docstrings

## Success Criteria

The application should:
1. ✓ Successfully organize files by semantic similarity
2. ✓ Preserve folder structures when moving folders
3. ✓ Respect existing output directory themes
4. ✓ Create meaningful, human-readable theme names
5. ✓ Handle edge cases gracefully
6. ✓ Provide clear feedback and logging
7. ✓ Be configurable and extensible
8. ✓ Process typical user directories (100-500 items) in under 2 minutes

## Constraints

- Preserve original input directory structure (non-destructive option available)
- Do not modify file contents, only move them
- Maintain file permissions and timestamps
- Support cross-platform usage (Windows, macOS, Linux)
- Keep dependencies minimal and well-documented

## Future Enhancements (Optional)

- GUI interface
- Watch mode for continuous organization
- Undo functionality
- Integration with cloud storage
- Custom theme definitions via config file
- Machine learning model fine-tuning based on user corrections