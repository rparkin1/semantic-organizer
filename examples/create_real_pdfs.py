"""
Create actual PDF files using reportlab, with proper path handling.
"""

from pathlib import Path


def create_real_pdfs(base_dir: Path):
    """Create actual PDF files using reportlab."""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors

        styles = getSampleStyleSheet()

        # Research Paper PDF
        doc1 = SimpleDocTemplate(str(base_dir / "ai_research_paper.pdf"), pagesize=letter)
        story1 = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=1  # Center
        )

        story1.append(Paragraph("Advances in Natural Language Processing", title_style))
        story1.append(Spacer(1, 12))

        # Abstract
        story1.append(Paragraph("Abstract", styles['Heading2']))
        abstract_text = """
        This paper presents recent advances in natural language processing (NLP) using
        transformer-based architectures. We explore the applications of large language models
        in text classification, sentiment analysis, and machine translation. Our experiments
        show significant improvements over traditional approaches, with accuracy increases
        of up to 15% across multiple benchmarks. The research demonstrates the effectiveness
        of pre-trained models like BERT, GPT-3, and T5 in various NLP tasks.
        """
        story1.append(Paragraph(abstract_text, styles['Normal']))
        story1.append(Spacer(1, 12))

        # Introduction
        story1.append(Paragraph("1. Introduction", styles['Heading2']))
        intro_text = """
        Natural Language Processing has undergone a revolution with the introduction of
        transformer architectures. Models like BERT, GPT, and T5 have achieved
        state-of-the-art performance on numerous NLP tasks. This research investigates
        the practical applications of these models in real-world scenarios including
        document classification, sentiment analysis, and machine translation tasks.
        """
        story1.append(Paragraph(intro_text, styles['Normal']))
        story1.append(Spacer(1, 12))

        # Methodology
        story1.append(Paragraph("2. Methodology", styles['Heading2']))
        method_text = """
        We conducted experiments using three different transformer models:
        BERT for classification tasks, GPT-3 for text generation, and T5 for
        translation tasks. Each model was fine-tuned on domain-specific datasets
        and evaluated using standard metrics including accuracy, precision, recall, and F1-score.
        The experimental setup involved cross-validation and statistical significance testing.
        """
        story1.append(Paragraph(method_text, styles['Normal']))

        doc1.build(story1)

        # Financial Report PDF
        doc2 = SimpleDocTemplate(str(base_dir / "financial_quarterly_report.pdf"), pagesize=A4)
        story2 = []

        story2.append(Paragraph("Q4 2024 Financial Report", title_style))
        story2.append(Spacer(1, 20))

        story2.append(Paragraph("Executive Summary", styles['Heading2']))
        exec_summary = """
        The fourth quarter of 2024 showed strong financial performance with revenue
        growth of 12% compared to Q4 2023. Net income increased by 8%, driven by
        improved operational efficiency and cost management initiatives. Key highlights
        include expansion into new markets and successful product launches in our
        cloud services division.
        """
        story2.append(Paragraph(exec_summary, styles['Normal']))
        story2.append(Spacer(1, 12))

        # Financial table
        story2.append(Paragraph("Financial Highlights", styles['Heading2']))

        data = [
            ['Metric', 'Q4 2024', 'Q4 2023', 'Change'],
            ['Revenue ($M)', '125.6', '112.1', '+12.0%'],
            ['Net Income ($M)', '18.4', '17.0', '+8.2%'],
            ['Gross Margin (%)', '42.3', '40.1', '+2.2pp'],
            ['Operating Margin (%)', '15.2', '14.8', '+0.4pp'],
            ['EBITDA ($M)', '22.8', '20.5', '+11.2%']
        ]

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story2.append(table)
        story2.append(Spacer(1, 12))

        outlook_text = """
        Looking ahead to 2025, we expect continued growth driven by digital transformation
        initiatives and expansion into emerging markets. We project revenue growth of
        10-15% and maintain our commitment to innovation and operational excellence.
        Key investment areas include artificial intelligence, cloud infrastructure, and
        sustainable business practices.
        """
        story2.append(Paragraph("2025 Outlook", styles['Heading2']))
        story2.append(Paragraph(outlook_text, styles['Normal']))

        doc2.build(story2)

        # User Manual PDF
        doc3 = SimpleDocTemplate(str(base_dir / "software_user_manual.pdf"), pagesize=letter)
        story3 = []

        story3.append(Paragraph("DataAnalyzer Pro User Manual", title_style))
        story3.append(Paragraph("Version 3.2", styles['Normal']))
        story3.append(Spacer(1, 20))

        story3.append(Paragraph("Table of Contents", styles['Heading2']))
        toc_text = """
        1. Installation Instructions<br/>
        2. Getting Started<br/>
        3. User Interface Overview<br/>
        4. Data Import and Export<br/>
        5. Analysis Features<br/>
        6. Visualization Tools<br/>
        7. Reporting Functions<br/>
        8. Troubleshooting<br/>
        9. Technical Support
        """
        story3.append(Paragraph(toc_text, styles['Normal']))
        story3.append(Spacer(1, 12))

        story3.append(Paragraph("1. Installation Instructions", styles['Heading2']))
        install_text = """
        DataAnalyzer Pro is a comprehensive business intelligence and data analysis
        software solution. To install the software, please follow these steps:<br/>
        <br/>
        1. Download the installer from our website<br/>
        2. Run the installer as administrator<br/>
        3. Follow the setup wizard instructions<br/>
        4. Enter your license key when prompted<br/>
        5. Restart your computer when installation completes<br/>
        6. Launch the application from the desktop shortcut
        """
        story3.append(Paragraph(install_text, styles['Normal']))
        story3.append(Spacer(1, 12))

        story3.append(Paragraph("2. Getting Started", styles['Heading2']))
        start_text = """
        After installation, the first time you launch DataAnalyzer Pro, you will be
        guided through an initial setup process. This includes creating your user
        profile, configuring data source connections, and setting up your workspace
        preferences. The setup wizard will walk you through each step and provide
        helpful tips for new users.
        """
        story3.append(Paragraph(start_text, styles['Normal']))

        doc3.build(story3)

        print("✓ Created 3 real PDF files using reportlab")
        return True

    except ImportError:
        print("⚠ reportlab not available for creating real PDFs")
        return False
    except Exception as e:
        print(f"⚠ Error creating PDFs: {e}")
        return False


def create_alternative_pdfs(base_dir: Path):
    """Create text files with PDF-like content that our extractor can handle."""

    # Remove the incorrect PDF files
    for pdf_file in base_dir.glob("*.pdf"):
        pdf_file.unlink()
        print(f"Removed incorrect PDF file: {pdf_file.name}")

    # Create properly formatted content files that simulate PDF extraction
    pdf_contents = {
        "ai_research_paper_content.txt": """
AI Research Paper: Advances in Natural Language Processing

Abstract
This paper presents recent advances in natural language processing using transformer-based architectures. We explore applications of large language models in text classification, sentiment analysis, and machine translation. Our experiments show significant improvements over traditional approaches.

Keywords: natural language processing, transformers, BERT, GPT, machine learning, deep learning

1. Introduction
Natural Language Processing has been revolutionized by transformer architectures. Models like BERT, GPT, and T5 have achieved state-of-the-art performance on numerous NLP benchmarks. This research investigates practical applications of these models in real-world scenarios.

2. Related Work
Previous work in NLP relied heavily on recurrent neural networks and convolutional approaches. The transformer architecture, introduced by Vaswani et al., changed the landscape with its attention mechanisms and parallelizable structure.

3. Methodology
We conducted experiments using three transformer models:
- BERT for text classification tasks
- GPT-3 for text generation and completion
- T5 for text-to-text transfer tasks

Each model was fine-tuned on domain-specific datasets and evaluated using standard metrics including accuracy, precision, recall, and F1-score.

4. Results
Our findings demonstrate consistent improvements across evaluation metrics:
- Text classification: 15% accuracy improvement over baseline
- Sentiment analysis: 12% F1-score increase
- Machine translation: 8% BLEU score improvement

5. Conclusion
Transformer-based models represent a significant advancement in NLP capabilities. Future work will explore few-shot learning and cross-lingual applications.
        """,

        "financial_quarterly_report_content.txt": """
TechCorp Industries - Q4 2024 Financial Report

Executive Summary
Q4 2024 delivered exceptional financial performance with revenue of $125.6M, representing 12% growth year-over-year. Net income increased to $18.4M, up from $17.0M in Q4 2023.

Key Financial Metrics:
• Revenue: $125.6M (+12.0% YoY)
• Gross Profit: $53.1M (42.3% margin)
• Operating Income: $19.1M (15.2% margin)
• Net Income: $18.4M (+8.2% YoY)
• EBITDA: $22.8M (+11.2% YoY)
• Earnings per Share: $2.15

Business Performance Highlights:
Strong performance was driven by continued growth in our cloud services division, which increased revenue by 25% year-over-year. Successful product launches in mobile analytics and AI-powered business intelligence contributed significantly to quarterly results.

Segment Performance:
• Cloud Services: $67.2M revenue (+25% YoY)
• Software Licensing: $34.1M revenue (+5% YoY)
• Professional Services: $24.3M revenue (+8% YoY)

Geographic Revenue Distribution:
• North America: 65% of total revenue
• Europe: 25% of total revenue
• Asia-Pacific: 10% of total revenue

2025 Outlook:
We project continued strong growth with full-year 2025 revenue targets of $550M, representing 10-15% growth. Key investment areas include artificial intelligence, cloud infrastructure expansion, and international market development.

Risk Factors:
Market competition, regulatory changes, and economic conditions may impact future performance. We maintain strong cash reserves and diversified revenue streams to mitigate risks.
        """,

        "software_user_manual_content.txt": """
DataAnalyzer Pro User Manual - Version 3.2

Table of Contents:
1. Installation and System Requirements
2. Getting Started Guide
3. User Interface Overview
4. Data Import and Export
5. Analysis Features and Tools
6. Visualization and Reporting
7. Advanced Features
8. Troubleshooting Guide
9. Technical Support

Chapter 1: Installation and System Requirements

DataAnalyzer Pro is a comprehensive business intelligence and data analysis software designed for professionals working with large datasets and complex analytical tasks.

System Requirements:
• Operating System: Windows 10/11 or macOS 10.15+
• RAM: 8GB minimum, 16GB recommended
• Storage: 2GB free disk space for installation
• Processor: Intel i5 or equivalent AMD processor
• Internet connection required for activation and updates

Installation Process:
1. Download the installer from the official website
2. Run the installer with administrator privileges
3. Accept the license agreement and terms of use
4. Choose installation directory (default recommended)
5. Enter your license key when prompted
6. Complete installation and restart computer
7. Launch application from desktop shortcut

Chapter 2: Getting Started

Upon first launch, DataAnalyzer Pro will guide you through workspace configuration:
• Create or import user profile
• Configure data source connections
• Set up default preferences and settings
• Initialize sample datasets for learning

The main interface consists of:
• Menu bar with File, Edit, View, Tools, and Help options
• Toolbar with quick access to common functions
• Data panel for viewing imported datasets
• Analysis workspace for creating visualizations
• Results panel for viewing output and reports

Chapter 3: Core Features

Data Import Capabilities:
• CSV, Excel, JSON, XML file formats
• Database connections (SQL Server, MySQL, PostgreSQL)
• API integrations for real-time data
• Cloud storage connections (AWS S3, Google Drive)

Analysis Tools:
• Statistical analysis and hypothesis testing
• Regression and classification models
• Time series analysis and forecasting
• Data clustering and segmentation
• Custom formula and calculation engine

For technical support, visit our website or contact support@dataanalyzer.com
        """
    }

    # Write the content files
    for filename, content in pdf_contents.items():
        (base_dir / filename).write_text(content.strip())

    print("✓ Created 3 PDF-content text files as alternatives")


def main():
    """Create proper PDF files or alternatives."""
    base_dir = Path("test_input")

    print("Attempting to create real PDF files...")
    if not create_real_pdfs(base_dir):
        print("Creating alternative PDF-content files...")
        create_alternative_pdfs(base_dir)

    print(f"\nUpdated test files in: {base_dir}")
    print(f"Total files now: {len(list(base_dir.glob('*')))}")


if __name__ == "__main__":
    main()