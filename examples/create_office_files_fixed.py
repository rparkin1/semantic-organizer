"""
Fixed script to create example office files for testing.
"""

from pathlib import Path


def create_mock_office_files(base_dir: Path):
    """Create mock office files with realistic content."""
    base_dir.mkdir(exist_ok=True)

    # Mock PDF files with content that would be extracted
    (base_dir / "ai_research_paper.pdf").write_text("""
    Advances in Natural Language Processing Using Transformer Architectures

    Abstract: This research paper presents recent developments in natural language
    processing using transformer-based models. We investigate the performance of
    BERT, GPT-3, and T5 models on various NLP tasks including text classification,
    sentiment analysis, and machine translation. Our experimental results demonstrate
    significant improvements over traditional approaches.

    Keywords: natural language processing, transformers, BERT, GPT, machine learning

    1. Introduction
    The field of natural language processing has been revolutionized by transformer
    architectures. These models have achieved state-of-the-art results on numerous
    benchmarks and have enabled breakthrough applications in conversational AI,
    content generation, and language understanding.

    2. Methodology
    We conducted experiments using three pre-trained transformer models:
    - BERT for text classification tasks
    - GPT-3 for text generation and completion
    - T5 for text-to-text transfer tasks

    3. Results
    Our findings show consistent improvements across all evaluation metrics,
    with accuracy increases of 10-15% over baseline methods.
    """)

    (base_dir / "financial_quarterly_report.pdf").write_text("""
    Q4 2024 Financial Report - TechCorp Industries

    Executive Summary
    Q4 2024 delivered strong financial performance with revenue of $45.2M,
    representing 18% growth year-over-year. Net income increased to $8.7M,
    up from $7.1M in Q4 2023.

    Financial Highlights:
    - Revenue: $45.2M (+18% YoY)
    - Gross Profit: $27.1M (60% margin)
    - Operating Income: $10.3M
    - Net Income: $8.7M (+23% YoY)
    - Earnings per Share: $1.42

    Business Overview:
    Strong performance was driven by continued growth in our cloud services
    division and successful product launches in the mobile sector. International
    expansion contributed 35% of total revenue.

    2025 Outlook:
    We project continued growth with revenue targets of $200M for full year 2025,
    driven by AI product initiatives and market expansion.
    """)

    (base_dir / "software_user_manual.pdf").write_text("""
    DataAnalyzer Pro - User Manual Version 3.2

    Table of Contents:
    1. Installation and Setup
    2. Getting Started Guide
    3. Data Import and Export
    4. Analysis Features
    5. Visualization Tools
    6. Reporting Functions
    7. Troubleshooting

    Chapter 1: Installation and Setup
    DataAnalyzer Pro is a comprehensive data analysis software designed for
    business intelligence and statistical analysis. This manual provides
    step-by-step instructions for installation, configuration, and usage.

    System Requirements:
    - Windows 10/11 or macOS 10.15+
    - 8GB RAM minimum, 16GB recommended
    - 2GB free disk space
    - Internet connection for activation

    Installation Steps:
    1. Download installer from company website
    2. Run installer with administrator privileges
    3. Follow setup wizard instructions
    4. Enter license key when prompted
    5. Complete initial configuration

    Chapter 2: Getting Started
    Upon first launch, the software will guide you through workspace setup
    and data source configuration. The main interface consists of menu bar,
    toolbar, data panel, and results area.
    """)

    # Mock Excel files
    (base_dir / "quarterly_sales_report.xlsx").write_text("""
    EXCEL: Quarterly Sales Report Q4 2024

    Sheet: Sales Data
    Product Category | Q1 Sales | Q2 Sales | Q3 Sales | Q4 Sales | Total
    Electronics      | 125,000  | 142,000  | 156,000  | 189,000  | 612,000
    Home & Garden    | 89,000   | 98,000   | 145,000  | 167,000  | 499,000
    Clothing         | 67,000   | 78,000   | 82,000   | 95,000   | 322,000
    Sports Equipment | 45,000   | 52,000   | 78,000   | 85,000   | 260,000
    Books & Media    | 34,000   | 38,000   | 41,000   | 44,000   | 157,000

    Sheet: Regional Performance
    Region    | Revenue  | Growth Rate | Market Share
    Northeast | $485,000 | +12%        | 28%
    Southeast | $456,000 | +15%        | 26%
    West      | $523,000 | +18%        | 30%
    Midwest   | $386,000 | +8%         | 16%

    Summary: Total annual revenue of $1.85M with 13% overall growth
    Best performing category: Electronics
    Fastest growing region: West Coast
    """)

    (base_dir / "annual_budget_analysis.xlsx").write_text("""
    EXCEL: Annual Budget Analysis 2024

    Sheet: Department Budgets
    Department      | Allocated | Actual   | Variance | Percentage
    Marketing       | $150,000  | $142,500 | +$7,500  | 95%
    R&D            | $200,000  | $218,000 | -$18,000 | 109%
    Operations     | $300,000  | $295,000 | +$5,000  | 98%
    HR             | $120,000  | $115,000 | +$5,000  | 96%
    IT             | $180,000  | $175,000 | +$5,000  | 97%
    Administration | $80,000   | $78,000  | +$2,000  | 98%

    Sheet: Expense Categories
    Category        | Amount   | Percentage of Total
    Salaries        | $650,000 | 62%
    Technology      | $175,000 | 17%
    Marketing       | $85,000  | 8%
    Facilities      | $65,000  | 6%
    Training        | $35,000  | 3%
    Other           | $40,000  | 4%

    Total Budget: $1,050,000
    Total Spent: $1,023,500
    Budget Utilization: 97.5%
    """)

    (base_dir / "employee_database.xlsx").write_text("""
    EXCEL: Employee Database - HR Records

    Sheet: Employee Information
    ID    | Name            | Department  | Position          | Salary  | Start Date
    E001  | Alice Johnson   | Engineering | Senior Developer  | $95,000 | 2020-03-15
    E002  | Bob Smith       | Marketing   | Marketing Manager | $75,000 | 2019-07-01
    E003  | Carol Davis     | HR          | HR Specialist    | $55,000 | 2021-01-10
    E004  | David Wilson    | Engineering | DevOps Engineer   | $85,000 | 2020-11-20
    E005  | Emma Brown      | Finance     | Financial Analyst | $65,000 | 2021-06-01
    E006  | Frank Miller    | Sales       | Sales Manager     | $70,000 | 2018-09-15
    E007  | Grace Taylor    | Engineering | Frontend Developer| $78,000 | 2022-02-01

    Sheet: Department Summary
    Department  | Headcount | Avg Salary | Total Payroll
    Engineering | 3         | $86,000    | $258,000
    Marketing   | 1         | $75,000    | $75,000
    HR          | 1         | $55,000    | $55,000
    Finance     | 1         | $65,000    | $65,000
    Sales       | 1         | $70,000    | $70,000

    Company Total: 7 employees, $523,000 annual payroll
    """)

    # Mock PowerPoint files
    (base_dir / "marketing_strategy_presentation.pptx").write_text("""
    POWERPOINT: Digital Marketing Strategy 2025

    Slide 1: Title Slide
    Digital Marketing Strategy 2025
    Driving Growth Through Innovation and Customer Engagement
    Presented by: Marketing Team

    Slide 2: Executive Summary
    • 25% increase in digital presence planned
    • Multi-channel approach: social media, email, content marketing
    • Budget allocation: $500K for digital initiatives
    • Expected ROI: 300% within 18 months
    • Focus on customer acquisition and retention

    Slide 3: Market Analysis
    • Target demographics: Millennials and Gen Z (ages 25-40)
    • Primary channels: Instagram, LinkedIn, TikTok, YouTube
    • Competitor analysis shows opportunity in mobile marketing
    • Market size: $2.8B addressable market
    • Growth projections: 15% annual increase

    Slide 4: Strategy Components
    • Social Media Marketing: Engaging content creation
    • Search Engine Optimization: Improved organic visibility
    • Content Marketing: Educational and entertaining content
    • Email Campaigns: Personalized customer communications
    • Influencer Partnerships: Authentic brand endorsements

    Slide 5: Implementation Timeline
    • Q1: Team expansion and platform setup
    • Q2: Content creation and campaign launch
    • Q3: Optimization and scaling
    • Q4: Performance analysis and planning for next year

    Slide 6: Expected Results
    • Website traffic increase: +40%
    • Social media followers: +60%
    • Conversion rate improvement: +25%
    • Brand awareness growth: +50%
    • Customer acquisition cost reduction: -20%
    """)

    (base_dir / "ml_implementation_slides.pptx").write_text("""
    POWERPOINT: Machine Learning Implementation Project

    Slide 1: Project Overview
    Machine Learning Implementation for Predictive Analytics
    Transforming Business Intelligence with AI
    Project Timeline: 6 months
    Budget: $2.5M

    Slide 2: Business Objectives
    • Improve decision-making with predictive insights
    • Automate routine data analysis tasks
    • Reduce operational costs by 20%
    • Enhance customer experience through personalization
    • Create competitive advantage in market

    Slide 3: Technical Architecture
    • Data Collection Layer: APIs, databases, streaming data
    • Data Processing Pipeline: ETL processes, data cleaning
    • Machine Learning Platform: Model training and deployment
    • API Service Layer: RESTful APIs for model inference
    • Frontend Dashboard: User interface for insights

    Slide 4: Technology Stack
    • Python & Scikit-learn for machine learning models
    • Apache Kafka for real-time data streaming
    • PostgreSQL for structured data storage
    • Docker & Kubernetes for containerization
    • React & D3.js for interactive dashboards

    Slide 5: Model Development Process
    • Data exploration and feature engineering
    • Model selection and hyperparameter tuning
    • Cross-validation and performance evaluation
    • A/B testing for model comparison
    • Continuous monitoring and model updates

    Slide 6: Success Metrics
    • Model accuracy: >85% for classification tasks
    • Response time: <200ms for API calls
    • System uptime: 99.9% availability
    • User adoption: 80% of target users
    • Business impact: 15% improvement in KPIs
    """)

    (base_dir / "safety_training_presentation.pptx").write_text("""
    POWERPOINT: Workplace Safety Training Program

    Slide 1: Safety First
    Comprehensive Workplace Safety Training
    Protecting Our Team, Protecting Our Future
    Mandatory Training for All Employees

    Slide 2: Safety Statistics
    • Workplace injuries cost companies $170B annually
    • 95% of serious injuries are preventable
    • Proper training reduces incidents by 60%
    • Every employee has the right to a safe workplace
    • Safety is everyone's responsibility

    Slide 3: Core Safety Principles
    • Always wear appropriate Personal Protective Equipment (PPE)
    • Report hazards and unsafe conditions immediately
    • Follow established safety procedures and protocols
    • Keep work areas clean and organized
    • Use equipment and tools properly and safely

    Slide 4: Emergency Procedures
    • Fire Emergency: RACE protocol (Rescue, Alarm, Confine, Evacuate)
    • Medical Emergency: Call 911, provide first aid if trained
    • Chemical Spill: Evacuate area, notify safety team
    • Power Outage: Remain calm, use emergency lighting
    • Evacuation Routes: Know all exits from your work area

    Slide 5: Hazard Recognition
    • Physical hazards: Slips, trips, falls, moving machinery
    • Chemical hazards: Toxic substances, flammable materials
    • Biological hazards: Infectious materials, allergens
    • Ergonomic hazards: Repetitive motions, heavy lifting
    • Environmental hazards: Temperature, noise, lighting

    Slide 6: Your Safety Responsibilities
    • Attend all required safety training sessions
    • Use PPE correctly and consistently
    • Report unsafe conditions or behaviors
    • Follow safety procedures without exception
    • Participate in safety meetings and drills
    • Ask questions when uncertain about safety protocols
    """)

    print(f"✅ Created comprehensive mock office files!")
    print(f"📁 PDF files: 3 (research paper, financial report, user manual)")
    print(f"📊 Excel files: 3 (sales report, budget analysis, employee database)")
    print(f"📽️ PowerPoint files: 3 (marketing strategy, ML implementation, safety training)")


def main():
    """Create all mock office files."""
    base_dir = Path("test_input")
    create_mock_office_files(base_dir)

    print(f"\n📋 Total files in test_input: {len(list(base_dir.glob('*')))}")
    print("\nTo test the semantic organizer with these files:")
    print("python -m semantic_organizer test_input test_output --dry-run --verbose")


if __name__ == "__main__":
    main()