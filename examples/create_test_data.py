"""
Script to create test data for demonstrating the semantic file organizer.
"""

import os
from pathlib import Path


def create_test_data(base_dir: Path):
    """
    Create a variety of test files to demonstrate the organizer.

    Args:
        base_dir: Directory to create test files in
    """
    base_dir.mkdir(exist_ok=True)

    # Research and academic files
    (base_dir / "machine_learning_survey.txt").write_text("""
    A Comprehensive Survey of Machine Learning Techniques

    This document provides an overview of various machine learning algorithms
    including supervised learning, unsupervised learning, and reinforcement learning.
    We discuss neural networks, decision trees, and support vector machines.
    """)

    (base_dir / "ai_ethics_paper.txt").write_text("""
    Ethical Considerations in Artificial Intelligence

    This paper examines the ethical implications of AI systems in society.
    We discuss bias, fairness, transparency, and accountability in AI algorithms.
    The research covers deep learning, natural language processing, and computer vision.
    """)

    (base_dir / "data_science_notes.txt").write_text("""
    Data Science Project Notes

    Notes from the customer analytics project using Python and pandas.
    Applied machine learning models for predictive analysis.
    Used scikit-learn for classification and regression tasks.
    """)

    # Financial documents
    (base_dir / "budget_2024.txt").write_text("""
    Annual Budget Report 2024

    Revenue: $150,000
    Expenses: $120,000
    Profit: $30,000

    Major expense categories:
    - Marketing: $25,000
    - Operations: $45,000
    - Development: $35,000
    - Administration: $15,000
    """)

    (base_dir / "investment_analysis.txt").write_text("""
    Investment Portfolio Analysis

    Stock performance review for Q4 2024:
    - Technology stocks: +12%
    - Healthcare: +8%
    - Energy: -3%
    - Financial services: +15%

    Recommended portfolio adjustments based on market trends.
    """)

    # Recipe and cooking files
    (base_dir / "chocolate_cake_recipe.txt").write_text("""
    Chocolate Cake Recipe

    Ingredients:
    - 2 cups flour
    - 1.5 cups sugar
    - 3/4 cup cocoa powder
    - 2 tsp baking soda
    - 1 cup milk
    - 1/2 cup vegetable oil
    - 2 eggs

    Instructions:
    1. Preheat oven to 350°F
    2. Mix dry ingredients
    3. Add wet ingredients and mix
    4. Bake for 30-35 minutes
    """)

    (base_dir / "pasta_carbonara.txt").write_text("""
    Pasta Carbonara Recipe

    Ingredients:
    - 400g spaghetti
    - 200g pancetta or bacon
    - 4 eggs
    - 100g Parmesan cheese
    - Black pepper
    - Salt

    A classic Italian pasta dish with creamy egg sauce.
    Cook the pasta, crisp the pancetta, mix with egg and cheese.
    """)

    # Travel and personal files
    (base_dir / "europe_travel_itinerary.txt").write_text("""
    European Travel Itinerary - Summer 2024

    Day 1-3: Paris, France
    - Visit Eiffel Tower, Louvre Museum
    - Stay at Hotel de la Paix

    Day 4-6: Rome, Italy
    - Colosseum, Vatican City
    - Food tour in Trastevere

    Day 7-9: Barcelona, Spain
    - Sagrada Familia, Park Güell
    - Beach time at Barceloneta
    """)

    (base_dir / "japan_trip_photos.txt").write_text("""
    Japan Trip Photo Collection

    Photos from our amazing trip to Japan:
    - Tokyo: Shibuya crossing, Tokyo Tower, Senso-ji Temple
    - Kyoto: Bamboo forest, Golden Pavilion, Geisha district
    - Osaka: Osaka Castle, street food tours
    - Mount Fuji: Sunrise hike, lake views

    Total: 847 photos and 23 videos
    """)

    # Technology and programming files
    (base_dir / "python_web_scraping.txt").write_text("""
    Python Web Scraping Tutorial

    Learn how to scrape websites using Python:
    - BeautifulSoup for HTML parsing
    - Requests for HTTP requests
    - Selenium for JavaScript-heavy sites
    - Scrapy framework for large projects

    Code examples for scraping news articles, product prices, and social media data.
    """)

    (base_dir / "react_component_guide.txt").write_text("""
    React Component Development Guide

    Best practices for building React components:
    - Functional vs class components
    - Hooks: useState, useEffect, useContext
    - Props and state management
    - Component lifecycle methods

    Includes examples of reusable UI components and testing strategies.
    """)

    # Health and fitness files
    (base_dir / "workout_routine.txt").write_text("""
    Weekly Workout Routine

    Monday - Chest and Triceps:
    - Bench press: 4 sets x 8-10 reps
    - Incline dumbbell press: 3 sets x 10-12 reps
    - Tricep dips: 3 sets x 12-15 reps

    Wednesday - Back and Biceps:
    - Pull-ups: 4 sets x 8-10 reps
    - Rows: 3 sets x 10-12 reps
    - Bicep curls: 3 sets x 12-15 reps

    Friday - Legs and Shoulders:
    - Squats: 4 sets x 8-10 reps
    - Shoulder press: 3 sets x 10-12 reps
    """)

    (base_dir / "nutrition_meal_plan.txt").write_text("""
    Weekly Nutrition and Meal Plan

    Daily macronutrient targets:
    - Protein: 150g (30%)
    - Carbohydrates: 200g (40%)
    - Fat: 67g (30%)
    - Total calories: 2000

    Meal ideas:
    - Breakfast: Oatmeal with berries and protein powder
    - Lunch: Grilled chicken with quinoa and vegetables
    - Dinner: Salmon with sweet potato and broccoli
    - Snacks: Greek yogurt, nuts, fruit
    """)

    # Create a folder with mixed content
    project_folder = base_dir / "website_project"
    project_folder.mkdir(exist_ok=True)

    (project_folder / "project_readme.txt").write_text("""
    E-commerce Website Project

    A full-stack e-commerce website built with React and Node.js.
    Features include user authentication, product catalog, shopping cart, and payment processing.
    Database: MongoDB
    Frontend: React, Redux, CSS3
    Backend: Node.js, Express, JWT authentication
    """)

    (project_folder / "database_schema.txt").write_text("""
    Database Schema Design

    Users table:
    - user_id (primary key)
    - username, email, password_hash
    - created_at, updated_at

    Products table:
    - product_id (primary key)
    - name, description, price, category
    - stock_quantity, image_url

    Orders table:
    - order_id (primary key)
    - user_id (foreign key), total_amount
    - status, created_at
    """)

    print(f"Created test data in: {base_dir}")
    print(f"Files created: {len(list(base_dir.rglob('*.txt')))}")


if __name__ == "__main__":
    test_dir = Path("test_input")
    create_test_data(test_dir)
    print("\nRun the organizer with:")
    print(f"python -m semantic_organizer {test_dir} test_output --dry-run --verbose")