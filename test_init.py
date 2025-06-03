#!/usr/bin/env python3
"""Test component initialization"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Testing component initialization...")

# Test each component individually
components = []

try:
    from modules.database_manager import DatabaseManager
    db_manager = DatabaseManager()
    print('✅ DatabaseManager initialized')
    components.append(('database_manager', True))
except Exception as e:
    print(f'❌ DatabaseManager failed: {e}')
    components.append(('database_manager', False))

try:
    from modules.llm_processor import LLMProcessor
    llm_processor = LLMProcessor()
    print('✅ LLMProcessor initialized')
    components.append(('llm_processor', True))
except Exception as e:
    print(f'❌ LLMProcessor failed: {e}')
    components.append(('llm_processor', False))

try:
    from modules.query_processor import QueryProcessor
    query_processor = QueryProcessor(llm_processor)
    print('✅ QueryProcessor initialized')
    components.append(('query_processor', True))
except Exception as e:
    print(f'❌ QueryProcessor failed: {e}')
    components.append(('query_processor', False))

try:
    from modules.search_engine import SearchEngine
    search_engine = SearchEngine(db_manager)
    print('✅ SearchEngine initialized')
    components.append(('search_engine', True))
except Exception as e:
    print(f'❌ SearchEngine failed: {e}')
    components.append(('search_engine', False))

try:
    from modules.ranking_engine import RankingEngine
    ranking_engine = RankingEngine()
    print('✅ RankingEngine initialized')
    components.append(('ranking_engine', True))
except Exception as e:
    print(f'❌ RankingEngine failed: {e}')
    components.append(('ranking_engine', False))

try:
    from modules.feedback_system import FeedbackSystem
    feedback_system = FeedbackSystem(db_manager)
    print('✅ FeedbackSystem initialized')
    components.append(('feedback_system', True))
except Exception as e:
    print(f'❌ FeedbackSystem failed: {e}')
    components.append(('feedback_system', False))

try:
    from modules.user_manager import UserManager
    user_manager = UserManager(db_manager)
    print('✅ UserManager initialized')
    components.append(('user_manager', True))
except Exception as e:
    print(f'❌ UserManager failed: {e}')
    components.append(('user_manager', False))

print("\nComponent initialization summary:")
for component, success in components:
    status = "✅" if success else "❌"
    print(f"{status} {component}")

# Check environment variables
print(f"\nEnvironment variables:")
print(f"GEMINI_API_KEY: {'SET' if os.getenv('GEMINI_API_KEY') else 'NOT SET'}")
print(f"MONGODB_URI_ACADEMIC_LIBRARY: {os.getenv('MONGODB_URI_ACADEMIC_LIBRARY', 'NOT SET')}") 