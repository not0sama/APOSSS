#!/usr/bin/env python3
"""
Test script for funding database integration
Tests the complete funding search functionality including institutions and research projects
"""

import os
import sys
import json
import asyncio
from datetime import datetime

# Add the modules directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from database_manager import DatabaseManager
from llm_processor import LLMProcessor
from query_processor import QueryProcessor
from search_engine import SearchEngine

def test_database_connections():
    """Test all database connections including the new funding database"""
    print("🔗 Testing database connections...")
    
    db_manager = DatabaseManager()
    status = db_manager.test_connections()
    
    for db_name, db_status in status.items():
        if db_status['connected']:
            print(f"✅ {db_name}: Connected")
            if 'collections' in db_status:
                for collection, count in db_status['collections'].items():
                    print(f"   📚 {collection}: {count} documents")
        else:
            print(f"❌ {db_name}: Failed - {db_status.get('error', 'Unknown error')}")
    
    return status

def test_funding_search():
    """Test the funding search functionality"""
    print("\n🔍 Testing funding search functionality...")
    
    # Initialize components
    db_manager = DatabaseManager()
    llm_processor = LLMProcessor()
    query_processor = QueryProcessor(llm_processor)
    search_engine = SearchEngine(db_manager)
    
    # Test queries related to funding
    test_queries = [
        "water resource management",
        "artificial intelligence",
        "environmental engineering",
        "machine learning",
        "renewable energy"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        
        try:
            # Process query with LLM
            processed_query = query_processor.process_query(query)
            
            # Search all databases including funding
            search_results = search_engine.search_all_databases(processed_query, hybrid_search=False)
            
            # Count results by database and type
            funding_results = [r for r in search_results.get('results', []) if r.get('database') == 'funding']
            institution_results = [r for r in funding_results if r.get('type') == 'funding_institution']
            project_results = [r for r in funding_results if r.get('type') == 'research_project']
            
            print(f"📊 Results summary:")
            print(f"   Total results: {search_results.get('total_results', 0)}")
            print(f"   Funding results: {len(funding_results)}")
            print(f"   - Institutions: {len(institution_results)}")
            print(f"   - Research projects: {len(project_results)}")
            
            # Display institution details
            if institution_results:
                print(f"\n🏛️ Found institutions:")
                for inst in institution_results[:3]:  # Show first 3
                    print(f"   - {inst.get('title', 'N/A')}")
                    if inst.get('funding_info'):
                        funding_info = inst['funding_info']
                        print(f"     💰 Total funding: ${funding_info.get('total_funding_amount', 0):,.2f}")
                        print(f"     📋 Projects funded: {funding_info.get('funding_count', 0)}")
            
            # Display project details
            if project_results:
                print(f"\n🔬 Found research projects:")
                for proj in project_results[:3]:  # Show first 3
                    print(f"   - {proj.get('title', 'N/A')}")
                    metadata = proj.get('metadata', {})
                    print(f"     🎯 Field: {metadata.get('field_category', 'N/A')}")
                    print(f"     📅 Status: {metadata.get('status', 'N/A')}")
                    
        except Exception as e:
            print(f"❌ Error testing query '{query}': {str(e)}")
    
    print("\n✅ Funding search test completed!")

def test_funding_api_endpoints():
    """Test the funding API endpoints"""
    print("\n🌐 Testing funding API endpoints...")
    
    # This would require the Flask app to be running
    # For now, we'll just verify the functions exist
    try:
        from app import get_institution_details, get_research_project_details
        print("✅ Funding API endpoints are properly defined in app.py")
    except ImportError as e:
        print(f"❌ Error importing funding API endpoints: {e}")

def main():
    """Main test function"""
    print("🚀 Starting Funding Database Integration Tests")
    print("=" * 60)
    
    # Test 1: Database connections
    db_status = test_database_connections()
    
    # Check if funding database is connected
    funding_connected = db_status.get('funding', {}).get('connected', False)
    
    if not funding_connected:
        print("\n⚠️  Funding database not connected. Please ensure:")
        print("   1. MongoDB is running")
        print("   2. Funding database exists with sample data")
        print("   3. Environment variables are properly set")
        return
    
    # Test 2: Funding search functionality
    test_funding_search()
    
    # Test 3: API endpoints
    test_funding_api_endpoints()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("\n📝 Next steps:")
    print("   1. Start the Flask app: python app.py")
    print("   2. Navigate to /results and search for funding-related terms")
    print("   3. Click on funding institutions to see detailed modals")
    print("   4. Verify that research projects show funding information")

if __name__ == "__main__":
    main() 