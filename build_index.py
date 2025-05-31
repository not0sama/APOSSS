#!/usr/bin/env python3
"""
Pre-indexing script for APOSSS - Build FAISS index for all documents
"""
import logging
import os
import json
import time
from datetime import datetime
from modules.database_manager import DatabaseManager
from modules.embedding_ranker import EmbeddingRanker
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentIndexBuilder:
    """Build and manage pre-computed document embeddings index"""
    
    def __init__(self, cache_dir: str = 'production_index_cache'):
        self.db_manager = DatabaseManager()
        self.cache_dir = cache_dir
        self.embedding_ranker = EmbeddingRanker(cache_dir=cache_dir)
        self.progress_file = os.path.join(cache_dir, 'indexing_progress.json')
        
    def save_progress(self, progress_data: Dict[str, Any]):
        """Save indexing progress to resume later if interrupted"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def load_progress(self) -> Dict[str, Any]:
        """Load previous indexing progress"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
        return {}
        
    def fetch_all_documents(self, resume: bool = True) -> List[Dict[str, Any]]:
        """Fetch all documents from all databases for indexing"""
        all_documents = []
        progress = self.load_progress() if resume else {}
        processed_collections = progress.get('processed_collections', [])
        
        # Database and collection mappings
        db_collections = {
            'academic_library': ['books', 'journals', 'projects'],
            'experts_system': ['experts', 'certificates'], 
            'research_papers': ['articles', 'conferences', 'theses'],
            'laboratories': ['equipments', 'materials']
        }
        
        total_collections = sum(len(cols) for cols in db_collections.values())
        current_collection = 0
        
        logger.info(f"Fetching documents from {total_collections} collections...")
        if resume and processed_collections:
            logger.info(f"Resuming from previous run, skipping {len(processed_collections)} completed collections")
        
        for db_name, collections in db_collections.items():
            try:
                database = self.db_manager.get_database(db_name)
                if database is None:
                    logger.warning(f"Could not connect to {db_name}")
                    continue
                    
                for collection_name in collections:
                    current_collection += 1
                    collection_key = f"{db_name}.{collection_name}"
                    
                    # Skip if already processed in resume mode
                    if resume and collection_key in processed_collections:
                        logger.info(f"[{current_collection}/{total_collections}] Skipping {collection_key} (already processed)")
                        continue
                    
                    try:
                        logger.info(f"[{current_collection}/{total_collections}] Fetching from {collection_key}...")
                        collection = database[collection_name]
                        
                        # Get total count for progress tracking
                        total_docs = collection.count_documents({})
                        logger.info(f"Found {total_docs} documents in {collection_key}")
                        
                        if total_docs == 0:
                            processed_collections.append(collection_key)
                            continue
                        
                        # Fetch documents in batches to avoid memory issues
                        batch_size = 1000
                        fetched_count = 0
                        
                        for skip in range(0, total_docs, batch_size):
                            batch_docs = list(collection.find({}).skip(skip).limit(batch_size))
                            
                            # Standardize document format for indexing
                            for doc in batch_docs:
                                try:
                                    standardized_doc = self._standardize_document(doc, db_name, collection_name)
                                    all_documents.append(standardized_doc)
                                    fetched_count += 1
                                except Exception as e:
                                    logger.warning(f"Error standardizing document {doc.get('_id', 'unknown')}: {e}")
                            
                            # Progress update
                            if fetched_count % 1000 == 0:
                                logger.info(f"  Processed {fetched_count}/{total_docs} documents from {collection_key}")
                        
                        logger.info(f"✅ Completed {collection_key}: {fetched_count} documents")
                        processed_collections.append(collection_key)
                        
                        # Save progress after each collection
                        self.save_progress({
                            'processed_collections': processed_collections,
                            'total_documents': len(all_documents),
                            'last_updated': datetime.now().isoformat()
                        })
                        
                    except Exception as e:
                        logger.error(f"Error fetching from {collection_key}: {str(e)}")
                        
            except Exception as e:
                logger.error(f"Error connecting to {db_name}: {str(e)}")
        
        logger.info(f"📊 Total documents fetched: {len(all_documents)}")
        return all_documents
    
    def _standardize_document(self, doc: Dict[str, Any], db_name: str, collection_name: str) -> Dict[str, Any]:
        """Standardize document format for consistent indexing"""
        # Convert ObjectId to string
        doc_id = str(doc.get('_id', ''))
        
        # Extract common fields with fallbacks
        title = (doc.get('title', '') or 
                doc.get('name', '') or 
                doc.get('equipment_name', '') or 
                doc.get('material_name', '') or 
                'Untitled')
        
        description = (doc.get('description', '') or 
                      doc.get('abstract', '') or 
                      doc.get('summary', '') or 
                      doc.get('specifications', '') or 
                      '')
        
        author = (doc.get('author', '') or 
                 doc.get('student_name', '') or 
                 doc.get('supervisor', '') or 
                 doc.get('editor', '') or
                 '')
        
        # Determine document type
        doc_type = collection_name.rstrip('s')  # Remove plural 's'
        
        # Extract keywords
        keywords = doc.get('keywords', [])
        if isinstance(keywords, str):
            keywords = [keywords]
        elif not isinstance(keywords, list):
            keywords = []
        
        return {
            'id': doc_id,
            'title': title,
            'description': description,
            'author': author,
            'type': doc_type,
            'database': db_name,
            'collection': collection_name,
            'metadata': {
                'keywords': keywords,
                'category': doc.get('category', ''),
                'publication_date': str(doc.get('publication_date', '')),
                'language': doc.get('language', ''),
                'status': doc.get('status', ''),
                'institution': doc.get('institution', '') or doc.get('university', ''),
                'department': doc.get('department', '')
            }
        }
    
    def build_full_index(self, resume: bool = True):
        """Build complete FAISS index for all documents"""
        start_time = time.time()
        
        logger.info("=" * 80)
        logger.info("🚀 APOSSS Pre-indexing System - Building Full Document Index")
        logger.info("=" * 80)
        
        try:
            # Step 1: Fetch all documents
            logger.info("📥 Step 1: Fetching all documents from databases...")
            all_documents = self.fetch_all_documents(resume=resume)
            
            if not all_documents:
                logger.error("❌ No documents found to index!")
                return False
            
            # Save document summary
            self.save_progress({
                'total_documents': len(all_documents),
                'fetch_completed': True,
                'fetch_time': time.time() - start_time,
                'last_updated': datetime.now().isoformat()
            })
            
            # Step 2: Build embeddings index
            logger.info(f"🧠 Step 2: Building FAISS embeddings index for {len(all_documents)} documents...")
            embedding_start = time.time()
            
            self.embedding_ranker.build_document_index(all_documents, batch_size=100)
            embedding_time = time.time() - embedding_start
            
            # Step 3: Verify index
            logger.info("✅ Step 3: Verifying index...")
            stats = self.embedding_ranker.get_cache_stats()
            logger.info("📊 Index Statistics:")
            logger.info(f"  • Total vectors: {stats['total_vectors']:,}")
            logger.info(f"  • Cache size: {stats['cache_size']:,}")
            logger.info(f"  • Model: {stats['model_name']}")
            logger.info(f"  • Dimensions: {stats['embedding_dimension']}")
            logger.info(f"  • Cache directory: {stats['cache_directory']}")
            
            # Step 4: Test search performance
            logger.info("🔍 Step 4: Testing semantic search performance...")
            test_queries = [
                "machine learning medical diagnosis",
                "renewable energy solar panels",
                "artificial intelligence algorithms",
                "laboratory equipment microscope",
                "research methodology statistics"
            ]
            
            search_times = []
            for query in test_queries:
                search_start = time.time()
                test_results = self.embedding_ranker.search_similar_documents(query, k=10)
                search_time = time.time() - search_start
                search_times.append(search_time)
                
                logger.info(f"  Query: '{query}' -> {len(test_results)} results in {search_time:.3f}s")
                if test_results:
                    top_result = test_results[0]
                    logger.info(f"    Top: {top_result.get('title', 'Untitled')} (Score: {top_result.get('similarity_score', 0):.4f})")
            
            avg_search_time = sum(search_times) / len(search_times)
            total_time = time.time() - start_time
            
            # Final summary
            logger.info("=" * 80)
            logger.info("🎉 PRE-INDEXING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"📊 Summary:")
            logger.info(f"  • Total documents indexed: {len(all_documents):,}")
            logger.info(f"  • Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
            logger.info(f"  • Embedding time: {embedding_time:.1f} seconds")
            logger.info(f"  • Average search time: {avg_search_time:.3f} seconds")
            logger.info(f"  • Index size: {stats['total_vectors']:,} vectors")
            logger.info("=" * 80)
            
            # Save final progress
            self.save_progress({
                'completed': True,
                'total_documents': len(all_documents),
                'total_time': total_time,
                'embedding_time': embedding_time,
                'avg_search_time': avg_search_time,
                'completion_date': datetime.now().isoformat(),
                'index_stats': stats
            })
            
            # Clean up progress file
            try:
                os.remove(self.progress_file)
            except:
                pass
            
            return True
            
        except KeyboardInterrupt:
            logger.info("\n⏸️  Indexing interrupted by user. Progress saved for resuming later.")
            return False
        except Exception as e:
            logger.error(f"❌ Error building index: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        stats = self.embedding_ranker.get_cache_stats()
        
        # Add additional metadata if available
        progress = self.load_progress()
        if progress:
            stats.update(progress)
        
        return stats
    
    def search_prebuilt_index(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """Search the pre-built index directly"""
        return self.embedding_ranker.search_similar_documents(query, k)

def main():
    """Main function to build the index"""
    print("🚀 APOSSS Document Pre-indexing System")
    print("=" * 60)
    
    # Check if resuming
    builder = DocumentIndexBuilder()
    progress = builder.load_progress()
    
    if progress and not progress.get('completed', False):
        print(f"📋 Previous indexing session found:")
        print(f"   • Total documents: {progress.get('total_documents', 'Unknown')}")
        print(f"   • Last updated: {progress.get('last_updated', 'Unknown')}")
        print(f"   • Completed collections: {len(progress.get('processed_collections', []))}")
        
        resume = input("\n🔄 Resume from previous session? (y/n): ").lower().strip() == 'y'
    else:
        resume = False
    
    # Build the complete index
    success = builder.build_full_index(resume=resume)
    
    if success:
        print("\n🎉 Pre-indexing completed successfully!")
        print("\n✨ Benefits now available:")
        print("  ✅ Lightning-fast semantic search")
        print("  ✅ Search across entire document corpus")
        print("  ✅ Superior semantic matching")
        print("  ✅ Scalable to millions of documents")
        print("\n🔗 The index is now ready for production use!")
    else:
        print("\n⚠️  Pre-indexing incomplete.")
        print("   Run the script again to resume from where it left off.")

if __name__ == "__main__":
    main() 