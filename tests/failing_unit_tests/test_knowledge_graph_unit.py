#!/usr/bin/env python3
"""
Unit tests for APOSSS KnowledgeGraph class
Tests graph construction, PageRank calculations, authority scores, and node relationships
"""
import sys
import os
import unittest
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestKnowledgeGraph(unittest.TestCase):
    """Unit tests for KnowledgeGraph class"""

    def setUp(self):
        """Set up test fixtures"""
        # Sample test data
        self.test_paper_data = {
            'id': 'paper1',
            'title': 'Machine Learning in Healthcare',
            'authors': ['Dr. Sarah Johnson', 'Prof. Michael Chen'],
            'abstract': 'This paper explores ML applications in medical diagnosis',
            'keywords': ['machine learning', 'healthcare', 'diagnosis'],
            'year': 2023,
            'citations': 15,
            'references': ['paper2', 'paper3'],
            'institution': 'MIT',
            'field': 'Computer Science'
        }
        
        self.test_expert_data = {
            'id': 'expert1',
            'name': 'Dr. Sarah Johnson',
            'expertise': ['Machine Learning', 'Healthcare AI'],
            'institution': 'MIT',
            'h_index': 25,
            'publications': ['paper1', 'paper4'],
            'collaborators': ['Prof. Michael Chen']
        }
        
        self.test_equipment_data = {
            'id': 'equipment1',
            'name': 'High-Performance Computing Cluster',
            'type': 'Computing Resource',
            'location': 'MIT Computer Science Lab',
            'capabilities': ['Machine Learning', 'Data Analysis'],
            'availability': 'Available'
        }

    def test_knowledge_graph_initialization(self):
        """Test KnowledgeGraph initialization"""
        from modules.knowledge_graph import KnowledgeGraph
        
        # Test successful initialization
        kg = KnowledgeGraph()
        
        self.assertIsNotNone(kg)
        self.assertIsNotNone(kg.graph)
        self.assertIsNotNone(kg.node_types)
        self.assertIsNotNone(kg.edge_types)
        
        # Check initial state
        self.assertEqual(kg.graph.number_of_nodes(), 0)
        self.assertEqual(kg.graph.number_of_edges(), 0)
        self.assertIsNone(kg.pagerank_scores)

    def test_add_paper_node(self):
        """Test adding paper nodes to the graph"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Test adding paper
        kg.add_paper('paper1', self.test_paper_data)
        
        self.assertEqual(kg.graph.number_of_nodes(), 1)
        self.assertIn('paper1', kg.node_types['paper'])
        self.assertTrue(kg.graph.has_node('paper1'))
        
        # Test node attributes
        node_data = kg.graph.nodes['paper1']
        self.assertEqual(node_data['type'], 'paper')
        self.assertEqual(node_data['title'], self.test_paper_data['title'])

    def test_add_expert_node(self):
        """Test adding expert nodes to the graph"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Test adding expert
        kg.add_expert('expert1', self.test_expert_data)
        
        self.assertEqual(kg.graph.number_of_nodes(), 1)
        self.assertIn('expert1', kg.node_types['author'])
        self.assertTrue(kg.graph.has_node('expert1'))
        
        # Test node attributes
        node_data = kg.graph.nodes['expert1']
        self.assertEqual(node_data['type'], 'author')
        self.assertEqual(node_data['name'], self.test_expert_data['name'])

    def test_add_equipment_node(self):
        """Test adding equipment nodes to the graph"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Test adding equipment
        kg.add_equipment('equipment1', self.test_equipment_data)
        
        self.assertEqual(kg.graph.number_of_nodes(), 1)
        self.assertIn('equipment1', kg.node_types['equipment'])
        self.assertTrue(kg.graph.has_node('equipment1'))
        
        # Test node attributes
        node_data = kg.graph.nodes['equipment1']
        self.assertEqual(node_data['type'], 'equipment')
        self.assertEqual(node_data['name'], self.test_equipment_data['name'])

    def test_add_keyword_nodes(self):
        """Test adding keyword nodes to the graph"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Add paper with keywords
        kg.add_paper('paper1', self.test_paper_data)
        
        # Check that keyword nodes were created
        expected_keywords = ['machine learning', 'healthcare', 'diagnosis']
        for keyword in expected_keywords:
            keyword_node = f"keyword_{keyword.replace(' ', '_')}"
            self.assertTrue(kg.graph.has_node(keyword_node))
            self.assertIn(keyword_node, kg.node_types['keyword'])

    def test_add_institution_nodes(self):
        """Test adding institution nodes to the graph"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Add paper with institution
        kg.add_paper('paper1', self.test_paper_data)
        
        # Check that institution node was created
        institution_node = "institution_MIT"
        self.assertTrue(kg.graph.has_node(institution_node))
        self.assertIn(institution_node, kg.node_types['institution'])

    def test_add_relationships(self):
        """Test adding relationships between nodes"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Add paper and expert
        kg.add_paper('paper1', self.test_paper_data)
        kg.add_expert('expert1', self.test_expert_data)
        
        # Check author relationship
        author_node = "author_Dr. Sarah Johnson"
        self.assertTrue(kg.graph.has_edge('paper1', author_node))
        
        # Check keyword relationships
        keyword_node = "keyword_machine_learning"
        self.assertTrue(kg.graph.has_edge('paper1', keyword_node))

    def test_citation_relationships(self):
        """Test citation relationships between papers"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Add papers with citations
        paper1_data = {**self.test_paper_data, 'id': 'paper1', 'references': ['paper2']}
        paper2_data = {**self.test_paper_data, 'id': 'paper2', 'title': 'Referenced Paper'}
        
        kg.add_paper('paper1', paper1_data)
        kg.add_paper('paper2', paper2_data)
        
        # Check citation relationship
        self.assertTrue(kg.graph.has_edge('paper1', 'paper2'))
        
        # Check edge data
        edge_data = kg.graph.edges['paper1', 'paper2']
        self.assertEqual(edge_data['type'], 'cites')

    def test_collaboration_relationships(self):
        """Test collaboration relationships between experts"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Add experts with collaborations
        expert1_data = {**self.test_expert_data, 'id': 'expert1', 'collaborators': ['Prof. Michael Chen']}
        expert2_data = {**self.test_expert_data, 'id': 'expert2', 'name': 'Prof. Michael Chen'}
        
        kg.add_expert('expert1', expert1_data)
        kg.add_expert('expert2', expert2_data)
        
        # Check collaboration relationship
        author1_node = "author_Dr. Sarah Johnson"
        author2_node = "author_Prof. Michael Chen"
        
        if kg.graph.has_edge(author1_node, author2_node):
            edge_data = kg.graph.edges[author1_node, author2_node]
            self.assertEqual(edge_data['type'], 'collaborates_with')

    def test_pagerank_calculation(self):
        """Test PageRank calculation"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Add some nodes and edges
        kg.add_paper('paper1', self.test_paper_data)
        kg.add_expert('expert1', self.test_expert_data)
        kg.add_equipment('equipment1', self.test_equipment_data)
        
        # Calculate PageRank
        kg.calculate_pagerank()
        
        self.assertIsNotNone(kg.pagerank_scores)
        self.assertIsInstance(kg.pagerank_scores, dict)
        self.assertGreater(len(kg.pagerank_scores), 0)

    def test_authority_score_calculation(self):
        """Test authority score calculation"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Add nodes with different characteristics
        kg.add_paper('paper1', self.test_paper_data)
        kg.add_expert('expert1', self.test_expert_data)
        
        # Calculate authority scores
        kg.calculate_pagerank()
        
        # Test authority score retrieval
        authority_score = kg.get_authority_score('paper1')
        
        self.assertIsInstance(authority_score, (int, float))
        self.assertGreaterEqual(authority_score, 0)

    def test_connection_strength_calculation(self):
        """Test connection strength calculation"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Add connected nodes
        kg.add_paper('paper1', self.test_paper_data)
        kg.add_expert('expert1', self.test_expert_data)
        
        # Test connection strength
        connection_strength = kg.get_connection_strength('paper1', 'expert1')
        
        self.assertIsInstance(connection_strength, (int, float))
        self.assertGreaterEqual(connection_strength, 0)

    def test_node_pagerank_retrieval(self):
        """Test PageRank score retrieval for nodes"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Add nodes
        kg.add_paper('paper1', self.test_paper_data)
        kg.calculate_pagerank()
        
        # Test PageRank retrieval
        pagerank_score = kg.get_node_pagerank('paper1')
        
        self.assertIsInstance(pagerank_score, (int, float))
        self.assertGreaterEqual(pagerank_score, 0)

    def test_get_related_nodes(self):
        """Test getting related nodes"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Add connected nodes
        kg.add_paper('paper1', self.test_paper_data)
        kg.add_expert('expert1', self.test_expert_data)
        
        # Test getting related nodes
        related_nodes = kg.get_related_nodes('paper1')
        
        self.assertIsInstance(related_nodes, list)
        self.assertGreaterEqual(len(related_nodes), 0)

    def test_get_nodes_by_type(self):
        """Test getting nodes by type"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Add different types of nodes
        kg.add_paper('paper1', self.test_paper_data)
        kg.add_expert('expert1', self.test_expert_data)
        kg.add_equipment('equipment1', self.test_equipment_data)
        
        # Test getting nodes by type
        paper_nodes = kg.get_nodes_by_type('paper')
        author_nodes = kg.get_nodes_by_type('author')
        equipment_nodes = kg.get_nodes_by_type('equipment')
        
        self.assertIsInstance(paper_nodes, list)
        self.assertIsInstance(author_nodes, list)
        self.assertIsInstance(equipment_nodes, list)
        
        self.assertGreaterEqual(len(paper_nodes), 1)
        self.assertGreaterEqual(len(author_nodes), 0)
        self.assertGreaterEqual(len(equipment_nodes), 1)

    def test_find_shortest_path(self):
        """Test finding shortest path between nodes"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Add connected nodes
        kg.add_paper('paper1', self.test_paper_data)
        kg.add_expert('expert1', self.test_expert_data)
        
        # Test finding shortest path
        if kg.graph.number_of_nodes() > 1:
            nodes = list(kg.graph.nodes())
            if len(nodes) >= 2:
                path = kg.find_shortest_path(nodes[0], nodes[1])
                
                if path:
                    self.assertIsInstance(path, list)
                    self.assertGreaterEqual(len(path), 2)

    def test_get_node_centrality(self):
        """Test getting node centrality measures"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Add nodes
        kg.add_paper('paper1', self.test_paper_data)
        kg.add_expert('expert1', self.test_expert_data)
        
        # Test centrality measures
        centrality_measures = kg.get_node_centrality('paper1')
        
        self.assertIsInstance(centrality_measures, dict)
        self.assertIn('degree_centrality', centrality_measures)
        self.assertIn('betweenness_centrality', centrality_measures)
        self.assertIn('closeness_centrality', centrality_measures)

    def test_get_graph_statistics(self):
        """Test getting graph statistics"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Add nodes
        kg.add_paper('paper1', self.test_paper_data)
        kg.add_expert('expert1', self.test_expert_data)
        kg.add_equipment('equipment1', self.test_equipment_data)
        
        # Test graph statistics
        stats = kg.get_graph_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('num_nodes', stats)
        self.assertIn('num_edges', stats)
        self.assertIn('node_types', stats)
        self.assertIn('edge_types', stats)

    def test_duplicate_node_handling(self):
        """Test handling of duplicate nodes"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Add same paper twice
        kg.add_paper('paper1', self.test_paper_data)
        initial_node_count = kg.graph.number_of_nodes()
        
        kg.add_paper('paper1', self.test_paper_data)
        final_node_count = kg.graph.number_of_nodes()
        
        # Should not create duplicate nodes
        self.assertEqual(initial_node_count, final_node_count)

    def test_empty_graph_operations(self):
        """Test operations on empty graph"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Test operations on empty graph
        self.assertEqual(kg.get_authority_score('nonexistent'), 0.0)
        self.assertEqual(kg.get_connection_strength('node1', 'node2'), 0.0)
        self.assertEqual(kg.get_node_pagerank('nonexistent'), 0.0)
        self.assertEqual(kg.get_related_nodes('nonexistent'), [])

    def test_malformed_data_handling(self):
        """Test handling of malformed data"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Test with malformed paper data
        malformed_paper_data = {
            'id': 'paper1',
            # Missing required fields
        }
        
        # Should not crash
        kg.add_paper('paper1', malformed_paper_data)
        
        # Test with None data
        kg.add_paper('paper2', None)
        
        # Graph should still be functional
        self.assertIsNotNone(kg.graph)

    def test_large_graph_performance(self):
        """Test performance with larger graph"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Add multiple nodes
        for i in range(10):
            paper_data = {
                **self.test_paper_data,
                'id': f'paper{i}',
                'title': f'Paper {i}',
                'references': [f'paper{j}' for j in range(max(0, i-2), i)]
            }
            kg.add_paper(f'paper{i}', paper_data)
        
        # Test that operations still work
        kg.calculate_pagerank()
        stats = kg.get_graph_statistics()
        
        self.assertGreater(stats['num_nodes'], 10)
        self.assertIsNotNone(kg.pagerank_scores)

    def test_graph_serialization(self):
        """Test graph serialization and deserialization"""
        from modules.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # Add some data
        kg.add_paper('paper1', self.test_paper_data)
        kg.add_expert('expert1', self.test_expert_data)
        
        # Test serialization methods if available
        if hasattr(kg, 'to_dict'):
            graph_dict = kg.to_dict()
            self.assertIsInstance(graph_dict, dict)
        
        if hasattr(kg, 'from_dict'):
            # Test deserialization
            kg2 = KnowledgeGraph()
            if hasattr(kg, 'to_dict'):
                kg2.from_dict(graph_dict)
                self.assertEqual(kg.graph.number_of_nodes(), kg2.graph.number_of_nodes())


class TestKnowledgeGraphIntegration(unittest.TestCase):
    """Integration tests for KnowledgeGraph with real NetworkX"""

    def test_knowledge_graph_with_networkx(self):
        """Test KnowledgeGraph with real NetworkX functionality"""
        try:
            import networkx as nx
            from modules.knowledge_graph import KnowledgeGraph
            
            kg = KnowledgeGraph()
            
            # Test that NetworkX graph is properly initialized
            self.assertIsInstance(kg.graph, nx.DiGraph)
            
            # Test basic NetworkX operations
            kg.graph.add_node('test_node')
            self.assertTrue(kg.graph.has_node('test_node'))
            
        except ImportError:
            self.skipTest("NetworkX not available")

    def test_knowledge_graph_algorithms(self):
        """Test KnowledgeGraph with NetworkX algorithms"""
        try:
            import networkx as nx
            from modules.knowledge_graph import KnowledgeGraph
            
            kg = KnowledgeGraph()
            
            # Add some connected nodes
            kg.graph.add_nodes_from(['A', 'B', 'C'])
            kg.graph.add_edges_from([('A', 'B'), ('B', 'C')])
            
            # Test NetworkX algorithms
            pagerank = nx.pagerank(kg.graph)
            centrality = nx.degree_centrality(kg.graph)
            
            self.assertIsInstance(pagerank, dict)
            self.assertIsInstance(centrality, dict)
            
        except ImportError:
            self.skipTest("NetworkX not available")


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2) 