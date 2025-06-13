import networkx as nx
from typing import Dict, List, Any, Set, Tuple
import logging
from collections import defaultdict
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Knowledge Graph system for APOSSS that models relationships between:
    - Papers/Articles
    - Authors/Experts
    - Keywords/Topics
    - Equipment/Resources
    - Projects
    - Institutions
    """
    
    def __init__(self):
        """Initialize the knowledge graph"""
        self.graph = nx.DiGraph()  # Directed graph for relationships
        self.node_types = {
            'paper': set(),
            'author': set(),
            'keyword': set(),
            'equipment': set(),
            'project': set(),
            'institution': set()
        }
        self.edge_types = {
            'cites': defaultdict(int),
            'authored': defaultdict(int),
            'contains_keyword': defaultdict(int),
            'uses_equipment': defaultdict(int),
            'affiliated_with': defaultdict(int),
            'collaborates_with': defaultdict(int)
        }
        self.pagerank_scores = None
        self.last_update = None
    
    def add_paper(self, paper_id: str, metadata: Dict[str, Any]) -> None:
        """Add a paper and its relationships to the graph"""
        # Add paper node
        self.graph.add_node(paper_id, type='paper', metadata=metadata)
        self.node_types['paper'].add(paper_id)
        
        # Add author relationships
        authors = metadata.get('authors', [])
        for author in authors:
            author_id = f"author_{author['id']}" if isinstance(author, dict) else f"author_{author}"
            self.graph.add_node(author_id, type='author')
            self.node_types['author'].add(author_id)
            self.graph.add_edge(paper_id, author_id, type='authored')
            self.edge_types['authored'][(paper_id, author_id)] += 1
        
        # Add citation relationships
        citations = metadata.get('citations', [])
        for citation in citations:
            cited_id = f"paper_{citation['id']}" if isinstance(citation, dict) else f"paper_{citation}"
            self.graph.add_edge(paper_id, cited_id, type='cites')
            self.edge_types['cites'][(paper_id, cited_id)] += 1
        
        # Add keyword relationships
        keywords = metadata.get('keywords', [])
        for keyword in keywords:
            keyword_id = f"keyword_{keyword.lower()}"
            self.graph.add_node(keyword_id, type='keyword')
            self.node_types['keyword'].add(keyword_id)
            self.graph.add_edge(paper_id, keyword_id, type='contains_keyword')
            self.edge_types['contains_keyword'][(paper_id, keyword_id)] += 1
    
    def add_expert(self, expert_id: str, metadata: Dict[str, Any]) -> None:
        """Add an expert and their relationships to the graph"""
        # Add expert node
        self.graph.add_node(expert_id, type='author', metadata=metadata)
        self.node_types['author'].add(expert_id)
        
        # Add institution relationship
        institution = metadata.get('institution')
        if institution:
            institution_id = f"institution_{institution.lower()}"
            self.graph.add_node(institution_id, type='institution')
            self.node_types['institution'].add(institution_id)
            self.graph.add_edge(expert_id, institution_id, type='affiliated_with')
            self.edge_types['affiliated_with'][(expert_id, institution_id)] += 1
        
        # Add collaboration relationships
        collaborations = metadata.get('collaborations', [])
        for collab in collaborations:
            collab_id = f"author_{collab['id']}" if isinstance(collab, dict) else f"author_{collab}"
            self.graph.add_edge(expert_id, collab_id, type='collaborates_with')
            self.edge_types['collaborates_with'][(expert_id, collab_id)] += 1
    
    def add_equipment(self, equipment_id: str, metadata: Dict[str, Any]) -> None:
        """Add equipment and its relationships to the graph"""
        # Add equipment node
        self.graph.add_node(equipment_id, type='equipment', metadata=metadata)
        self.node_types['equipment'].add(equipment_id)
        
        # Add project relationships
        projects = metadata.get('used_in_projects', [])
        for project in projects:
            project_id = f"project_{project['id']}" if isinstance(project, dict) else f"project_{project}"
            self.graph.add_node(project_id, type='project')
            self.node_types['project'].add(project_id)
            self.graph.add_edge(project_id, equipment_id, type='uses_equipment')
            self.edge_types['uses_equipment'][(project_id, equipment_id)] += 1
    
    def calculate_pagerank(self, alpha: float = 0.85, max_iter: int = 100) -> Dict[str, float]:
        """Calculate PageRank scores for all nodes"""
        try:
            self.pagerank_scores = nx.pagerank(self.graph, alpha=alpha, max_iter=max_iter)
            self.last_update = datetime.now()
            logger.info(f"PageRank calculated for {len(self.pagerank_scores)} nodes")
            return self.pagerank_scores
        except Exception as e:
            logger.error(f"Error calculating PageRank: {e}")
            return {}
    
    def get_node_pagerank(self, node_id: str) -> float:
        """Get the PageRank score for a specific node"""
        if self.pagerank_scores is None:
            self.calculate_pagerank()
        return self.pagerank_scores.get(node_id, 0.0)
    
    def get_shortest_path(self, source_id: str, target_id: str) -> List[str]:
        """Find the shortest path between two nodes"""
        try:
            return nx.shortest_path(self.graph, source_id, target_id)
        except nx.NetworkXNoPath:
            return []
    
    def get_related_nodes(self, node_id: str, max_depth: int = 2) -> Set[str]:
        """Get all nodes related to a given node within max_depth hops"""
        related = set()
        if node_id not in self.graph:
            return related
        
        # BFS to find related nodes
        queue = [(node_id, 0)]  # (node, depth)
        visited = {node_id}
        
        while queue:
            current, depth = queue.pop(0)
            if depth > max_depth:
                continue
            
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    related.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        return related
    
    def extract_graph_features(self, node_id: str) -> Dict[str, float]:
        """Extract graph-based features for a node"""
        features = {}
        
        if node_id not in self.graph:
            return features
        
        # Basic graph metrics
        features['pagerank'] = self.get_node_pagerank(node_id)
        features['in_degree'] = self.graph.in_degree(node_id)
        features['out_degree'] = self.graph.out_degree(node_id)
        
        # Node type specific features
        node_type = self.graph.nodes[node_id].get('type')
        if node_type == 'paper':
            # Paper-specific features
            features['citation_count'] = len([e for e in self.graph.edges(node_id) 
                                           if self.graph.edges[e]['type'] == 'cites'])
            features['author_count'] = len([e for e in self.graph.edges(node_id) 
                                          if self.graph.edges[e]['type'] == 'authored'])
        
        elif node_type == 'author':
            # Author-specific features
            features['paper_count'] = len([e for e in self.graph.edges(node_id) 
                                         if self.graph.edges[e]['type'] == 'authored'])
            features['collaboration_count'] = len([e for e in self.graph.edges(node_id) 
                                                if self.graph.edges[e]['type'] == 'collaborates_with'])
        
        elif node_type == 'equipment':
            # Equipment-specific features
            features['usage_count'] = len([e for e in self.graph.edges(node_id) 
                                         if self.graph.edges[e]['type'] == 'uses_equipment'])
        
        return features
    
    def get_authority_score(self, node_id: str) -> float:
        """Calculate a comprehensive authority score for a node"""
        if node_id not in self.graph:
            return 0.0
        
        features = self.extract_graph_features(node_id)
        
        # Weighted combination of different metrics
        weights = {
            'pagerank': 0.4,
            'in_degree': 0.2,
            'out_degree': 0.1
        }
        
        # Add type-specific weights
        node_type = self.graph.nodes[node_id].get('type')
        if node_type == 'paper':
            weights['citation_count'] = 0.3
        elif node_type == 'author':
            weights['paper_count'] = 0.2
            weights['collaboration_count'] = 0.1
        elif node_type == 'equipment':
            weights['usage_count'] = 0.3
        
        # Calculate weighted score
        score = sum(features.get(metric, 0) * weight 
                   for metric, weight in weights.items())
        
        # Normalize to [0,1]
        return min(1.0, score / 10.0)  # Assuming max score around 10
    
    def get_connection_strength(self, source_id: str, target_id: str) -> float:
        """Calculate the strength of connection between two nodes"""
        if source_id not in self.graph or target_id not in self.graph:
            return 0.0
        
        # Get all paths between nodes
        try:
            paths = list(nx.all_simple_paths(self.graph, source_id, target_id, cutoff=3))
        except nx.NetworkXNoPath:
            return 0.0
        
        if not paths:
            return 0.0
        
        # Calculate path-based strength
        path_strengths = []
        for path in paths:
            strength = 1.0
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                edge_type = self.graph.edges[edge]['type']
                edge_weight = self.edge_types[edge_type][edge]
                strength *= edge_weight
            path_strengths.append(strength)
        
        # Combine path strengths
        return min(1.0, sum(path_strengths) / 10.0)  # Normalize to [0,1] 