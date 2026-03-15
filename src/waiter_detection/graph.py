"""Graph construction and graph-based feature extraction."""

from typing import List
import numpy as np
import pandas as pd
import networkx as nx


class WaiterCardGraph:
    """Bipartite graph connecting waiters and loyalty cards."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize graph builder.
        
        Args:
            df: Processed transaction dataframe
        """
        self.df = df.copy()
        self.graph = None
        
    def build(self, min_transactions: int = 1) -> nx.Graph:
        """
        Build bipartite graph from transactions.
        
        Args:
            min_transactions: Minimum transactions to include an edge
            
        Returns:
            NetworkX bipartite graph
        """
        print("Building bipartite graph: Waiters <-> Loyalty Cards...")
        
        # Create edge list with weights
        edge_data = (
            self.df.groupby(['waiter_id', 'person_id'])
            .agg({
                'trn_id': 'count',
                'gross_amount': ['sum', 'mean'],
                'bonusses_used': ['sum', 'mean'],
                'bonusses_accum': ['sum', 'mean'],
                'discount_amount': ['sum', 'mean'],
                'bonus_used_flag': 'sum',
            })
            .reset_index()
        )
        
        # Flatten column names
        edge_data.columns = [
            'waiter_id', 'person_id', 'trn_count',
            'amount_sum', 'amount_mean',
            'bonus_used_sum', 'bonus_used_mean',
            'bonus_accum_sum', 'bonus_accum_mean',
            'discount_sum', 'discount_mean',
            'bonus_trn_count'
        ]
        
        # Filter by minimum transactions
        edge_data = edge_data[edge_data['trn_count'] >= min_transactions]
        
        # Create bipartite graph
        G = nx.Graph()
        
        # Add nodes with type attribute
        waiters = edge_data['waiter_id'].unique()
        cards = edge_data['person_id'].unique()
        
        for waiter in waiters:
            G.add_node(waiter, node_type='waiter')
        
        for card in cards:
            G.add_node(card, node_type='card')
        
        # Add edges with attributes
        for _, row in edge_data.iterrows():
            G.add_edge(
                row['waiter_id'],
                row['person_id'],
                trn_count=row['trn_count'],
                amount_sum=row['amount_sum'],
                amount_mean=row['amount_mean'],
                bonus_used_sum=row['bonus_used_sum'],
                bonus_used_mean=row['bonus_used_mean'],
                bonus_accum_sum=row['bonus_accum_sum'],
                bonus_accum_mean=row['bonus_accum_mean'],
                discount_sum=row['discount_sum'],
                discount_mean=row['discount_mean'],
                bonus_trn_count=row['bonus_trn_count'],
            )
        
        self.graph = G
        print(f"Graph built: {G.number_of_nodes()} nodes ({len(waiters)} waiters, {len(cards)} cards), "
              f"{G.number_of_edges()} edges")
        
        return G
    
    def extract_graph_features(self) -> pd.DataFrame:
        """
        Extract graph-based features for waiters (NO fraud information).
        
        Returns:
            DataFrame with waiter_id and graph features
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build() first.")
        
        print("Extracting graph-based features for waiters...")
        
        waiters = [n for n in self.graph.nodes() if self.graph.nodes[n].get('node_type') == 'waiter']
        
        features = []
        
        for waiter in waiters:
            feat = {'waiter_id': waiter}
            
            # Basic connectivity
            neighbors = list(self.graph.neighbors(waiter))
            feat['degree'] = len(neighbors)
            
            # Edge weight aggregations (transaction counts)
            edge_weights = [self.graph[waiter][n]['trn_count'] for n in neighbors]
            feat['total_trn_count'] = sum(edge_weights)
            feat['mean_trn_per_card'] = np.mean(edge_weights) if edge_weights else 0
            feat['max_trn_per_card'] = max(edge_weights) if edge_weights else 0
            feat['std_trn_per_card'] = np.std(edge_weights) if len(edge_weights) > 1 else 0
            feat['min_trn_per_card'] = min(edge_weights) if edge_weights else 0
            
            # Amount-based features
            amounts = [self.graph[waiter][n]['amount_sum'] for n in neighbors]
            feat['total_amount'] = sum(amounts)
            feat['mean_amount_per_card'] = np.mean(amounts) if amounts else 0
            feat['max_amount_per_card'] = max(amounts) if amounts else 0
            feat['std_amount_per_card'] = np.std(amounts) if len(amounts) > 1 else 0
            
            # Bonus features
            bonus_used = [self.graph[waiter][n]['bonus_used_sum'] for n in neighbors]
            feat['total_bonus_used'] = sum(bonus_used)
            feat['mean_bonus_used_per_card'] = np.mean(bonus_used) if bonus_used else 0
            
            bonus_accum = [self.graph[waiter][n]['bonus_accum_sum'] for n in neighbors]
            feat['total_bonus_accum'] = sum(bonus_accum)
            feat['mean_bonus_accum_per_card'] = np.mean(bonus_accum) if bonus_accum else 0
            
            # Discount features
            discounts = [self.graph[waiter][n]['discount_sum'] for n in neighbors]
            feat['total_discount'] = sum(discounts)
            feat['mean_discount_per_card'] = np.mean(discounts) if discounts else 0
            
            # Centrality measures (on waiter subgraph)
            try:
                waiter_graph = self._build_waiter_graph(waiter)
                if waiter_graph.number_of_nodes() > 1:
                    centrality = nx.degree_centrality(waiter_graph)
                    feat['waiter_centrality'] = centrality.get(waiter, 0)
                else:
                    feat['waiter_centrality'] = 0
            except:
                feat['waiter_centrality'] = 0
            
            # Clustering coefficient (on waiter subgraph)
            try:
                if waiter_graph.number_of_nodes() > 2:
                    clustering = nx.clustering(waiter_graph)
                    feat['waiter_clustering'] = clustering.get(waiter, 0)
                else:
                    feat['waiter_clustering'] = 0
            except:
                feat['waiter_clustering'] = 0
            
            # Card sharing patterns
            shared_cards = self._count_shared_cards(waiter)
            feat['avg_shared_cards'] = np.mean(shared_cards) if shared_cards else 0
            feat['max_shared_cards'] = max(shared_cards) if shared_cards else 0
            feat['min_shared_cards'] = min(shared_cards) if shared_cards else 0
            
            # First transaction patterns
            first_trn_cards = self._count_first_transaction_cards(waiter)
            feat['first_trn_card_count'] = first_trn_cards
            feat['first_trn_card_ratio'] = first_trn_cards / feat['degree'] if feat['degree'] > 0 else 0
            
            features.append(feat)
        
        graph_features_df = pd.DataFrame(features)
        return graph_features_df
    
    def _build_waiter_graph(self, waiter: str) -> nx.Graph:
        """Build a graph of waiters connected if they share cards."""
        waiter_graph = nx.Graph()
        waiter_neighbors = list(self.graph.neighbors(waiter))
        
        for card in waiter_neighbors:
            card_waiters = [n for n in self.graph.neighbors(card) 
                          if self.graph.nodes[n].get('node_type') == 'waiter']
            for w1 in card_waiters:
                for w2 in card_waiters:
                    if w1 != w2:
                        if not waiter_graph.has_edge(w1, w2):
                            waiter_graph.add_edge(w1, w2, weight=0)
                        waiter_graph[w1][w2]['weight'] += 1
        
        return waiter_graph
    
    def _count_shared_cards(self, waiter: str) -> List[int]:
        """Count how many cards each neighbor waiter shares with this waiter."""
        waiter_neighbors = list(self.graph.neighbors(waiter))
        shared_counts = []
        
        for card in waiter_neighbors:
            card_waiters = [n for n in self.graph.neighbors(card) 
                          if self.graph.nodes[n].get('node_type') == 'waiter' and n != waiter]
            shared_counts.append(len(card_waiters))
        
        return shared_counts
    
    def _count_first_transaction_cards(self, waiter: str) -> int:
        """Count cards where this waiter processed the first transaction."""
        waiter_cards = list(self.graph.neighbors(waiter))
        first_trn_count = 0
        
        for card in waiter_cards:
            waiter_card_trns = self.df[
                (self.df['waiter_id'] == waiter) & 
                (self.df['person_id'] == card)
            ].sort_values('trn_date')
            
            if len(waiter_card_trns) > 0:
                card_first_trn = self.df[self.df['person_id'] == card].sort_values('trn_date').iloc[0]
                if card_first_trn['waiter_id'] == waiter:
                    first_trn_count += 1
        
        return first_trn_count

