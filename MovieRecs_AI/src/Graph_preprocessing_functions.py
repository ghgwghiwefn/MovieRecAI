import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops, label
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import HyperParameters
from sklearn.metrics.pairwise import cosine_similarity

import GNN_Data_cleanup

def calculate_cosine_similarity(genres_movie_1, genres_movie_2):
    vector_1 = np.array(genres_movie_1)
    vector_2 = np.array(genres_movie_2)
    return cosine_similarity([vector_1], [vector_2])[0][0]

def prune_graph(G, max_ratio=HyperParameters.PRUNE_RATIO):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    similarity_scores = sorted(set(attrs['similarity_score'].item() if isinstance(attrs['similarity_score'], np.ndarray) else attrs['similarity_score'] 
                               for node1, node2, attrs in G.edges(data=True) if 'similarity_score' in attrs))
    if similarity_scores:  # Ensure the list is not empty
        similarity_scores.pop(0)
    #print(similarity_scores)
    for score in similarity_scores:
        if score > HyperParameters.MAX_REMOVAL_THRESHOLD or num_edges/num_nodes < max_ratio:
            break 
        
        edges_to_remove = []  # Collect edges to be removed
        for node in G.nodes:
            if node == "U":
                continue
            connected_edges = G.edges(node, data=True)
            for node1, node2, attrs in connected_edges:
                if attrs['similarity_score'] == score:
                    edges_to_remove.append((node1, node2))
        #Check which nodes to skip edge removal on
        #print(edges_to_remove)
        edges_being_removed = []
        for node1, node2 in edges_to_remove:
            num_edges_node1 = G.degree(node1)
            num_edges_node2 = G.degree(node2)
            node1_similarity_scores = [
                    attrs['similarity_score']
                    for _, _, attrs in G.edges(node1, data=True)
                    if 'similarity_score' in attrs
                ]
            node2_similarity_scores = [
                    attrs['similarity_score']
                    for _, _, attrs in G.edges(node1, data=True)
                    if 'similarity_score' in attrs
                ]
            num_lowest_in_node1 = node1_similarity_scores.count(score)
            num_lowest_in_node2 = node2_similarity_scores.count(score)
            if num_edges_node1-num_lowest_in_node1 >= HyperParameters.MIN_CONNECTION_THRESHOLD and num_edges_node2-num_lowest_in_node2 >= HyperParameters.MIN_CONNECTION_THRESHOLD:
                edges_being_removed.append((node1, node2))

        # Now remove the edges outside the loop
        for node1, node2 in edges_being_removed:
            try:
                G.remove_edge(node1, node2)
            except:
                continue
        
        # Recompute number of edges after pruning
        num_edges = G.number_of_edges()

    return G

def convert_to_data(data):
    # Ensure x includes all node features
    data.rating = torch.tensor(data.rating, dtype=torch.float32) if not isinstance(data.rating, torch.Tensor) else data.rating.clone().detach()
    data.similarity_score = torch.tensor(data.similarity_score, dtype=torch.float32) if not isinstance(data.similarity_score, torch.Tensor) else data.similarity_score.clone().detach()
    
    data = Data(
        x=torch.cat([data.label.unsqueeze(1), data.gender, data.age, data.genres], dim=1).to(torch.float32),
        edge_index=data.edge_index.to(torch.long),
        edge_attr=torch.cat(
            [data.rating.unsqueeze(1), data.similarity_score.unsqueeze(1)], dim=1
        ).to(torch.float32),  # Combine edge attributes into a single tensor
        num_nodes=data.num_nodes  # Ensure number of nodes is explicitly set
    )
    #print(data)  # Debugging print to check the input structure
    return data

def create_graph(user):
    # Create graph
    G = nx.Graph()

    #PADDING ARRAYS
    gender_pad = np.full(1, -1)
    age_pad = np.full(1, -1)
    genres_pad = np.full(len(HyperParameters.GENRES), -1)

    rating_pad = np.full(1, -1)
    similarity_score_pad = -1
    # Add User Node
    #user_id_string = str(user['user_id'])
    user_id_string = "U"
    user_id_pad = -1
    G.add_node(user_id_string, label=user_id_pad, gender=np.array([user['gender']]), age=np.array([user['age']]), genres=genres_pad)
    
    # Add movie nodes and user edges
    for index, movie in enumerate(user['movie_ids']):
        #print(f"Movie label: {movie}")
        G.add_node(movie, label=movie, gender=gender_pad, age=age_pad, genres=user['genres'][index])
        G.add_edge(user_id_string, movie, rating=user['ratings'][index], similarity_score=similarity_score_pad)

    for index, current_node in enumerate(user['movie_ids']):
        movie_similarity_scores = [
            (user['movie_ids'][index2], calculate_cosine_similarity(user['genres'][index], user['genres'][index2]))
            for index2 in range(index + 1, len(user['movie_ids']))
        ]
        sorted_similarity_scores = sorted(movie_similarity_scores, key=lambda x: x[1], reverse=True)

        # Get current similarity scores of edges on this node
        current_similarity_scores = sorted(
            [attributes['similarity_score'] for _, _, attributes in G.edges(current_node, data=True) if 'similarity_score' in attributes]
        )
        num_edges = G.degree(current_node)
        freedom = HyperParameters.FREEDOM - num_edges

        # Determine threshold
        if freedom <= 0:
            threshold = max(current_similarity_scores[0], HyperParameters.THRESHOLD) if current_similarity_scores else HyperParameters.THRESHOLD
        elif len(sorted_similarity_scores) > freedom:
            threshold = sorted_similarity_scores[freedom][1]
        else:
            threshold = HyperParameters.THRESHOLD

        # Add edges between movies
        for movie, similarity_score in sorted_similarity_scores:
            if similarity_score < threshold or similarity_score == 0:
                break
            G.add_edge(current_node, movie, rating=rating_pad, similarity_score=similarity_score)
            current_similarity_scores.append(similarity_score)
        
    #Edge pruning for large graphs
    G = prune_graph(G)

    # Debug information
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    if G.number_of_nodes() <= 1000:
        draw_graph(G)
    
    #Converting graph to Data Object
    data = from_networkx(G)
    data = convert_to_data(data)

    return data


def draw_graph(G, label=''):
    # Create a figure
    plt.figure(figsize=(8, 8))
    
    # Position nodes using the spring layout (you can experiment with other layouts)
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='skyblue', alpha=0.7)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='gray')
    
    # Draw labels for nodes (user and movie IDs)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
    
    # Filter edges with rating != -1
    edge_labels = {
        (u, v): rating
        for u, v, rating in G.edges(data='rating')
        if rating != -1
    }

    # Draw edge labels (ratings) for edges where rating != -1
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='red')

    # Show plot
    plt.title(label)
    plt.axis('off')  # Hide axis
    plt.show()

if __name__ == "__main__":
    GNN_Data_cleanup.clean_data()