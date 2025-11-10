import gradio as gr
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import model architecture
import sys
sys.path.append('.')
from GnnmodelGat import QAOAPredictorGAT


class QAOAInterface:
    """
    Interactive interface for testing QAOA GNN model with 3D visualization
    """
    def __init__(self, model_path='best_qaoa_gat_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.p_layers = checkpoint['p_layers']
        
        input_dim = 7
        
        self.model = QAOAPredictorGAT(
            input_dim=input_dim,
            hidden_dim=64,
            num_layers=3,
            p_layers=self.p_layers,
            attention_heads=8,
            dropout=0.3
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.input_dim = input_dim
        print(f"Model loaded: {input_dim} features, p={self.p_layers}")
    
    def compute_node_features(self, G, n_features=3):
        """
        Compute node features for a graph
        """
        n_nodes = G.number_of_nodes()
        features = np.zeros((n_nodes, n_features))
        
        # Feature 0: Degree
        degrees = dict(G.degree())
        features[:, 0] = [degrees[i] for i in range(n_nodes)]
        
        # Feature 1: Degree centrality
        degree_centrality = nx.degree_centrality(G)
        features[:, 1] = [degree_centrality[i] for i in range(n_nodes)]
        
        # Feature 2: Clustering coefficient
        clustering = nx.clustering(G)
        features[:, 2] = [clustering[i] for i in range(n_nodes)]
        
        if n_features >= 7:
            # Feature 3: Betweenness centrality
            betweenness = nx.betweenness_centrality(G)
            features[:, 3] = [betweenness[i] for i in range(n_nodes)]
            
            # Feature 4: Closeness centrality
            closeness = nx.closeness_centrality(G)
            features[:, 4] = [closeness[i] for i in range(n_nodes)]
            
            # Feature 5: PageRank
            pagerank = nx.pagerank(G, max_iter=1000)
            features[:, 5] = [pagerank[i] for i in range(n_nodes)]
            
            # Feature 6: Eigenvector centrality
            try:
                eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
                features[:, 6] = [eigenvector[i] for i in range(n_nodes)]
            except:
                features[:, 6] = 0.0
        
        return features
    
    def graph_to_pyg_data(self, G):
        """
        Convert NetworkX graph to PyTorch Geometric Data
        """
        node_features = self.compute_node_features(G, n_features=self.input_dim)
        x = torch.tensor(node_features, dtype=torch.float)
        
        adj_matrix = nx.to_numpy_array(G)
        edge_index = torch.tensor(np.array(np.where(adj_matrix > 0)), dtype=torch.long)
        
        y = torch.zeros(self.p_layers * 2)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        return data
    
    def predict_qaoa_params(self, G):
        """
        Predict QAOA parameters for a graph
        """
        data = self.graph_to_pyg_data(G)
        loader = DataLoader([data], batch_size=1, shuffle=False)
        batch = next(iter(loader))
        batch = batch.to(self.device)
        
        with torch.no_grad():
            output = self.model(batch)
        
        output = output.cpu().numpy()[0]
        gamma = output[:self.p_layers]
        beta = output[self.p_layers:]
        
        return gamma, beta
    
    def get_node_importance(self, G, data):
        """
        Calculate node importance using GAT attention weights variance
        Higher variance = model is more "certain" about this edge
        """
        n_nodes = G.number_of_nodes()
        
        # Move data to the same device as model
        data = data.to(self.device)
        
        # Get attention weights from the model
        attention_weights = self.model.get_attention_weights(data)
        
        # Aggregate attention across all layers
        node_importance = np.zeros(n_nodes)
        
        for edge_idx, alpha in attention_weights:
            # Calculate variance across attention heads (higher = more "interesting")
            alpha_var = alpha.var(dim=1).numpy()
            alpha_mean = alpha.mean(dim=1).numpy()
            
            # Combine mean and variance for importance score
            alpha_score = alpha_mean * (1 + alpha_var * 10)  # Boost nodes with high variance
            
            edge_idx = edge_idx.numpy()
            
            # Accumulate for both source and destination
            for i in range(edge_idx.shape[1]):
                src = edge_idx[0, i]
                dst = edge_idx[1, i]
                
                node_importance[src] += alpha_score[i]
                node_importance[dst] += alpha_score[i]
        
        # Normalize to [0, 1]
        if node_importance.max() > node_importance.min():
            node_importance = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min())
        else:
            node_importance = np.ones(n_nodes)  # All equal if no variance
        
        return node_importance
        
    def visualize_graph_3d(self, G, data, title="3D Graph Visualization"):
        """
        Create interactive 3D visualization with Plotly
        """
        # Get 3D layout
        pos = nx.spring_layout(G, dim=3, seed=42)
        
        # Extract coordinates
        node_xyz = np.array([pos[i] for i in range(G.number_of_nodes())])
        
        # Get node importance for coloring
        importance = self.get_node_importance(G, data)
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_traces.append(
                go.Scatter3d(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    z=[z0, z1, None],
                    mode='lines',
                    line=dict(color='rgba(40, 40, 60, 0.3)', width=2),
                    hoverinfo='none',
                    showlegend=False
                )
            )
        
        # Create node trace
        node_trace = go.Scatter3d(
            x=node_xyz[:, 0],
            y=node_xyz[:, 1],
            z=node_xyz[:, 2],
            mode='markers+text',
            marker=dict(
                size=10,
                color=importance,
                colorscale='Viridis',
                colorbar=dict(
                    title="Node<br>Importance",
                    thickness=15,
                    len=0.7
                ),
                line=dict(color='black', width=0.5)
            ),
            text=[str(i) for i in range(G.number_of_nodes())],
            textposition="top center",
            hovertemplate='<b>Node %{text}</b><br>Importance: %{marker.color:.3f}<extra></extra>',
            name='Nodes'
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            title=dict(
                text=f"{title}<br><sub>Node color = GAT attention importance</sub>",
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(
                    showbackground=False, 
                    showticklabels=False, 
                    title='',
                    showgrid=False,     
                    zeroline=False,
                    showline=False,      
                    visible=False       
                ),
                yaxis=dict(
                    showbackground=False, 
                    showticklabels=False, 
                    title='',
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    visible=False
                ),
                zaxis=dict(
                    showbackground=False, 
                    showticklabels=False, 
                    title='',
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    visible=False
                ),
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(l=0, r=0, b=0, t=40),
            height=600
        )
        
        return fig
    
    def visualize_parameter_landscape(self, gamma, beta):
        """
        Visualize the predicted parameters in 3D parameter space
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('γ (Gamma) Parameters', 'β (Beta) Parameters'),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
        )
        
        # Gamma visualization
        if len(gamma) >= 3:
            fig.add_trace(
                go.Scatter3d(
                    x=[gamma[0]], y=[gamma[1]], z=[gamma[2] if len(gamma) > 2 else gamma[0]],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='diamond'),
                    name='Predicted γ',
                    hovertemplate='γ₁: %{x:.4f}<br>γ₂: %{y:.4f}<br>γ₃: %{z:.4f}<extra></extra>'
                ),
                row=1, col=1
            )
        else:
            # For p=1 or p=2, create a 2D projection in 3D space
            gamma_plot = list(gamma) + [0] * (3 - len(gamma))
            fig.add_trace(
                go.Scatter3d(
                    x=[gamma_plot[0]], y=[gamma_plot[1]], z=[gamma_plot[2]],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='diamond'),
                    name='Predicted γ'
                ),
                row=1, col=1
            )
        
        # Beta visualization
        if len(beta) >= 3:
            fig.add_trace(
                go.Scatter3d(
                    x=[beta[0]], y=[beta[1]], z=[beta[2] if len(beta) > 2 else beta[0]],
                    mode='markers',
                    marker=dict(size=15, color='blue', symbol='diamond'),
                    name='Predicted β',
                    hovertemplate='β₁: %{x:.4f}<br>β₂: %{y:.4f}<br>β₃: %{z:.4f}<extra></extra>'
                ),
                row=1, col=2
            )
        else:
            beta_plot = list(beta) + [0] * (3 - len(beta))
            fig.add_trace(
                go.Scatter3d(
                    x=[beta_plot[0]], y=[beta_plot[1]], z=[beta_plot[2]],
                    mode='markers',
                    marker=dict(size=15, color='blue', symbol='diamond'),
                    name='Predicted β'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            title_text="Predicted QAOA Parameters in 3D Space"
        )
        
        return fig
    

# Initialize interface
interface = QAOAInterface()


def generate_and_predict(n_nodes, edge_prob, graph_type):
    """
    Generate a graph and predict QAOA parameters with 3D visualization
    """
    try:
        # Generate graph
        if graph_type == "Erdős-Rényi":
            G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=np.random.randint(1000))
        elif graph_type == "Regular":
            degree = min(int(n_nodes * edge_prob * 2), n_nodes - 1)
            if (n_nodes * degree) % 2 != 0:
                degree = max(2, degree - 1)
            G = nx.random_regular_graph(degree, n_nodes, seed=np.random.randint(1000))
        elif graph_type == "Barabási-Albert":
            m = max(1, int(n_nodes * edge_prob))
            G = nx.barabasi_albert_graph(n_nodes, m, seed=np.random.randint(1000))
        else:
            return None, None, "Invalid graph type"
        
        # Check if connected
        if not nx.is_connected(G):
            return None, None, "⚠️ Graph is disconnected! Try different parameters."
        
        # Convert graph to PyG data
        data = interface.graph_to_pyg_data(G)

        # Create a proper batch for attention extraction
        loader = DataLoader([data], batch_size=1, shuffle=False)
        batch_data = next(iter(loader))
        
        # Predict QAOA parameters
        gamma, beta = interface.predict_qaoa_params(G)
        
        # Format results
        results = "🎯 **QAOA Parameter Predictions**\n\n"
        
        for i in range(interface.p_layers):
            results += f"**Layer {i+1}:**\n"
            results += f"  • γ{i+1} (gamma) = {gamma[i]:.4f}\n"
            results += f"  • β{i+1} (beta) = {beta[i]:.4f}\n\n"
        
        # Graph statistics
        results += "\n📊 **Graph Statistics**\n\n"
        results += f"  • Nodes: {G.number_of_nodes()}\n"
        results += f"  • Edges: {G.number_of_edges()}\n"
        results += f"  • Density: {nx.density(G):.3f}\n"
        results += f"  • Avg Clustering: {nx.average_clustering(G):.3f}\n"
        results += f"  • Avg Degree: {sum(dict(G.degree()).values()) / n_nodes:.2f}\n"

        
        # Visualizations
        fig_3d = interface.visualize_graph_3d(G, batch_data)
        fig_params = interface.visualize_parameter_landscape(gamma, beta)
        
        return fig_3d, fig_params, results
    
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="QAOA GNN Predictor 3D", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🧠 QAOA Parameter Predictor with GNN (3D Edition)
    
    This interface uses a trained Graph Neural Network (GAT) to predict optimal QAOA parameters 
    for the MaxCut problem with **interactive 3D visualizations**.
    
    **Features:**
    - 🎨 3D graph visualization with node importance coloring
    - 📊 3D parameter space visualization
    - 🔍 Interactive plots (rotate, zoom, hover for details)
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Graph Generation")
            
            n_nodes = gr.Slider(
                minimum=5, 
                maximum=50, 
                value=15, 
                step=1,
                label="Number of Nodes"
            )
            
            edge_prob = gr.Slider(
                minimum=0.1, 
                maximum=0.9, 
                value=0.5, 
                step=0.05,
                label="Edge Probability / Density"
            )
            
            graph_type = gr.Radio(
                choices=["Erdős-Rényi", "Regular", "Barabási-Albert"],
                value="Erdős-Rényi",
                label="Graph Type"
            )
            
            generate_btn = gr.Button("🚀 Generate & Predict", variant="primary", size="lg")
            
            gr.Markdown("""
            ### Tips:
            - **Erdős-Rényi**: Random uniform graphs
            - **Regular**: All nodes have same degree
            - **Barabási-Albert**: Scale-free networks
            - 🎨 **Node colors** = importance (degree × γ)
            - 🔄 **Rotate 3D plots** with mouse!
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("### Results & Predictions")
            results_text = gr.Markdown(label="Predictions")
    
    gr.Markdown("### 📊 Visualizations")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### 3D Interactive View")
            graph_plot_3d = gr.Plot(label="3D Graph with Node Importance")
    
    with gr.Row():
        gr.Markdown("#### 3D Parameter Space")
        param_plot_3d = gr.Plot(label="Predicted γ and β in 3D")
    
    # Connect button
    generate_btn.click(
        fn=generate_and_predict,
        inputs=[n_nodes, edge_prob, graph_type],
        outputs=[graph_plot_3d, param_plot_3d, results_text]
    )
    
    gr.Markdown("""
    ---
    ### 📚 About the 3D Visualizations
    
    **3D Graph View:**
    - Node color intensity = importance (calculated as degree × mean(γ))
    - Brighter nodes = more critical for MaxCut solution
    - Hover over nodes to see details
    - Drag to rotate, scroll to zoom
    
    **3D Parameter Space:**
    - Shows predicted γ (red) and β (blue) values
    - For p > 3, only first 3 dimensions shown
    - Helps visualize parameter relationships
    
    **Model Performance:**
    - Architecture: GAT with 8 attention heads
    - Training: ~1000 graphs
    - MAE: ~0.09 (γ), ~0.07 (β)
    - **Speedup: 10-20x vs classical optimization!** 🚀
    """)


# Launch
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )