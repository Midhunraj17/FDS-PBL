import numpy as np

def validate_traffic_array(data, expected_nodes=None, expected_features=None):
    if not isinstance(data, np.ndarray) or data.ndim != 3:
        raise ValueError('Traffic data must be a 3D NumPy array [time, nodes, features].')
    if not np.isfinite(data).all():
        raise ValueError('Traffic data contains NaN or infinite values.')
    if expected_nodes is not None and data.shape[1] != expected_nodes:
        raise ValueError(f'Expected {expected_nodes} nodes, got {data.shape[1]}.')
    if expected_features is not None and data.shape[2] != expected_features:
        raise ValueError(f'Expected {expected_features} features, got {data.shape[2]}.')
    return {'time_steps': data.shape[0], 'nodes': data.shape[1], 'features': data.shape[2], 'min': float(data.min()), 'max': float(data.max())}

def validate_edges(edges, num_nodes):
    if edges.ndim != 2 or edges.shape[1] < 2:
        raise ValueError('Edges must have at least two columns: from, to.')
    src, dst = edges[:,0].astype(int), edges[:,1].astype(int)
    if (src < 0).any() or (dst < 0).any() or (src >= num_nodes).any() or (dst >= num_nodes).any():
        raise ValueError('Edge contains an invalid node index.')
    return {'edges': int(len(edges)), 'nodes': int(num_nodes)}
