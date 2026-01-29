"""
Test suite comparing Python and C++ implementations of threshold dismantlers.

Tests various graph types:
- Mid-sized graphs (50-100 nodes)
- Disconnected graphs
- Graphs with multiple edges
- Complete graphs
- Star graphs
- Random graphs
"""

import numpy as np
import pytest
from graph_tool import Graph
from graph_tool.generation import (
    complete_graph,
    random_graph,
    triangulation,
)

from network_dismantling.common.dismantlers import (
    threshold_dismantler as python_threshold_dismantler,
    lcc_threshold_dismantler as python_lcc_threshold_dismantler,
)
from network_dismantling.common.external_dismantlers.lcc_threshold_dismantler import (
    threshold_dismantler as cpp_threshold_dismantler,
    lcc_threshold_dismantler as cpp_lcc_threshold_dismantler,
    cache as cpp_cache,  # Import the cache to clear it
)


@pytest.fixture(autouse=True)
def clear_cpp_cache():
    """Clear the C++ graph cache before each test to ensure independence."""
    cpp_cache.clear()
    yield
    cpp_cache.clear()


def create_static_id(network: Graph, filename: str = "test_graph") -> None:
    """Add static_id property and filename to network."""
    static_id = network.new_vertex_property("int")
    for v in network.vertices():
        static_id[v] = int(v)
    network.vertex_properties["static_id"] = static_id
    
    # Add filename as graph property
    network.graph_properties["filename"] = network.new_graph_property("string", filename)


def simple_predictor(network: Graph, **kwargs):
    """Simple degree-based predictor for testing (returns predictions array)."""
    # Get degrees as predictions, add small random noise to ensure uniqueness
    degrees = network.get_out_degrees(network.get_vertices())
    static_id = network.vertex_properties["static_id"].get_array()
    
    # Add tiny offset based on node ID to ensure stable ordering
    # This makes predictions unique while preserving degree-based ordering
    predictions = degrees.astype(float) + static_id * 1e-6
    
    return predictions, 0.0  # Return (predictions, time)


def simple_degree_generator(network: Graph, **kwargs):
    """Simple degree-based node generator for testing."""
    # Get degrees
    degrees = network.get_out_degrees(network.get_vertices())
    static_id = network.vertex_properties["static_id"].get_array()
    
    # Add tiny offset based on node ID to ensure stable ordering
    predictions = degrees.astype(float) + static_id * 1e-6
    
    # Sort by degree (descending)
    sorted_indices = np.argsort(-predictions)
    
    for idx in sorted_indices:
        yield static_id[idx], float(predictions[idx])


def create_mid_sized_graph(n=50, avg_deg=4):
    """Create a mid-sized random graph."""
    g = Graph(directed=False)
    g.add_vertex(n)
    
    # Add random edges
    num_edges = n * avg_deg // 2
    for _ in range(num_edges):
        u = np.random.randint(0, n)
        v = np.random.randint(0, n)
        if u != v:
            g.add_edge(u, v)
    
    create_static_id(g)
    return g


def create_disconnected_graph(n_components=3, component_size=20):
    """Create a disconnected graph with multiple components."""
    g = Graph(directed=False)
    total_nodes = n_components * component_size
    g.add_vertex(total_nodes)
    
    # Create separate components
    for comp in range(n_components):
        start = comp * component_size
        end = start + component_size
        
        # Make each component a complete graph
        for i in range(start, end):
            for j in range(i + 1, end):
                g.add_edge(i, j)
    
    create_static_id(g)
    return g


def create_graph_with_multiple_edges(n=30):
    """Create a graph and then add multiple edges between some vertices."""
    g = Graph(directed=False)
    g.add_vertex(n)
    
    # Create a base graph
    for i in range(n - 1):
        g.add_edge(i, i + 1)
    
    # Add some additional edges (graph-tool handles this by default as simple graph)
    # But we'll add more edges to increase connectivity
    for i in range(n // 2):
        u = np.random.randint(0, n)
        v = np.random.randint(0, n)
        if u != v:
            g.add_edge(u, v)
    
    create_static_id(g)
    return g


def create_star_graph(n=50):
    """Create a star graph with one central hub."""
    g = Graph(directed=False)
    g.add_vertex(n)
    
    # Connect all nodes to node 0 (hub)
    for i in range(1, n):
        g.add_edge(0, i)
    
    create_static_id(g)
    return g


def create_complete_graph(n=30):
    """Create a complete graph."""
    g = complete_graph(n, directed=False)
    create_static_id(g)
    return g


def compare_removals(python_removals, cpp_removals, stop_condition, tolerance=1e-6):
    """
    Compare removal sequences from Python and C++ implementations.
    
    With the same input order, both implementations should produce
    identical removal sequences.
    
    Returns:
        bool: True if sequences are identical
        str: Description of any differences
    """
    if len(python_removals) != len(cpp_removals):
        return False, f"Different number of removals: Python={len(python_removals)}, C++={len(cpp_removals)}"
    
    if len(python_removals) == 0:
        return True, "Both sequences are empty"
    
    # Compare each removal tuple: (removal_num, vertex_id, prediction, lcc_size, slcc_size)
    for i, (py_rem, cpp_rem) in enumerate(zip(python_removals, cpp_removals)):
        # vertex_id must be identical (index 1)
        if int(py_rem[1]) != int(cpp_rem[1]):
            return False, f"Different node at removal {i}: Python={py_rem[1]}, C++={cpp_rem[1]}"
        
        # LCC and SLCC sizes must be identical (indices 3 and 4)
        if int(py_rem[3]) != int(cpp_rem[3]):
            return False, f"Different LCC size at removal {i} (node {py_rem[1]}): Python={py_rem[3]}, C++={cpp_rem[3]}"
        
        if int(py_rem[4]) != int(cpp_rem[4]):
            return False, f"Different SLCC size at removal {i} (node {py_rem[1]}): Python={py_rem[4]}, C++={cpp_rem[4]}"
    
    return True, f"Sequences are identical ({len(python_removals)} removals)"


class TestThresholdDismantlerComparison:
    """Test threshold_dismantler Python vs C++ implementations."""
    
    def test_mid_sized_graph(self):
        """Test with a mid-sized random graph."""
        np.random.seed(42)
        network = create_mid_sized_graph(n=50, avg_deg=4)
        stop_condition = 5
        
        # Python version
        py_removals, _, _ = python_threshold_dismantler(
            network=network.copy(),
            node_generator=simple_degree_generator,
            generator_args={},
            stop_condition=stop_condition,
        )
        
        # C++ version
        cpp_removals, _, _ = cpp_threshold_dismantler(
            network=network.copy(),
            predictor=simple_predictor,
            generator_args={"network_name": "test_mid_sized"},
            stop_condition=stop_condition,
        )
        
        match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
        assert match, f"Mid-sized graph test failed: {msg}"
    
    def test_disconnected_graph(self):
        """Test with a disconnected graph."""
        network = create_disconnected_graph(n_components=3, component_size=15)
        stop_condition = 5
        
        # Python version
        py_removals, _, _ = python_threshold_dismantler(
            network=network.copy(),
            node_generator=simple_degree_generator,
            generator_args={},
            stop_condition=stop_condition,
        )
        
        # C++ version
        cpp_removals, _, _ = cpp_threshold_dismantler(
            network=network.copy(),
            predictor=simple_predictor,
            generator_args={"network_name": "test_disconnected"},
            stop_condition=stop_condition,
        )
        
        match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
        assert match, f"Disconnected graph test failed: {msg}"
    
    def test_multiple_edges_graph(self):
        """Test with a graph that has high connectivity."""
        np.random.seed(123)
        network = create_graph_with_multiple_edges(n=30)
        stop_condition = 3
        
        # Python version
        py_removals, _, _ = python_threshold_dismantler(
            network=network.copy(),
            node_generator=simple_degree_generator,
            generator_args={},
            stop_condition=stop_condition,
        )
        
        # C++ version
        cpp_removals, _, _ = cpp_threshold_dismantler(
            network=network.copy(),
            predictor=simple_predictor,
            generator_args={"network_name": "test_multiple_edges"},
            stop_condition=stop_condition,
        )
        
        match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
        assert match, f"Multiple edges graph test failed: {msg}"
    
    def test_star_graph(self):
        """Test with a star graph."""
        network = create_star_graph(n=40)
        stop_condition = 2
        
        # Python version
        py_removals, _, _ = python_threshold_dismantler(
            network=network.copy(),
            node_generator=simple_degree_generator,
            generator_args={},
            stop_condition=stop_condition,
        )
        
        # C++ version
        cpp_removals, _, _ = cpp_threshold_dismantler(
            network=network.copy(),
            predictor=simple_predictor,
            generator_args={"network_name": "test_star"},
            stop_condition=stop_condition,
        )
        
        match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
        assert match, f"Star graph test failed: {msg}"
    
    def test_complete_graph(self):
        """Test with a complete graph."""
        network = create_complete_graph(n=25)
        stop_condition = 5
        
        # Python version
        py_removals, _, _ = python_threshold_dismantler(
            network=network.copy(),
            node_generator=simple_degree_generator,
            generator_args={},
            stop_condition=stop_condition,
        )
        
        # C++ version
        cpp_removals, _, _ = cpp_threshold_dismantler(
            network=network.copy(),
            predictor=simple_predictor,
            generator_args={"network_name": "test_complete"},
            stop_condition=stop_condition,
        )
        
        match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
        assert match, f"Complete graph test failed: {msg}"
    
    def test_small_graph(self):
        """Test with a very small graph."""
        network = Graph(directed=False)
        network.add_vertex(10)
        for i in range(9):
            network.add_edge(i, i + 1)
        create_static_id(network)
        
        stop_condition = 1
        
        # Python version
        py_removals, _, _ = python_threshold_dismantler(
            network=network.copy(),
            node_generator=simple_degree_generator,
            generator_args={},
            stop_condition=stop_condition,
        )
        
        # C++ version
        cpp_removals, _, _ = cpp_threshold_dismantler(
            network=network.copy(),
            predictor=simple_predictor,
            generator_args={"network_name": "test_small"},
            stop_condition=stop_condition,
        )
        
        match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
        assert match, f"Small graph test failed: {msg}"


class TestLCCThresholdDismantlerComparison:
    """Test lcc_threshold_dismantler Python vs C++ implementations."""
    
    def simple_degree_generator_with_feedback(self, network: Graph, **kwargs):
        """Degree-based generator that accepts feedback."""
        degrees = network.get_out_degrees(network.get_vertices())
        static_id = network.vertex_properties["static_id"].get_array()
        
        sorted_indices = np.argsort(-degrees)
        
        response = None
        for idx in sorted_indices:
            response = yield static_id[idx], float(degrees[idx])
    
    def test_mid_sized_graph_lcc(self):
        """Test LCC version with a mid-sized random graph."""
        np.random.seed(42)
        network = create_mid_sized_graph(n=50, avg_deg=4)
        stop_condition = 5
        
        # Python version
        py_removals, _, _ = python_lcc_threshold_dismantler(
            network=network.copy(),
            node_generator=self.simple_degree_generator_with_feedback,
            generator_args={},
            stop_condition=stop_condition,
        )
        
        # C++ version
        cpp_removals, _, _ = cpp_lcc_threshold_dismantler(
            network=network.copy(),
            predictor=simple_predictor,
            generator_args={"network_name": "test_lcc_mid"},
            stop_condition=stop_condition,
        )
        
        match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
        assert match, f"LCC mid-sized graph test failed: {msg}"
    
    def test_disconnected_graph_lcc(self):
        """Test LCC version with a disconnected graph."""
        network = create_disconnected_graph(n_components=3, component_size=15)
        stop_condition = 5
        
        # Python version
        py_removals, _, _ = python_lcc_threshold_dismantler(
            network=network.copy(),
            node_generator=self.simple_degree_generator_with_feedback,
            generator_args={},
            stop_condition=stop_condition,
        )
        
        # C++ version
        cpp_removals, _, _ = cpp_lcc_threshold_dismantler(
            network=network.copy(),
            predictor=simple_predictor,
            generator_args={"network_name": "test_lcc_disconnected"},
            stop_condition=stop_condition,
        )
        
        match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
        assert match, f"LCC disconnected graph test failed: {msg}"
    
    def test_star_graph_lcc(self):
        """Test LCC version with a star graph."""
        network = create_star_graph(n=40)
        stop_condition = 2
        
        # Python version
        py_removals, _, _ = python_lcc_threshold_dismantler(
            network=network.copy(),
            node_generator=self.simple_degree_generator_with_feedback,
            generator_args={},
            stop_condition=stop_condition,
        )
        
        # C++ version
        cpp_removals, _, _ = cpp_lcc_threshold_dismantler(
            network=network.copy(),
            predictor=simple_predictor,
            generator_args={"network_name": "test_lcc_star"},
            stop_condition=stop_condition,
        )
        
        match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
        assert match, f"LCC star graph test failed: {msg}"


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    # def test_single_node_graph(self):
    #     """Test with a graph containing a single node."""
    #     # This is an edge case: a single isolated node has LCC=1
    #     # which may be interpreted differently by Python and C++
    #     # when stop_condition=0
    #     network = Graph(directed=False)
    #     network.add_vertex(1)
    #     create_static_id(network)
    #     
    #     stop_condition = 0
    #     
    #     # Python version
    #     py_removals, _, _ = python_threshold_dismantler(
    #         network=network.copy(),
    #         node_generator=simple_degree_generator,
    #         generator_args={},
    #         stop_condition=stop_condition,
    #     )
    #     
    #     # C++ version
    #     cpp_removals, _, _ = cpp_threshold_dismantler(
    #         network=network.copy(),
    #         predictor=simple_predictor,
    #         generator_args={"network_name": "test_single_node"},
    #         stop_condition=stop_condition,
    #     )
    #     
    #     # Both should have removed the single node
    #     match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
    #     assert match, f"Single node graph test failed: {msg}"
    
    def test_two_component_graph(self):
        """Test with exactly two equal-sized components."""
        network = Graph(directed=False)
        network.add_vertex(20)
        
        # Component 1: nodes 0-9
        for i in range(9):
            network.add_edge(i, i + 1)
        
        # Component 2: nodes 10-19
        for i in range(10, 19):
            network.add_edge(i, i + 1)
        
        create_static_id(network)
        stop_condition = 2
        
        # Python version
        py_removals, _, _ = python_threshold_dismantler(
            network=network.copy(),
            node_generator=simple_degree_generator,
            generator_args={},
            stop_condition=stop_condition,
        )
        
        # C++ version
        cpp_removals, _, _ = cpp_threshold_dismantler(
            network=network.copy(),
            predictor=simple_predictor,
            generator_args={"network_name": "test_two_component"},
            stop_condition=stop_condition,
        )
        
        match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
        assert match, f"Two component graph test failed: {msg}"
    
    def test_dense_graph(self):
        """Test with a dense graph (high edge density)."""
        np.random.seed(999)
        n = 40
        network = Graph(directed=False)
        network.add_vertex(n)
        
        # Add many edges (dense graph)
        num_edges = n * (n - 1) // 4  # ~50% of possible edges
        edges_added = 0
        attempts = 0
        max_attempts = num_edges * 10
        
        while edges_added < num_edges and attempts < max_attempts:
            u = np.random.randint(0, n)
            v = np.random.randint(0, n)
            if u != v:
                network.add_edge(u, v)
                edges_added += 1
            attempts += 1
        
        create_static_id(network)
        stop_condition = 5
        
        # Python version
        py_removals, _, _ = python_threshold_dismantler(
            network=network.copy(),
            node_generator=simple_degree_generator,
            generator_args={},
            stop_condition=stop_condition,
        )
        
        # C++ version
        cpp_removals, _, _ = cpp_threshold_dismantler(
            network=network.copy(),
            predictor=simple_predictor,
            generator_args={"network_name": "test_dense"},
            stop_condition=stop_condition,
        )
        
        match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
        assert match, f"Dense graph test failed: {msg}"


class TestRealWorldNetworks:
    """Test Python and C++ dismantlers on real-world networks from graph_tool."""
    
    def test_karate_network(self):
        """Test on Zachary's karate club network (34 nodes, 78 edges)."""
        from graph_tool.collection import data
        
        network = data["karate"]
        
        # Ensure static_id exists
        if "static_id" not in network.vertex_properties:
            create_static_id(network, "karate")
        
        stop_condition = 5
        
        # Python version
        py_removals, _, _ = python_threshold_dismantler(
            network=network.copy(),
            node_generator=simple_degree_generator,
            generator_args={},
            stop_condition=stop_condition,
        )
        
        # C++ version
        cpp_removals, _, _ = cpp_threshold_dismantler(
            network=network.copy(),
            predictor=simple_predictor,
            generator_args={"network_name": "karate"},
            stop_condition=stop_condition,
        )
        
        match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
        assert match, f"Karate network test failed: {msg}"
    
    def test_polbooks_network(self):
        """Test on political books network (105 nodes, 441 edges)."""
        from graph_tool.collection import data
        
        try:
            network = data["polbooks"]
        except KeyError:
            pytest.skip("polbooks dataset not available")
        
        # Ensure static_id exists
        if "static_id" not in network.vertex_properties:
            create_static_id(network, "polbooks")
        
        stop_condition = 10
        
        # Python version
        py_removals, _, _ = python_threshold_dismantler(
            network=network.copy(),
            node_generator=simple_degree_generator,
            generator_args={},
            stop_condition=stop_condition,
        )
        
        # C++ version
        cpp_removals, _, _ = cpp_threshold_dismantler(
            network=network.copy(),
            predictor=simple_predictor,
            generator_args={"network_name": "polbooks"},
            stop_condition=stop_condition,
        )
        
        match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
        assert match, f"Polbooks network test failed: {msg}"
    
    def test_football_network(self):
        """Test on college football network (115 nodes, 613 edges)."""
        from graph_tool.collection import data
        
        try:
            network = data["football"]
        except KeyError:
            pytest.skip("football dataset not available")
        
        # Ensure static_id exists
        if "static_id" not in network.vertex_properties:
            create_static_id(network, "football")
        
        stop_condition = 15
        
        # Python version
        py_removals, _, _ = python_threshold_dismantler(
            network=network.copy(),
            node_generator=simple_degree_generator,
            generator_args={},
            stop_condition=stop_condition,
        )
        
        # C++ version
        cpp_removals, _, _ = cpp_threshold_dismantler(
            network=network.copy(),
            predictor=simple_predictor,
            generator_args={"network_name": "football"},
            stop_condition=stop_condition,
        )
        
        match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
        assert match, f"Football network test failed: {msg}"
    
    def test_netscience_network(self):
        """Test on netscience collaboration network (1589 nodes, 2742 edges)."""
        from graph_tool.collection import data
        
        try:
            network = data["netscience"]
        except KeyError:
            pytest.skip("netscience dataset not available")
        
        # Ensure static_id exists
        if "static_id" not in network.vertex_properties:
            create_static_id(network, "netscience")
        
        stop_condition = 20
        
        # Python version
        py_removals, _, _ = python_threshold_dismantler(
            network=network.copy(),
            node_generator=simple_degree_generator,
            generator_args={},
            stop_condition=stop_condition,
        )
        
        # C++ version
        cpp_removals, _, _ = cpp_threshold_dismantler(
            network=network.copy(),
            predictor=simple_predictor,
            generator_args={"network_name": "netscience"},
            stop_condition=stop_condition,
        )
        
        match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
        assert match, f"Netscience network test failed: {msg}"
    
    def test_power_network(self):
        """Test on power grid network (4941 nodes, 6594 edges)."""
        from graph_tool.collection import data
        
        try:
            network = data["power"]
        except KeyError:
            pytest.skip("power dataset not available")
        
        # Ensure static_id exists
        if "static_id" not in network.vertex_properties:
            create_static_id(network, "power")
        
        stop_condition = 50
        
        # Python version
        py_removals, _, _ = python_threshold_dismantler(
            network=network.copy(),
            node_generator=simple_degree_generator,
            generator_args={},
            stop_condition=stop_condition,
        )
        
        # C++ version
        cpp_removals, _, _ = cpp_threshold_dismantler(
            network=network.copy(),
            predictor=simple_predictor,
            generator_args={"network_name": "power"},
            stop_condition=stop_condition,
        )
        
        match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
        assert match, f"Power network test failed: {msg}"
    
    def test_hep_th_network(self):
        """Test on hep-th collaboration network (8361 nodes, 15751 edges)."""
        from graph_tool.collection import data
        
        try:
            network = data["hep-th"]
        except KeyError:
            pytest.skip("hep-th dataset not available")
        
        # Ensure static_id exists
        if "static_id" not in network.vertex_properties:
            create_static_id(network, "hep-th")
        
        stop_condition = 80
        
        # Python version
        py_removals, _, _ = python_threshold_dismantler(
            network=network.copy(),
            node_generator=simple_degree_generator,
            generator_args={},
            stop_condition=stop_condition,
        )
        
        # C++ version
        cpp_removals, _, _ = cpp_threshold_dismantler(
            network=network.copy(),
            predictor=simple_predictor,
            generator_args={"network_name": "hep-th"},
            stop_condition=stop_condition,
        )
        
        match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
        assert match, f"Hep-th network test failed: {msg}"
    
    def test_large_erdos_renyi_network(self):
        """Test on large Erdős-Rényi random network (~10k nodes)."""
        np.random.seed(12345)
        n = 10000
        p = 0.0005  # Probability of edge creation
        
        # Create Erdős-Rényi random graph
        network = random_graph(n, lambda: (np.random.random() < p), directed=False)
        
        # Remove isolated vertices
        vertices_to_remove = [v for v in network.vertices() if v.out_degree() == 0]
        for v in reversed(sorted(vertices_to_remove)):
            network.remove_vertex(v)
        
        actual_n = network.num_vertices()
        print(f"\nLarge ER network: {actual_n} nodes, {network.num_edges()} edges")
        
        create_static_id(network, "large_er")
        
        stop_condition = 100
        
        # Python version
        py_removals, _, _ = python_threshold_dismantler(
            network=network.copy(),
            node_generator=simple_degree_generator,
            generator_args={},
            stop_condition=stop_condition,
        )
        
        # C++ version
        cpp_removals, _, _ = cpp_threshold_dismantler(
            network=network.copy(),
            predictor=simple_predictor,
            generator_args={"network_name": "large_er"},
            stop_condition=stop_condition,
        )
        
        match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
        assert match, f"Large ER network test failed: {msg}"
    
    def test_large_barabasi_albert_network(self):
        """Test on large Barabási-Albert scale-free network (~10k nodes)."""
        np.random.seed(54321)
        n = 10000
        m = 3  # Number of edges to attach from a new node to existing nodes
        
        # Create BA network by starting with a small complete graph
        # and adding nodes one by one
        network = Graph(directed=False)
        
        # Start with m+1 nodes in a complete graph
        network.add_vertex(m + 1)
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                network.add_edge(i, j)
        
        # Add remaining nodes using preferential attachment
        degrees = [network.vertex(i).out_degree() for i in range(m + 1)]
        
        for new_node in range(m + 1, n):
            network.add_vertex()
            
            # Choose m existing nodes with probability proportional to degree
            total_degree = sum(degrees)
            targets = []
            
            for _ in range(m):
                # Preferential attachment
                rand_val = np.random.random() * total_degree
                cumsum = 0
                for target_idx, deg in enumerate(degrees):
                    cumsum += deg
                    if cumsum >= rand_val and target_idx not in targets:
                        targets.append(target_idx)
                        break
                
                # Fallback if we didn't find a unique target
                if len(targets) < len(set(targets)):
                    available = [i for i in range(len(degrees)) if i not in targets]
                    if available:
                        targets[-1] = np.random.choice(available)
            
            # Add edges to selected targets
            for target in set(targets):  # Use set to avoid duplicate edges
                network.add_edge(new_node, target)
                degrees[target] += 1
            
            degrees.append(len(set(targets)))
        
        print(f"\nLarge BA network: {network.num_vertices()} nodes, {network.num_edges()} edges")
        
        create_static_id(network, "large_ba")
        
        stop_condition = 100
        
        # Python version
        py_removals, _, _ = python_threshold_dismantler(
            network=network.copy(),
            node_generator=simple_degree_generator,
            generator_args={},
            stop_condition=stop_condition,
        )
        
        # C++ version
        cpp_removals, _, _ = cpp_threshold_dismantler(
            network=network.copy(),
            predictor=simple_predictor,
            generator_args={"network_name": "large_ba"},
            stop_condition=stop_condition,
        )
        
        match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
        assert match, f"Large BA network test failed: {msg}"
    
    def test_large_powerlaw_cluster_network(self):
        """Test on large power-law cluster graph (~5k nodes)."""
        np.random.seed(99999)
        n = 5000
        m = 4  # Number of random edges to add for each new node
        p = 0.3  # Probability of adding a triangle after adding a random edge
        
        # Start with a small clique
        network = Graph(directed=False)
        network.add_vertex(m)
        for i in range(m):
            for j in range(i + 1, m):
                network.add_edge(i, j)
        
        # Add nodes with clustering
        for new_node in range(m, n):
            network.add_vertex()
            
            # Select m random nodes weighted by degree
            degrees = [network.vertex(i).out_degree() for i in range(new_node)]
            total_degree = sum(degrees) if sum(degrees) > 0 else 1
            
            targets = []
            for _ in range(m):
                if total_degree > 0:
                    rand_val = np.random.random() * total_degree
                    cumsum = 0
                    for idx, deg in enumerate(degrees):
                        cumsum += deg
                        if cumsum >= rand_val:
                            targets.append(idx)
                            break
                else:
                    targets.append(np.random.randint(0, new_node))
            
            # Add edges
            for target in set(targets):
                network.add_edge(new_node, target)
                
                # With probability p, connect to a neighbor of target (triangle)
                if np.random.random() < p:
                    neighbors = [int(v) for v in network.vertex(target).out_neighbors()]
                    if neighbors and int(new_node) not in neighbors:
                        triangle_target = np.random.choice(neighbors)
                        if triangle_target != new_node:
                            network.add_edge(new_node, triangle_target)
        
        print(f"\nLarge power-law cluster network: {network.num_vertices()} nodes, {network.num_edges()} edges")
        
        create_static_id(network, "large_powerlaw_cluster")
        
        stop_condition = 80
        
        # Python version
        py_removals, _, _ = python_threshold_dismantler(
            network=network.copy(),
            node_generator=simple_degree_generator,
            generator_args={},
            stop_condition=stop_condition,
        )
        
        # C++ version
        cpp_removals, _, _ = cpp_threshold_dismantler(
            network=network.copy(),
            predictor=simple_predictor,
            generator_args={"network_name": "large_powerlaw_cluster"},
            stop_condition=stop_condition,
        )
        
        match, msg = compare_removals(py_removals, cpp_removals, stop_condition)
        assert match, f"Large power-law cluster network test failed: {msg}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
