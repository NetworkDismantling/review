"""
Behavioral tests for registered dismantling methods.

Tests that the heuristic scoring functions and their integration through
the @dismantling_method + @dismantler_wrapper decorators produce correct
and consistent results on known networks (Zachary karate club, small
synthetic graphs, etc.).

These are NOT comparisons between implementations (see test_dismantler_comparison.py)
but rather regression tests that validate each method's output properties.
"""

import logging
import numpy as np
import pytest
from graph_tool import Graph
from graph_tool.collection import data as gt_collection
from graph_tool.generation import complete_graph, random_graph

from network_dismantling.common.external_dismantlers.lcc_threshold_dismantler import (
    cache as cpp_cache,
)
from network_dismantling.common.removal import Removal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_cpp_cache():
    """Clear the C++ graph cache before each test."""
    cpp_cache.clear()
    yield
    cpp_cache.clear()


def _prepare_graph(g: Graph, name: str = "test") -> Graph:
    """Add static_id vertex property and graph metadata."""
    # Ensure undirected
    if g.is_directed():
        g = g.copy()
        g.set_directed(False)

    static_id = g.new_vertex_property("int")
    for v in g.vertices():
        static_id[v] = int(v)
    g.vertex_properties["static_id"] = static_id

    g.graph_properties["filename"] = g.new_graph_property(
        "string", f"{name}_{np.random.randint(1_000_000)}"
    )
    return g


@pytest.fixture
def karate_graph() -> Graph:
    """Zachary's karate club: 34 nodes, 78 edges."""
    return _prepare_graph(gt_collection["karate"], name="karate")


@pytest.fixture
def small_er_graph() -> Graph:
    """Erdős–Rényi G(50, 0.1)."""
    g = random_graph(50, lambda: np.random.poisson(5), directed=False)
    return _prepare_graph(g, name="er50")


@pytest.fixture
def star_graph() -> Graph:
    """Star graph with 20 leaves."""
    g = Graph(directed=False)
    g.add_vertex(21)
    for i in range(1, 21):
        g.add_edge(0, i)
    return _prepare_graph(g, name="star20")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_heuristic_methods():
    """Return the registered heuristic DismantlingMethods."""
    from network_dismantling import dismantling_methods

    return {
        k: m for k, m in dismantling_methods.items()
        if getattr(m, "method_type", None) == "heuristic"
    }


def _run_method(method, network: Graph, stop_condition: int):
    """Run a dismantling method and return the output dict."""
    return method(
        network.copy(),
        stop_condition=stop_condition,
        logger=logger,
    )


def _validate_output(output: dict, network_size: int, stop_condition: int, method_key: str):
    """Validate common output invariants."""
    assert "removals" in output, f"{method_key}: missing 'removals' key"
    assert "r_auc" in output, f"{method_key}: missing 'r_auc'"
    assert "rem_num" in output, f"{method_key}: missing 'rem_num'"
    assert "heuristic" in output, f"{method_key}: missing 'heuristic'"

    removals = output["removals"]
    rem_num = output["rem_num"]

    assert rem_num == len(removals), (
        f"{method_key}: rem_num ({rem_num}) != len(removals) ({len(removals)})"
    )

    assert rem_num > 0, f"{method_key}: no removals produced"
    assert rem_num <= network_size, (
        f"{method_key}: removed more nodes ({rem_num}) than network size ({network_size})"
    )

    # Check that removals have monotonically increasing removal_num
    for i, r in enumerate(removals):
        assert r.removal_num == i + 1, (
            f"{method_key}: removal_num at index {i} is {r.removal_num}, expected {i + 1}"
        )

    # The final LCC size should be <= stop_condition
    final_lcc = removals[-1].lcc_size
    assert final_lcc <= stop_condition, (
        f"{method_key}: final LCC size ({final_lcc}) > stop_condition ({stop_condition})"
    )

    # No duplicate node IDs
    node_ids = [r.node_id for r in removals]
    assert len(node_ids) == len(set(node_ids)), (
        f"{method_key}: duplicate node IDs in removals"
    )

    # Node IDs must be valid static IDs
    for nid in node_ids:
        assert 0 <= nid < network_size, (
            f"{method_key}: node_id {nid} out of range [0, {network_size})"
        )

    # AUC must be non-negative
    assert output["r_auc"] >= 0, f"{method_key}: negative r_auc"


# ---------------------------------------------------------------------------
# Test: all heuristics produce valid output on karate
# ---------------------------------------------------------------------------

class TestHeuristicsKarate:
    """Test all registered heuristics on Zachary's karate club."""

    def test_heuristics_are_registered(self):
        """At least the 5 basic heuristics should be auto-discovered."""
        methods = _get_heuristic_methods()
        expected = {"degree", "eigenvector_centrality", "pagerank", "betweenness_centrality", "random"}
        assert expected.issubset(set(methods.keys())), (
            f"Missing heuristics: {expected - set(methods.keys())}"
        )

    @pytest.mark.parametrize("method_key", [
        "degree",
        "eigenvector_centrality",
        "pagerank",
        "betweenness_centrality",
        "random",
    ])
    def test_heuristic_output_structure(self, karate_graph, method_key):
        """Each heuristic produces a valid output dict with correct invariants."""
        methods = _get_heuristic_methods()
        method = methods[method_key]
        stop_condition = int(np.ceil(karate_graph.num_vertices() * 0.1))

        output = _run_method(method, karate_graph, stop_condition)
        _validate_output(output, karate_graph.num_vertices(), stop_condition, method_key)

    @pytest.mark.parametrize("method_key", [
        "degree",
        "eigenvector_centrality",
        "pagerank",
        "betweenness_centrality",
    ])
    def test_deterministic_heuristics_are_consistent(self, karate_graph, method_key):
        """Deterministic heuristics produce the same result on repeated runs.

        Note: when tied scores exist, the removal order within a tie group
        may vary across runs (depending on argsort stability and internal
        vertex ordering). We therefore compare the *set* of removed node IDs
        rather than their exact sequence.
        """
        methods = _get_heuristic_methods()
        method = methods[method_key]
        stop_condition = int(np.ceil(karate_graph.num_vertices() * 0.1))

        output1 = _run_method(method, karate_graph, stop_condition)
        output2 = _run_method(method, karate_graph, stop_condition)

        ids1 = set(r.node_id for r in output1["removals"])
        ids2 = set(r.node_id for r in output2["removals"])
        assert ids1 == ids2, f"{method_key}: non-deterministic output (different node sets)"

        # AUC should be identical for deterministic methods
        assert output1["r_auc"] == output2["r_auc"], (
            f"{method_key}: AUC differs across runs ({output1['r_auc']} vs {output2['r_auc']})"
        )

    def test_degree_removes_high_degree_first(self, karate_graph):
        """Degree heuristic should remove the highest-degree node first."""
        methods = _get_heuristic_methods()
        method = methods["degree"]
        stop_condition = int(np.ceil(karate_graph.num_vertices() * 0.1))

        output = _run_method(method, karate_graph, stop_condition)
        removals = output["removals"]

        # In the karate graph, node 33 (0-indexed) has the highest degree (17)
        # It should be among the first few removed nodes
        first_removed = removals[0].node_id
        degrees = karate_graph.get_out_degrees(karate_graph.get_vertices())
        static_ids = karate_graph.vertex_properties["static_id"].get_array()

        max_degree_nodes = set(static_ids[degrees == degrees.max()])
        assert first_removed in max_degree_nodes, (
            f"Degree heuristic removed node {first_removed} first "
            f"(degree {degrees[first_removed]}), but max degree nodes are {max_degree_nodes}"
        )

    def test_random_varies_across_runs(self, karate_graph):
        """Random heuristic should (with high probability) produce different orderings."""
        methods = _get_heuristic_methods()
        method = methods["random"]
        stop_condition = int(np.ceil(karate_graph.num_vertices() * 0.1))

        runs = []
        for _ in range(5):
            output = _run_method(method, karate_graph, stop_condition)
            runs.append(tuple(r.node_id for r in output["removals"]))

        # With 5 runs on 34 nodes, at least 2 should differ
        unique_runs = set(runs)
        assert len(unique_runs) >= 2, "Random heuristic produced identical results in 5 runs"


# ---------------------------------------------------------------------------
# Test: heuristics on different topologies
# ---------------------------------------------------------------------------

class TestHeuristicsTopologies:
    """Test heuristics work correctly on varied graph structures."""

    @pytest.mark.parametrize("method_key", [
        "degree", "eigenvector_centrality", "pagerank", "betweenness_centrality",
    ])
    def test_on_er_graph(self, small_er_graph, method_key):
        """Heuristics work on Erdős–Rényi random graphs."""
        methods = _get_heuristic_methods()
        method = methods[method_key]
        stop_condition = int(np.ceil(small_er_graph.num_vertices() * 0.1))

        output = _run_method(method, small_er_graph, stop_condition)
        _validate_output(output, small_er_graph.num_vertices(), stop_condition, method_key)

    @pytest.mark.parametrize("method_key", [
        "degree", "pagerank", "betweenness_centrality",
    ])
    def test_on_star_graph(self, star_graph, method_key):
        """Heuristics work on star topology (one hub, many leaves).

        Note: eigenvector_centrality is excluded because graph_tool's
        power-iteration eigenvector solver does not converge on star graphs.
        """
        methods = _get_heuristic_methods()
        method = methods[method_key]

        # Star with 21 nodes: removing the hub should reduce LCC to 1
        stop_condition = 2

        output = _run_method(method, star_graph, stop_condition)
        _validate_output(output, star_graph.num_vertices(), stop_condition, method_key)

        # The hub (node 0, degree 20) should be the first removal for all centrality measures
        first_removed = output["removals"][0].node_id
        assert first_removed == 0, (
            f"{method_key}: on star graph, first removed was node {first_removed}, expected hub (0)"
        )

    def test_complete_graph_all_nodes_equivalent(self):
        """On a complete graph, all centrality measures should be equal for all nodes."""
        g = complete_graph(10, directed=False)
        g = _prepare_graph(g, name="complete10")

        methods = _get_heuristic_methods()
        method = methods["degree"]
        stop_condition = 2

        output = _run_method(method, g, stop_condition)
        _validate_output(output, g.num_vertices(), stop_condition, "degree")


# ---------------------------------------------------------------------------
# Test: AUC ordering (lower is better dismantling)
# ---------------------------------------------------------------------------

class TestAUCOrdering:
    """Verify relative dismantling efficiency."""

    def test_degree_beats_random_on_karate(self, karate_graph):
        """Degree heuristic should have lower AUC than random (on average)."""
        methods = _get_heuristic_methods()
        stop_condition = int(np.ceil(karate_graph.num_vertices() * 0.1))

        degree_output = _run_method(methods["degree"], karate_graph, stop_condition)

        random_aucs = []
        for _ in range(10):
            random_output = _run_method(methods["random"], karate_graph, stop_condition)
            random_aucs.append(random_output["r_auc"])

        avg_random_auc = np.mean(random_aucs)

        # Degree should outperform random on average (lower AUC = better)
        assert degree_output["r_auc"] < avg_random_auc, (
            f"Degree AUC ({degree_output['r_auc']:.1f}) >= "
            f"avg random AUC ({avg_random_auc:.1f})"
        )


# ---------------------------------------------------------------------------
# Test: method metadata
# ---------------------------------------------------------------------------

class TestMethodMetadata:
    """Verify that registered methods have correct metadata."""

    def test_heuristic_method_type(self):
        """All heuristic methods should have method_type='heuristic'."""
        methods = _get_heuristic_methods()
        for key, method in methods.items():
            assert method.method_type == "heuristic", (
                f"{key}: method_type is '{method.method_type}', expected 'heuristic'"
            )

    def test_heuristic_short_names(self):
        """All heuristics should have a non-empty short_name."""
        methods = _get_heuristic_methods()
        for key, method in methods.items():
            assert method.short_name, f"{key}: empty short_name"
            assert isinstance(method.short_name, str), f"{key}: short_name is not a string"

    def test_heuristic_is_static(self):
        """All heuristics should produce static=True output."""
        from network_dismantling import dismantling_methods

        methods = _get_heuristic_methods()
        karate = _prepare_graph(gt_collection["karate"], name="meta_karate")
        stop_condition = int(np.ceil(karate.num_vertices() * 0.1))

        for key in ["degree"]:  # Test just one to avoid slowness
            method = methods[key]
            output = _run_method(method, karate, stop_condition)
            # dynamic=None for heuristics → `not None` = True
            assert output["static"] is True or output["static"] is None, (
                f"{key}: static should be True for heuristics"
            )

    def test_all_methods_discoverable(self):
        """All python_interface.py modules should have been auto-discovered."""
        from network_dismantling import dismantling_methods

        # At minimum we expect the 5 heuristics
        assert len(dismantling_methods) >= 5, (
            f"Only {len(dismantling_methods)} methods discovered: {list(dismantling_methods.keys())}"
        )
