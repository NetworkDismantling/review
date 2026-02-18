"""Regression tests comparing subprocess and graph-tool reinsertion backends.

Verifies that the in-process C++ extension (``libreinsertion_gt.so``) produces
identical results to the subprocess-based reinsertion binary for various graph
topologies and seed/removal configurations.

Both backends implement the same reverse-greedy algorithm, so the outputs
must be identical (not just similar) for deterministic inputs.
"""

import logging

import numpy as np
import pytest
from graph_tool import Graph
from graph_tool.generation import random_graph

from network_dismantling.common.reinsertion.reinsertion_gt import (
    is_available as gt_is_available,
    reverse_greedy_reinsertion_gt,
)
from network_dismantling.common.reinsertion.reverse_greedy import (
    reverse_greedy_reinsertion as subprocess_reinsertion,
    cleanup_network_cache,
)

logger = logging.getLogger(__name__)

# Skip all tests if extension is not compiled
pytestmark = pytest.mark.skipif(
    not gt_is_available(),
    reason="libreinsertion_gt.so not compiled",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph_with_static_id(g: Graph) -> Graph:
    """Ensure graph has ``static_id`` vertex property (identity mapping)."""
    if "static_id" not in g.vertex_properties:
        sid = g.new_vertex_property("int")
        for v in g.vertices():
            sid[v] = int(v)
        g.vertex_properties["static_id"] = sid
    return g


def _make_path(n: int) -> Graph:
    """Path graph: 0—1—2—…—(n-1)."""
    g = Graph(directed=False)
    g.add_vertex(n)
    for i in range(n - 1):
        g.add_edge(i, i + 1)
    return _make_graph_with_static_id(g)


def _make_cycle(n: int) -> Graph:
    """Cycle graph: 0—1—…—(n-1)—0."""
    g = _make_path(n)
    g.add_edge(n - 1, 0)
    return g


def _make_star(n: int) -> Graph:
    """Star graph: hub=0 connected to 1…(n-1)."""
    g = Graph(directed=False)
    g.add_vertex(n)
    for i in range(1, n):
        g.add_edge(0, i)
    return _make_graph_with_static_id(g)


def _make_barbell(n: int) -> Graph:
    """Two cliques of size n connected by a single bridge edge."""
    g = Graph(directed=False)
    g.add_vertex(2 * n)
    # First clique: 0..n-1
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(i, j)
    # Second clique: n..2n-1
    for i in range(n, 2 * n):
        for j in range(i + 1, 2 * n):
            g.add_edge(i, j)
    # Bridge
    g.add_edge(n - 1, n)
    return _make_graph_with_static_id(g)


def _make_random_graph(n: int, avg_deg: int, seed: int = 42) -> Graph:
    """Erdős–Rényi-like random graph via graph-tool."""
    g = random_graph(n, lambda: np.random.poisson(avg_deg), directed=False)
    return _make_graph_with_static_id(g)


def _top_k_degree_seeds(g: Graph, k: int) -> list:
    """Return static_ids of the top-k vertices by degree."""
    static_id = g.vertex_properties["static_id"]
    degs = [(g.vertex(v).out_degree(), int(static_id[v])) for v in range(g.num_vertices())]
    degs.sort(key=lambda x: x[0], reverse=True)
    return [sid for _, sid in degs[:k]]


@pytest.fixture(autouse=True)
def _cleanup():
    """Clear the network file cache between tests."""
    yield
    cleanup_network_cache()


# ---------------------------------------------------------------------------
# Core comparison helper
# ---------------------------------------------------------------------------

def _compare_backends(g: Graph, seeds: list, stop_condition: int,
                      sort_strategy: int = 2):
    """Run both backends and assert they select the same nodes for removal.

    The *set* of remaining nodes must be identical.  The *order* (priorities)
    may differ because the subprocess binary has a known off-by-one bug in
    its ``sort_nodes_by_degree`` function (``W[node_id - 1]`` underflows for
    node 0, and ``degree(i + 1, g)`` skips vertex 0).
    """
    result_gt = reverse_greedy_reinsertion_gt(
        g, seeds, stop_condition, sort_strategy=sort_strategy, logger=logger,
    )
    result_sp = subprocess_reinsertion(
        g, seeds, stop_condition, sort_strategy=sort_strategy, logger=logger,
    )

    # Compare which nodes are marked for removal (non-zero entries)
    gt_nodes = set(i for i in range(len(result_gt)) if result_gt[i] > 0)
    sp_nodes = set(i for i in range(len(result_sp)) if result_sp[i] > 0)

    assert gt_nodes == sp_nodes, (
        f"Different remaining node SETS.\n"
        f"Graph: {g.num_vertices()} vertices, {g.num_edges()} edges\n"
        f"Seeds ({len(seeds)}): {seeds}\n"
        f"Stop condition: {stop_condition}, Sort strategy: {sort_strategy}\n"
        f"GT nodes: {sorted(gt_nodes)}\n"
        f"SP nodes: {sorted(sp_nodes)}\n"
        f"Only in GT: {sorted(gt_nodes - sp_nodes)}\n"
        f"Only in SP: {sorted(sp_nodes - gt_nodes)}"
    )

    return result_gt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReinsertionRegression:
    """Ensure gt extension and subprocess produce identical results."""

    def test_path_graph_two_seeds(self):
        """Path 0-1-2-3-4, remove nodes 1 and 3."""
        g = _make_path(5)
        _compare_backends(g, seeds=[1, 3], stop_condition=3)

    def test_path_graph_all_but_endpoints(self):
        """Path of 10, remove all interior nodes."""
        g = _make_path(10)
        seeds = list(range(1, 9))
        _compare_backends(g, seeds=seeds, stop_condition=4)

    def test_cycle_graph(self):
        """Cycle of 8, remove 3 nodes."""
        g = _make_cycle(8)
        _compare_backends(g, seeds=[0, 3, 6], stop_condition=4)

    def test_star_remove_hub(self):
        """Star with 10 leaves, remove the hub."""
        g = _make_star(11)
        _compare_backends(g, seeds=[0], stop_condition=2)

    def test_star_remove_leaves(self):
        """Star with 10 leaves, remove 5 leaves."""
        g = _make_star(11)
        _compare_backends(g, seeds=[1, 3, 5, 7, 9], stop_condition=3)

    def test_barbell_remove_bridge(self):
        """Barbell (2×5), remove the bridge vertices."""
        g = _make_barbell(5)
        _compare_backends(g, seeds=[4, 5], stop_condition=4)

    def test_barbell_mixed_seeds(self):
        """Barbell (2×5), remove nodes from both cliques."""
        g = _make_barbell(5)
        _compare_backends(g, seeds=[0, 2, 5, 7], stop_condition=3)

    def test_random_graph_small(self):
        """Random graph (30 nodes, avg degree 4), top-5 seeds."""
        np.random.seed(42)
        g = _make_random_graph(30, 4, seed=42)
        seeds = _top_k_degree_seeds(g, 5)
        _compare_backends(g, seeds=seeds, stop_condition=5)

    def test_random_graph_medium(self):
        """Random graph (100 nodes, avg degree 6), top-15 seeds."""
        np.random.seed(123)
        g = _make_random_graph(100, 6, seed=123)
        seeds = _top_k_degree_seeds(g, 15)
        _compare_backends(g, seeds=seeds, stop_condition=10)

    def test_random_graph_large_seed_set(self):
        """Random graph (50 nodes), remove 40% of nodes."""
        np.random.seed(99)
        g = _make_random_graph(50, 5, seed=99)
        seeds = _top_k_degree_seeds(g, 20)
        _compare_backends(g, seeds=seeds, stop_condition=5)

    def test_sort_strategy_original(self):
        """sort_strategy=0 in the subprocess binary writes nothing.

        This is a known bug in the legacy binary (write_output skips output
        when SORT_STRATEGY==0).  The gt extension correctly returns nodes in
        original order.  We only test that the gt extension does not crash.
        """
        g = _make_barbell(5)
        result = reverse_greedy_reinsertion_gt(
            g, removals=[0, 2, 5, 7], stop_condition=3,
            sort_strategy=0, logger=logger,
        )
        # Just verify we get some output
        assert np.count_nonzero(result) >= 1

    def test_sort_strategy_ascending(self):
        """Verify sort_strategy=1 (ascending degree) matches."""
        g = _make_barbell(5)
        _compare_backends(g, seeds=[0, 2, 5, 7], stop_condition=3,
                          sort_strategy=1)

    def test_sort_strategy_descending(self):
        """Verify sort_strategy=2 (descending degree) matches."""
        g = _make_barbell(5)
        _compare_backends(g, seeds=[0, 2, 5, 7], stop_condition=3,
                          sort_strategy=2)

    def test_single_seed(self):
        """Only one seed — should return it unchanged."""
        g = _make_path(5)
        _compare_backends(g, seeds=[2], stop_condition=3)

    def test_all_nodes_as_seeds(self):
        """Every node is a seed (extreme case)."""
        g = _make_cycle(6)
        seeds = list(range(6))
        _compare_backends(g, seeds=seeds, stop_condition=2)

    def test_dict_removals(self):
        """Verify dict-style removals {\"id\": ...} are handled."""
        g = _make_path(5)
        dict_seeds = [{"id": 1}, {"id": 3}]
        result_gt = reverse_greedy_reinsertion_gt(
            g, dict_seeds, stop_condition=3, logger=logger,
        )
        # Also test with plain ints
        result_gt_int = reverse_greedy_reinsertion_gt(
            g, [1, 3], stop_condition=3, logger=logger,
        )
        np.testing.assert_array_equal(result_gt, result_gt_int)

    def test_high_stop_condition(self):
        """Stop condition larger than LCC → all seeds reinserted except one."""
        g = _make_path(5)
        result = _compare_backends(g, seeds=[1, 3], stop_condition=100)
        # With very high stop condition, most seeds get reinserted
        nonzero = np.count_nonzero(result)
        assert nonzero >= 1, "At least one seed should remain"

    def test_dataset_graph(self):
        """Test with an actual dataset file if available."""
        from pathlib import Path

        dataset_file = Path(__file__).parent.parent / "dataset" / "test_review" / "inf-USAir97.gt"
        if not dataset_file.exists():
            pytest.skip(f"Dataset file not found: {dataset_file}")

        from graph_tool import load_graph

        g = load_graph(str(dataset_file))
        g = _make_graph_with_static_id(g)
        N = g.num_vertices()

        seeds = _top_k_degree_seeds(g, max(5, N // 10))
        stop_condition = max(3, int(N * 0.1))
        _compare_backends(g, seeds=seeds, stop_condition=stop_condition)


class TestReinsertionGtExtension:
    """Unit tests for the gt extension wrapper (not comparison)."""

    def test_is_available(self):
        assert gt_is_available()

    def test_output_shape(self):
        """Output array has the same length as num_vertices."""
        g = _make_path(10)
        result = reverse_greedy_reinsertion_gt(g, [3, 5, 7], stop_condition=4)
        assert len(result) == g.num_vertices()

    def test_output_zeros_for_non_seeds(self):
        """Non-seed nodes should have priority 0."""
        g = _make_path(10)
        seeds = [3, 5, 7]
        result = reverse_greedy_reinsertion_gt(g, seeds, stop_condition=4)

        non_seed_indices = set(range(10)) - set(seeds)
        # Some seeds may also become 0 (reinserted), but non-seeds must all be 0
        for i in non_seed_indices:
            assert result[i] == 0, f"Non-seed node {i} has non-zero priority {result[i]}"

    def test_positive_priorities(self):
        """Remaining seeds should have positive priorities."""
        g = _make_path(10)
        result = reverse_greedy_reinsertion_gt(g, [3, 5, 7], stop_condition=4)
        positives = result[result > 0]
        assert len(positives) > 0
        assert all(p > 0 for p in positives)

    def test_duplicate_seeds_handled(self):
        """Duplicate seed IDs should not cause issues."""
        g = _make_path(5)
        result = reverse_greedy_reinsertion_gt(g, [1, 1, 3, 3], stop_condition=3)
        assert len(result) == 5
        nonzero = np.count_nonzero(result)
        assert nonzero >= 1

    def test_empty_graph(self):
        """Empty graph (0 vertices) should return empty array."""
        g = Graph(directed=False)
        g.vertex_properties["static_id"] = g.new_vertex_property("int")
        result = reverse_greedy_reinsertion_gt(g, [], stop_condition=1)
        assert len(result) == 0
