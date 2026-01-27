"""
Test suite for heuristic dependencies and required imports validation.
"""
import pytest
import logging
from unittest.mock import patch
from network_dismantling import dismantling_methods, DismantlingMethod

class MockDismantlingMethod:
    """Mock DismantlingMethod for testing."""
    
    def __init__(self, key: str, short_name: str, depends_on=None, required_imports=None):
        self.key = key
        self.short_name = short_name
        self.depends_on = depends_on
        self.required_imports = required_imports or []


class TestCyclicDependencies:
    """Test cyclic dependency detection."""
    
    def test_simple_cycle(self):
        """Test detection of a simple A -> B -> A cycle."""
        from network_dismantling.dismantler import check_dependencies
        
        # Create circular dependency: A -> B -> A
        method_a = MockDismantlingMethod("method_a", "Method A")
        method_b = MockDismantlingMethod("method_b", "Method B")
        method_a.depends_on = method_b
        method_b.depends_on = method_a
        
        dismantling_methods = {
            "method_a": method_a,
            "method_b": method_b,
        }
        
        with patch("network_dismantling.dismantling_methods", dismantling_methods):
            with pytest.raises(ValueError, match="Cyclic dependency detected"):
                check_dependencies(["method_a", "method_b"])
    
    def test_indirect_cycle(self):
        """Test detection of A -> B -> C -> A cycle."""
        from network_dismantling.dismantler import check_dependencies
        
        method_a = MockDismantlingMethod("method_a", "Method A")
        method_b = MockDismantlingMethod("method_b", "Method B")
        method_c = MockDismantlingMethod("method_c", "Method C")
        method_a.depends_on = method_b
        method_b.depends_on = method_c
        method_c.depends_on = method_a
        
        dismantling_methods = {
            "method_a": method_a,
            "method_b": method_b,
            "method_c": method_c,
        }
        
        with patch("network_dismantling.dismantling_methods", dismantling_methods):
            with pytest.raises(ValueError, match="Cyclic dependency detected"):
                check_dependencies(["method_a"])
    
    def test_no_cycle(self):
        """Test that linear dependencies work correctly."""
        from network_dismantling.dismantler import check_dependencies
        
        method_a = MockDismantlingMethod("method_a", "Method A")
        method_b = MockDismantlingMethod("method_b", "Method B")
        method_c = MockDismantlingMethod("method_c", "Method C")
        method_c.depends_on = method_b
        method_b.depends_on = method_a
        
        dismantling_methods = {
            "method_a": method_a,
            "method_b": method_b,
            "method_c": method_c,
        }
        
        with patch("network_dismantling.dismantling_methods", dismantling_methods):
            # Should not raise
            result = check_dependencies(["method_c"])
            # Should include all dependencies in correct order
            assert "method_a" in result
            assert "method_b" in result
            assert "method_c" in result
            # Method A should come before B, and B before C
            assert result.index("method_a") < result.index("method_b")
            assert result.index("method_b") < result.index("method_c")
    
    def test_self_reference(self):
        """Test detection of self-referencing dependency."""
        from network_dismantling.dismantler import check_dependencies
        
        method_a = MockDismantlingMethod("method_a", "Method A")
        method_a.depends_on = method_a
        
        dismantling_methods = {
            "method_a": method_a,
        }
        
        with patch("network_dismantling.dismantling_methods", dismantling_methods):
            with pytest.raises(ValueError, match="Cyclic dependency detected"):
                check_dependencies(["method_a"])


class TestDependencyResolution:
    """Test dependency resolution and ordering."""
    
    def test_add_missing_dependency(self):
        """Test that missing dependencies are automatically added."""
        from network_dismantling.dismantler import check_dependencies
        
        method_a = MockDismantlingMethod("method_a", "Method A")
        method_b = MockDismantlingMethod("method_b", "Method B")
        method_b.depends_on = method_a
        
        dismantling_methods = {
            "method_a": method_a,
            "method_b": method_b,
        }
        
        with patch("network_dismantling.dismantling_methods", dismantling_methods):
            # Only request method_b, but method_a should be added
            result = check_dependencies(["method_b"])
            assert "method_a" in result
            assert "method_b" in result
    
    def test_multiple_heuristics_share_dependency(self):
        """Test multiple heuristics depending on the same method."""
        from network_dismantling.dismantler import check_dependencies
        
        method_base = MockDismantlingMethod("method_base", "Base Method")
        method_a = MockDismantlingMethod("method_a", "Method A")
        method_b = MockDismantlingMethod("method_b", "Method B")
        method_a.depends_on = method_base
        method_b.depends_on = method_base
        
        dismantling_methods = {
            "method_base": method_base,
            "method_a": method_a,
            "method_b": method_b,
        }
        
        with patch("network_dismantling.dismantling_methods", dismantling_methods):
            result = check_dependencies(["method_a", "method_b"])
            # Base should be included only once
            assert result.count("method_base") == 1
            # Base should come before both A and B
            base_idx = result.index("method_base")
            assert result.index("method_a") > base_idx
            assert result.index("method_b") > base_idx
    
    def test_deep_dependency_chain(self):
        """Test a deep dependency chain: D -> C -> B -> A."""
        from network_dismantling.dismantler import check_dependencies
        
        method_a = MockDismantlingMethod("method_a", "Method A")
        method_b = MockDismantlingMethod("method_b", "Method B")
        method_c = MockDismantlingMethod("method_c", "Method C")
        method_d = MockDismantlingMethod("method_d", "Method D")
        
        method_b.depends_on = method_a
        method_c.depends_on = method_b
        method_d.depends_on = method_c
        
        dismantling_methods = {
            "method_a": method_a,
            "method_b": method_b,
            "method_c": method_c,
            "method_d": method_d,
        }
        
        with patch("network_dismantling.dismantling_methods", dismantling_methods):
            result = check_dependencies(["method_d"])
            # All dependencies should be added
            assert len(result) == 4
            assert "method_a" in result
            assert "method_b" in result
            assert "method_c" in result
            assert "method_d" in result
            # Check correct ordering
            assert result.index("method_a") < result.index("method_b")
            assert result.index("method_b") < result.index("method_c")
            assert result.index("method_c") < result.index("method_d")
    
    def test_diamond_dependency(self):
        """Test diamond dependency: D depends on B and C, both depend on A."""
        from network_dismantling.dismantler import check_dependencies
        
        method_a = MockDismantlingMethod("method_a", "Method A")
        method_b = MockDismantlingMethod("method_b", "Method B")
        method_c = MockDismantlingMethod("method_c", "Method C")
        method_d = MockDismantlingMethod("method_d", "Method D")
        
        method_b.depends_on = method_a
        method_c.depends_on = method_a
        method_d.depends_on = method_c  # D only explicitly depends on C
        
        dismantling_methods = {
            "method_a": method_a,
            "method_b": method_b,
            "method_c": method_c,
            "method_d": method_d,
        }
        
        with patch("network_dismantling.dismantling_methods", dismantling_methods):
            result = check_dependencies(["method_d", "method_b"])
            # A should appear only once
            assert result.count("method_a") == 1
            # A should come before both B and C
            a_idx = result.index("method_a")
            assert result.index("method_b") > a_idx
            assert result.index("method_c") > a_idx
            assert result.index("method_d") > result.index("method_c")
    
    def test_string_dependency_reference(self):
        """Test dependency specified as string instead of object."""
        from network_dismantling.dismantler import check_dependencies
        
        method_a = MockDismantlingMethod("method_a", "Method A")
        method_b = MockDismantlingMethod("method_b", "Method B")
        method_b.depends_on = "method_a"  # String reference
        
        dismantling_methods = {
            "method_a": method_a,
            "method_b": method_b,
        }
        
        with patch("network_dismantling.dismantling_methods", dismantling_methods):
            result = check_dependencies(["method_b"])
            assert "method_a" in result
            assert "method_b" in result
            assert result.index("method_a") < result.index("method_b")
    
    def test_no_dependencies(self):
        """Test methods with no dependencies."""
        from network_dismantling.dismantler import check_dependencies
        
        method_a = MockDismantlingMethod("method_a", "Method A")
        method_b = MockDismantlingMethod("method_b", "Method B")
        
        dismantling_methods = {
            "method_a": method_a,
            "method_b": method_b,
        }
        
        with patch("network_dismantling.dismantling_methods", dismantling_methods):
            result = check_dependencies(["method_a", "method_b"])
            assert len(result) == 2
            assert "method_a" in result
            assert "method_b" in result
    
    def test_dependency_ordering_preservation(self):
        """Test that the final ordering respects all dependency constraints."""
        from network_dismantling.dismantler import check_dependencies
        
        # Create a complex scenario:
        # base1, base2 have no dependencies
        # mid1 depends on base1
        # mid2 depends on base2
        # top depends on mid1
        method_base1 = MockDismantlingMethod("base1", "Base 1")
        method_base2 = MockDismantlingMethod("base2", "Base 2")
        method_mid1 = MockDismantlingMethod("mid1", "Mid 1")
        method_mid2 = MockDismantlingMethod("mid2", "Mid 2")
        method_top = MockDismantlingMethod("top", "Top")
        
        method_mid1.depends_on = method_base1
        method_mid2.depends_on = method_base2
        method_top.depends_on = method_mid1
        
        dismantling_methods = {
            "base1": method_base1,
            "base2": method_base2,
            "mid1": method_mid1,
            "mid2": method_mid2,
            "top": method_top,
        }
        
        with patch("network_dismantling.dismantling_methods", dismantling_methods):
            result = check_dependencies(["top", "mid2"])
            # Check constraints
            assert result.index("base1") < result.index("mid1")
            assert result.index("mid1") < result.index("top")
            assert result.index("base2") < result.index("mid2")
    
    def test_empty_heuristics_list(self):
        """Test with empty heuristics list."""
        from network_dismantling.dismantler import check_dependencies
        
        dismantling_methods = {}
        
        with patch("network_dismantling.dismantling_methods", dismantling_methods):
            result = check_dependencies([])
            assert result == []
    
    def test_dependency_already_in_list(self):
        """Test when dependency is already in the heuristics list."""
        from network_dismantling.dismantler import check_dependencies
        
        method_a = MockDismantlingMethod("method_a", "Method A")
        method_b = MockDismantlingMethod("method_b", "Method B")
        method_b.depends_on = method_a
        
        dismantling_methods = {
            "method_a": method_a,
            "method_b": method_b,
        }
        
        with patch("network_dismantling.dismantling_methods", dismantling_methods):
            # Both methods already in list
            result = check_dependencies(["method_a", "method_b"])
            assert len(result) == 2
            # A should still come before B
            assert result.index("method_a") < result.index("method_b")


class TestImportValidation:
    """Test required imports validation."""
    
    def test_all_imports_available(self):
        """Test heuristic with all required imports available."""
        from network_dismantling.dismantler import validate_heuristic_imports
        
        method_a = MockDismantlingMethod(
            "method_a",
            "Method A",
            required_imports=["os", "sys"]  # Standard library modules
        )
        
        dismantling_methods = {"method_a": method_a}
        
        with patch("network_dismantling.dismantling_methods", dismantling_methods):
            result = validate_heuristic_imports(["method_a"])
            assert "method_a" in result
    
    def test_missing_imports(self):
        """Test heuristic with missing required imports."""
        from network_dismantling.dismantler import validate_heuristic_imports
        
        method_a = MockDismantlingMethod(
            "method_a",
            "Method A",
            required_imports=["nonexistent_module_xyz"]
        )
        
        dismantling_methods = {"method_a": method_a}
        
        with patch("network_dismantling.dismantling_methods", dismantling_methods):
            result = validate_heuristic_imports(["method_a"])
            # Should be filtered out due to missing imports
            assert "method_a" not in result
    
    def test_mixed_imports(self):
        """Test multiple heuristics with mixed import availability."""
        from network_dismantling.dismantler import validate_heuristic_imports
        
        method_ok = MockDismantlingMethod(
            "method_ok",
            "Method OK",
            required_imports=["os"]
        )
        method_missing = MockDismantlingMethod(
            "method_missing",
            "Method Missing",
            required_imports=["nonexistent_module_xyz"]
        )
        
        dismantling_methods = {
            "method_ok": method_ok,
            "method_missing": method_missing,
        }
        
        with patch("network_dismantling.dismantling_methods", dismantling_methods):
            result = validate_heuristic_imports(["method_ok", "method_missing"])
            assert "method_ok" in result
            assert "method_missing" not in result
    
    def test_no_required_imports(self):
        """Test heuristic without required_imports attribute."""
        from network_dismantling.dismantler import validate_heuristic_imports
        
        method_a = MockDismantlingMethod("method_a", "Method A")
        # No required_imports attribute
        
        dismantling_methods = {"method_a": method_a}
        
        with patch("network_dismantling.dismantling_methods", dismantling_methods):
            result = validate_heuristic_imports(["method_a"])
            # Should pass since no imports are required
            assert "method_a" in result
    
    def test_empty_required_imports(self):
        """Test heuristic with empty required_imports list."""
        from network_dismantling.dismantler import validate_heuristic_imports
        
        method_a = MockDismantlingMethod(
            "method_a",
            "Method A",
            required_imports=[]
        )
        
        dismantling_methods = {"method_a": method_a}
        
        with patch("network_dismantling.dismantling_methods", dismantling_methods):
            result = validate_heuristic_imports(["method_a"])
            assert "method_a" in result


class TestIntegration:
    """Integration tests for the full dependency and import validation pipeline."""
    
    def test_full_pipeline(self):
        """Test the complete flow: import validation -> dependency resolution."""
        from network_dismantling.dismantler import validate_heuristic_imports, check_dependencies
        
        base = MockDismantlingMethod("base", "Base", required_imports=["os"])
        derived = MockDismantlingMethod(
            "derived",
            "Derived",
            required_imports=["sys"],
        )
        derived.depends_on = base
        
        missing = MockDismantlingMethod(
            "missing",
            "Missing",
            required_imports=["nonexistent_module"],
        )
        
        dismantling_methods = {
            "base": base,
            "derived": derived,
            "missing": missing,
        }
        
        with patch("network_dismantling.dismantling_methods", dismantling_methods):
            # First validate imports
            valid = validate_heuristic_imports(["derived", "missing"])
            # Only 'derived' should pass import validation
            assert "derived" in valid
            assert "missing" not in valid
            
            # Then check dependencies
            result = check_dependencies(valid)
            # Should have both base and derived
            assert "base" in result
            assert "derived" in result
            # Base should come first
            assert result.index("base") < result.index("derived")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
