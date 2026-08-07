import unittest
from unittest.mock import Mock, patch
import networkx as nx
from ..transistive_count import FinalCounter

class TestFinalCounter(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock COCO helper since we don't need actual JSON processing
        self.mock_coco_helper = Mock()
        
        # Common test data
        self.table_dict = {
            1: 100,  # annotation_id: category_id
            2: 100,
            3: 200,
            4: 200,
            5: 300
        }
        
    def test_init(self):
        """Test initialization of FinalCounter."""
        overlaps = [(1, 2)]
        counter = FinalCounter(overlaps, "dummy_path.json", self.table_dict)
        
        self.assertEqual(counter.overlaps, overlaps)
        self.assertEqual(counter.table_dict, self.table_dict)
        self.assertEqual(dict(counter.count_dict), {})

    def test_count_helper(self):
        """Test the _count_helper method."""
        counter = FinalCounter([], "dummy_path.json", self.table_dict)
        
        # Test counting single annotation
        counter._count_helper(1)
        self.assertEqual(counter.count_dict[100], 1)
        
        # Test counting another annotation with same category
        counter._count_helper(2)
        self.assertEqual(counter.count_dict[100], 2)
        
        # Test counting annotation with different category
        counter._count_helper(3)
        self.assertEqual(counter.count_dict[200], 1)
        
        # Test counting non-existent annotation ID
        counter._count_helper(999)
        self.assertNotIn(999, counter.count_dict)

    def test_filter_dataset_annotation_id_list(self):
        """Test the _filter_dataset_annotation_id_list method."""
        counter = FinalCounter([], "dummy_path.json", self.table_dict)
        
        # Test filtering with empty remove list
        filtered = counter._filter_dataset_annotation_id_list(set())
        self.assertEqual(set(filtered), set(self.table_dict.keys()))
        
        # Test filtering with some annotations to remove
        to_remove = {1, 2}
        filtered = counter._filter_dataset_annotation_id_list(to_remove)
        self.assertEqual(set(filtered), {3, 4, 5})

    def test_count_annotations_no_overlaps(self):
        """Test counting annotations when there are no overlaps."""
        counter = FinalCounter([], "dummy_path.json", self.table_dict)
        
        result = counter.count_annotations()
        
        # Each category should be counted for each annotation
        expected = {
            100: 2,  # Two annotations (1, 2)
            200: 2,  # Two annotations (3, 4)
            300: 1   # One annotation (5)
        }
        self.assertEqual(result, expected)

    def test_count_annotations_with_overlaps(self):
        """Test counting annotations with overlapping regions."""
        # Create overlaps: (1,2) overlap, (3,4) overlap
        overlaps = [(1, 2), (3, 4)]
        counter = FinalCounter(overlaps, "dummy_path.json", self.table_dict)
        
        result = counter.count_annotations()
        
        # Each overlapping group should be counted once
        expected = {
            100: 1,  # One count for overlapping pair (1,2)
            200: 1,  # One count for overlapping pair (3,4)
            300: 1   # One count for standalone annotation 5
        }
        self.assertEqual(result, expected)

    def test_count_annotations_complex_overlaps(self):
        """Test counting annotations with complex overlapping patterns."""
        # Create complex overlap pattern: (1,2), (2,3), (3,4) forming a chain
        overlaps = [(1, 2), (2, 3), (3, 4)]
        counter = FinalCounter(overlaps, "dummy_path.json", self.table_dict)
        
        result = counter.count_annotations()
        
        # All connected annotations should be counted as one group
        expected = {
            100: 1,  # One count for annotations 1,2 in the chain
            200: 1,  # One count for annotations 3,4 in the chain
            300: 1   # One count for standalone annotation 5
        }
        self.assertEqual(result, expected)

    def test_empty_input(self):
        """Test handling of empty input data."""
        counter = FinalCounter([], "dummy_path.json", {})
        result = counter.count_annotations()
        self.assertEqual(result, {})

    def test_invalid_annotation_ids(self):
        """Test handling of invalid annotation IDs in overlaps."""
        overlaps = [(1, 999)]  # 999 is not in table_dict
        counter = FinalCounter(overlaps, "dummy_path.json", self.table_dict)
        
        result = counter.count_annotations()
        self.assertEqual(result[100], 1)  # Valid annotation should still be counted

    @patch('networkx.connected_components')
    def test_large_graph_performance(self, mock_connected_components):
        """Test performance with large number of connected components."""
        # Mock connected_components to return predetermined groups
        mock_connected_components.return_value = [
            {1, 2},
            {3, 4},
            {5}
        ]
        
        counter = FinalCounter([(1,2), (3,4)], "dummy_path.json", self.table_dict)
        result = counter.count_annotations()
        
        expected = {
            100: 1,  # Group (1,2)
            200: 1,  # Group (3,4)
            300: 1   # Standalone 5
        }
        self.assertEqual(result, expected)


    
if __name__ == '__main__':
    unittest.main()