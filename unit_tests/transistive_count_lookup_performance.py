import unittest
import time
import random
from unittest.mock import Mock, patch
import numpy as np
from ..transistive_count import FinalCounter

class TestFinalCounterPerformance(unittest.TestCase):
    def generate_mock_annotations(self, size):
        """Generate mock annotations of specified size."""
        return [
            {'id': i, 'category_id': random.randint(1, 100)}
            for i in range(size)
        ]

    def benchmark_build_lookup(self, size):
        """Benchmark lookup table building with different sizes."""
        # Create mock COCO helper with generated annotations
        mock_coco = Mock()
        mock_coco.dataset = {
            'annotations': self.generate_mock_annotations(size)
        }
        
        with patch('coco_help.COCOAnnotationHelper') as MockCOCOHelper:
            MockCOCOHelper.return_value.coco = mock_coco
            
            # Time the initialization
            start_time = time.perf_counter()
            counter = FinalCounter([], "dummy_path.json")
            end_time = time.perf_counter()
            
            return end_time - start_time, len(counter.table_dict)

    def test_lookup_table_performance(self):
        """Test performance with different dataset sizes."""
        sizes = [100, 1000, 10000, 100000]
        results = {}
        
        for size in sizes:
            time_taken, table_size = self.benchmark_build_lookup(size)
            results[size] = {
                'time': time_taken,
                'size': table_size
            }
            print(f"\nDataset size: {size}")
            print(f"Build time: {time_taken:.4f} seconds")
            print(f"Table size: {table_size} entries")
            
        # Verify scaling is roughly linear or better
        times = [results[size]['time'] for size in sizes]
        ratios = [times[i+1]/times[i] for i in range(len(times)-1)]
        
        # Check if performance scales reasonably (should be sub-quadratic)
        for i, ratio in enumerate(ratios):
            self.assertLess(ratio, 20.0, 
                          f"Performance degradation too high between {sizes[i]} and {sizes[i+1]}")

if __name__ == '__main__':
    unittest.main()