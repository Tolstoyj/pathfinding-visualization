from grid_world import GridWorld
from pathfinding import PathFinder
import matplotlib.pyplot as plt
import numpy as np

def run_comparison(width: int = 40, height: int = 40, obstacle_density: float = 0.35):
    # Create grid world
    grid = GridWorld(width=width, height=height, obstacle_density=obstacle_density)
    pathfinder = PathFinder(grid)
    
    # Run all algorithms
    algorithms = {
        "A*": pathfinder.astar_search,
        "Greedy Best-First": pathfinder.greedy_search,
        "Beam Search (B*)": lambda: pathfinder.beam_search(beam_width=5)
    }
    
    results = {}
    for name, algo in algorithms.items():
        path, explored, nodes = algo()
        results[name] = {
            "path": path,
            "explored": explored,
            "nodes_expanded": nodes,
            "path_length": len(path) if path else float('inf')
        }
        
        # Visualize each algorithm's result
        grid.visualize(path, explored, f"{name} Search\nNodes Expanded: {nodes}, Path Length: {len(path) if path else 'No path'}")
    
    # Print comparison metrics
    print("\nAlgorithm Comparison:")
    print("-" * 50)
    for name, data in results.items():
        print(f"{name}:")
        print(f"  Nodes Expanded: {data['nodes_expanded']}")
        print(f"  Path Length: {data['path_length']}")
        print(f"  Total Explored: {len(data['explored'])}")
        print("-" * 50)

if __name__ == "__main__":
    # Run comparison with a larger, more complex grid
    run_comparison() 