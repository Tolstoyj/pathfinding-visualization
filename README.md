# Pathfinding Algorithm Visualization and Comparison

An interactive visualization tool for comparing different pathfinding algorithms (A*, Greedy Best-First Search, and B* Search) in a grid-based environment.

## Features

- Interactive grid creation with customizable size and obstacle density
- Real-time visualization of pathfinding algorithms
- Performance comparison metrics and analysis
- Support for three algorithms:
  - A* Search (optimal path finding)
  - Greedy Best-First Search (fast, near-optimal paths)
  - B* Search (beam search variant)
- Detailed performance metrics and visualization
- Interactive UI with matplotlib integration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pathfinding-visualization.git
cd pathfinding-visualization
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the interactive visualization:
```bash
python interactive_pathfinding.py
```

### Grid Parameters
- Width and Height: Set the grid dimensions (max 100x100)
- Obstacle Density: Control the percentage of obstacles (0-1)
- Beam Width: Adjust the beam width for B* search
- Adaptive Beam: Toggle adaptive beam width adjustment

### Features
1. Generate new grids with custom parameters
2. Run individual algorithms
3. Compare all algorithms simultaneously
4. View detailed performance metrics
5. Interactive visualization with zoom and pan
6. Performance comparison graphs

## Algorithm Details

### A* Search
- Optimal pathfinding algorithm
- Balances path cost and heuristic
- Guarantees shortest path
- Higher exploration in complex scenarios

### Greedy Best-First Search
- Heuristic-based pathfinding
- Fast execution with minimal exploration
- Near-optimal paths in many cases
- Efficient node expansion

### B* Search (Beam Search)
- Limited-width search algorithm
- Adaptive beam width based on obstacle density
- Memory-efficient for large grids
- Trade-off between exploration and path quality

## Performance Metrics

The tool provides detailed performance metrics:
- Path Length: Number of steps in the found path
- Nodes Explored: Total nodes visited
- Nodes Expanded: Total node expansions
- Path Efficiency: Relative to the best path found
- Exploration Rate: Nodes explored per path step

## Example Results

```
A* Analysis:
├─ Path Length: 129
├─ Nodes Explored: 173
├─ Nodes Expanded: 183
├─ Path Efficiency: 100.0%
└─ Exploration Rate: 1.34 nodes/step

Greedy Analysis:
├─ Path Length: 132
├─ Nodes Explored: 133
├─ Nodes Expanded: 133
├─ Path Efficiency: 97.7%
└─ Exploration Rate: 1.01 nodes/step

B* Analysis:
├─ Path Length: 151
├─ Nodes Explored: 290
├─ Nodes Expanded: 940
├─ Path Efficiency: 85.4%
└─ Exploration Rate: 1.92 nodes/step
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Python, Tkinter, and Matplotlib
- Inspired by pathfinding algorithm visualizations
- Thanks to all contributors and users

## Screenshots

[Add screenshots of the application here] 