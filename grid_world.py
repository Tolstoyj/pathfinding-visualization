"""
Grid World Environment for Pathfinding Algorithms

This module implements a 2D grid environment with obstacles for testing and
visualizing different pathfinding algorithms. The grid can be customized with
different sizes and obstacle densities.

Classes:
    GridWorld: Main class representing the grid environment
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Set

class GridWorld:
    """
    A 2D grid environment for pathfinding algorithms.
    
    The grid consists of free cells (0) and obstacles (1). The start point is
    always at (0,0) and the goal is at the opposite corner (height-1, width-1).
    
    Attributes:
        width (int): Width of the grid
        height (int): Height of the grid
        grid (np.ndarray): 2D array representing the grid (0: free, 1: obstacle)
        start (Tuple[int, int]): Starting position (0,0)
        goal (Tuple[int, int]): Goal position (height-1, width-1)
    """

    def __init__(self, width: int = 20, height: int = 20, obstacle_density: float = 0.3):
        """
        Initialize the grid world with given dimensions and obstacle density.
        
        Args:
            width (int): Width of the grid (default: 20)
            height (int): Height of the grid (default: 20)
            obstacle_density (float): Proportion of cells to be obstacles (default: 0.3)
        """
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        self.start = (0, 0)
        self.goal = (height-1, width-1)
        self._generate_obstacles(obstacle_density)
    
    def _generate_obstacles(self, density: float):
        """
        Generate random obstacles in the grid.
        
        Args:
            density (float): Proportion of cells to be obstacles (0-1)
        """
        n_obstacles = int(self.width * self.height * density)
        obstacles = set()
        while len(obstacles) < n_obstacles:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if (y, x) != self.start and (y, x) != self.goal:
                obstacles.add((y, x))
                self.grid[y, x] = 1

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid neighboring positions for a given position.
        
        Args:
            pos (Tuple[int, int]): Current position (y, x)
        
        Returns:
            List[Tuple[int, int]]: List of valid neighbor positions
        """
        y, x = pos
        neighbors = []
        # Include diagonal movements
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            new_y, new_x = y + dy, x + dx
            if (0 <= new_y < self.height and 
                0 <= new_x < self.width and 
                self.grid[new_y, new_x] == 0):
                neighbors.append((new_y, new_x))
        return neighbors

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Calculate Manhattan distance between two points.
        
        Args:
            pos1 (Tuple[int, int]): First position (y1, x1)
            pos2 (Tuple[int, int]): Second position (y2, x2)
        
        Returns:
            float: Manhattan distance between the positions
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            pos1 (Tuple[int, int]): First position (y1, x1)
            pos2 (Tuple[int, int]): Second position (y2, x2)
        
        Returns:
            float: Euclidean distance between the positions
        """
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def visualize(self, path: List[Tuple[int, int]] = None, 
                 explored: Set[Tuple[int, int]] = None,
                 title: str = "Grid World",
                 ax = None):
        """
        Visualize the grid, path, and explored nodes.
        
        Args:
            path (List[Tuple[int, int]], optional): Path to visualize
            explored (Set[Tuple[int, int]], optional): Set of explored nodes
            title (str, optional): Title for the plot
            ax (matplotlib.axes.Axes, optional): Axes to plot on
        """
        # Use provided axes or current axes
        if ax is None:
            ax = plt.gca()
        
        # Clear current axes
        ax.clear()
        
        # Create custom colormap for obstacles
        obstacle_cmap = plt.cm.Greys(np.linspace(0, 1, 2))
        obstacle_cmap[1] = [0.3, 0.3, 0.3, 1]  # Darker gray for obstacles
        
        # Plot the grid
        ax.imshow(self.grid, cmap=plt.matplotlib.colors.ListedColormap(obstacle_cmap), alpha=0.3)
        
        # Add grid lines
        ax.grid(True, color='gray', alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Plot explored nodes with a gradient based on exploration order
        if explored:
            explored_y, explored_x = zip(*explored)
            scatter = ax.scatter(explored_x, explored_y, 
                               c=range(len(explored)), cmap='Blues',
                               alpha=0.3, s=50)
            
            # Add colorbar for exploration order
            if len(explored) > 100:  # Only add colorbar for significant exploration
                plt.colorbar(scatter, ax=ax, label='Exploration Order')

        # Plot the path with a gradient color
        if path:
            path_y, path_x = zip(*path)
            points = np.array([path_x, path_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Create a gradient color for the path
            norm = plt.Normalize(0, len(path))
            lc = plt.matplotlib.collections.LineCollection(
                segments, cmap='viridis',
                norm=norm, linewidth=3, alpha=0.8
            )
            lc.set_array(np.linspace(0, len(path), len(path)))
            ax.add_collection(lc)
            
            # Add colorbar for path progression
            if len(path) > 20:  # Only add colorbar for longer paths
                plt.colorbar(lc, ax=ax, label='Path Progression')

        # Plot start and goal with distinctive markers
        ax.plot(self.start[1], self.start[0], 
                marker='*', color='#2ecc71', 
                markersize=15, label='Start',
                markeredgecolor='white', markeredgewidth=1.5)
        
        ax.plot(self.goal[1], self.goal[0], 
                marker='*', color='#e74c3c',
                markersize=15, label='Goal',
                markeredgecolor='white', markeredgewidth=1.5)

        # Customize the plot
        ax.set_title(title, pad=10, fontsize=10, fontweight='bold')
        
        # Remove axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add legend if this is not part of a subplot
        if len(ax.figure.axes) == 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Ensure proper layout
        ax.set_aspect('equal') 