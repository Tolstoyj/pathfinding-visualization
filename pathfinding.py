"""
Pathfinding Algorithms Implementation

This module implements three different pathfinding algorithms:
1. A* Search - Optimal pathfinding using both cost and heuristic
2. Greedy Best-First Search - Fast pathfinding using only heuristic
3. B* Search (Beam Search) - Memory-efficient search with limited branching

Each algorithm provides path finding capabilities in a grid-based environment
with performance metrics including nodes explored and expanded.
"""

from typing import List, Tuple, Set, Dict
import heapq
from grid_world import GridWorld
import numpy as np

class PathFinder:
    """
    A class implementing various pathfinding algorithms.
    
    This class provides implementations of multiple pathfinding algorithms
    that can be used to find paths in a GridWorld environment. Each algorithm
    returns the found path along with performance metrics.
    
    Attributes:
        grid (GridWorld): The grid environment to perform pathfinding in
    """

    def __init__(self, grid_world: GridWorld):
        """
        Initialize the PathFinder with a grid world.
        
        Args:
            grid_world (GridWorld): The grid environment to perform pathfinding in
        """
        self.grid = grid_world

    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from start to goal using the came_from dictionary.
        
        Args:
            came_from (Dict): Dictionary mapping each position to its predecessor
            current (Tuple[int, int]): Current position (usually the goal)
        
        Returns:
            List[Tuple[int, int]]: The reconstructed path from start to goal
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def astar_search(self) -> Tuple[List[Tuple[int, int]], Set[Tuple[int, int]], int]:
        """
        Perform A* Search algorithm.
        
        A* Search combines actual path cost with a heuristic estimate to find
        the optimal path. It guarantees the shortest path when using an
        admissible heuristic.
        
        Returns:
            Tuple containing:
            - List[Tuple[int, int]]: The found path (empty if no path exists)
            - Set[Tuple[int, int]]: Set of explored nodes
            - int: Number of nodes expanded
        """
        start = self.grid.start
        goal = self.grid.goal
        
        frontier = [(0, start)]  # Priority queue of (f_score, position)
        came_from = {}
        cost_so_far = {start: 0}  # g_score
        explored = set()
        nodes_expanded = 0

        while frontier:
            _, current = heapq.heappop(frontier)
            
            if current == goal:
                return self._reconstruct_path(came_from, goal), explored, nodes_expanded
            
            explored.add(current)
            nodes_expanded += 1

            for next_pos in self.grid.get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    # f_score = g_score + heuristic
                    priority = new_cost + self.grid.manhattan_distance(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        return [], explored, nodes_expanded

    def greedy_search(self) -> Tuple[List[Tuple[int, int]], Set[Tuple[int, int]], int]:
        """
        Perform Greedy Best-First Search algorithm.
        
        Greedy Best-First Search uses only the heuristic to guide the search,
        making it faster but not guaranteed to find the optimal path.
        
        Returns:
            Tuple containing:
            - List[Tuple[int, int]]: The found path (empty if no path exists)
            - Set[Tuple[int, int]]: Set of explored nodes
            - int: Number of nodes expanded
        """
        start = self.grid.start
        goal = self.grid.goal
        
        frontier = [(self.grid.manhattan_distance(start, goal), start)]
        came_from = {}
        explored = set()
        nodes_expanded = 0

        while frontier:
            _, current = heapq.heappop(frontier)
            
            if current == goal:
                return self._reconstruct_path(came_from, goal), explored, nodes_expanded
            
            explored.add(current)
            nodes_expanded += 1

            for next_pos in self.grid.get_neighbors(current):
                if next_pos not in explored and next_pos not in [pos for _, pos in frontier]:
                    priority = self.grid.manhattan_distance(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        return [], explored, nodes_expanded

    def beam_search(self, beam_width: int = 5, adaptive: bool = True) -> Tuple[List[Tuple[int, int]], Set[Tuple[int, int]], int]:
        """
        Perform B* (Beam) Search algorithm.
        
        B* Search is a variant of best-first search that limits the number of
        nodes expanded at each level. It can be more memory-efficient but may
        miss optimal paths.
        
        Args:
            beam_width (int): Maximum number of nodes to consider at each level
            adaptive (bool): Whether to adjust beam width based on obstacle density
        
        Returns:
            Tuple containing:
            - List[Tuple[int, int]]: The found path (empty if no path exists)
            - Set[Tuple[int, int]]: Set of explored nodes
            - int: Number of nodes expanded
        """
        start = self.grid.start
        goal = self.grid.goal
        
        # Initialize with a larger beam for dense obstacle grids
        if adaptive:
            obstacle_count = sum(sum(row) for row in self.grid.grid)
            grid_size = self.grid.width * self.grid.height
            obstacle_density = obstacle_count / grid_size
            
            # Adaptively adjust beam width based on obstacle density
            if obstacle_density > 0.4:
                beam_width = max(beam_width * 2, 10)
            elif obstacle_density > 0.3:
                beam_width = max(beam_width * 1.5, 8)

        frontier = [(self.grid.manhattan_distance(start, goal), 0, start)]  # Added step count
        came_from = {}
        explored = set()
        nodes_expanded = 0
        best_distance = float('inf')
        best_node = None
        steps_without_improvement = 0

        while frontier:
            next_frontier = []
            
            # Process current frontier
            for _ in range(min(beam_width, len(frontier))):
                if not frontier:
                    break
                    
                _, steps, current = heapq.heappop(frontier)
                
                if current == goal:
                    return self._reconstruct_path(came_from, goal), explored, nodes_expanded
                
                explored.add(current)
                nodes_expanded += 1

                # Check if this is the closest we've gotten to the goal
                current_distance = self.grid.manhattan_distance(current, goal)
                if current_distance < best_distance:
                    best_distance = current_distance
                    best_node = current
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1

                # Expand neighbors
                for next_pos in self.grid.get_neighbors(current):
                    if next_pos not in explored:
                        priority = self.grid.manhattan_distance(next_pos, goal)
                        # Add some randomness to break ties and increase exploration
                        priority += np.random.uniform(0, 0.1)
                        next_frontier.append((priority, steps + 1, next_pos))
                        came_from[next_pos] = current
            
            # If we're stuck, increase beam width temporarily
            if steps_without_improvement > beam_width * 2:
                beam_width = min(beam_width + 2, 20)
                steps_without_improvement = 0
            
            # Keep only the best beam_width nodes
            next_frontier.sort()
            frontier = next_frontier[:beam_width]

            # Early stopping if we're not making progress
            if not frontier or steps_without_improvement > beam_width * 4:
                break
        
        # If no path to goal, try to return the path to the closest point reached
        if best_node and best_node != start:
            return self._reconstruct_path(came_from, best_node), explored, nodes_expanded
        
        return [], explored, nodes_expanded 