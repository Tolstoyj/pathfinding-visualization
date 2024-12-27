import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
# Force Agg backend before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from grid_world import GridWorld
from pathfinding import PathFinder

class PathfindingUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pathfinding Algorithm Visualization")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Prevent matplotlib from opening windows
        plt.ioff()
        
        # Variables for grid parameters
        self.width_var = tk.StringVar(value="50")
        self.height_var = tk.StringVar(value="50")
        self.obstacle_density_var = tk.StringVar(value="0.35")
        self.beam_width_var = tk.StringVar(value="5")
        self.adaptive_beam_var = tk.BooleanVar(value=True)
        
        self.grid = None
        self.pathfinder = None
        self.current_figure = None
        self.canvas = None
        self.toolbar = None
        self.last_results = {}

        # Create initial figure
        self.current_figure = Figure(figsize=(8, 8))
        
        self._create_ui()
        self._setup_styles()
        
        # Configure window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        """Handle window closing event"""
        try:
            plt.close('all')  # Close all matplotlib figures
            self.root.quit()
            self.root.destroy()
        except:
            pass

    def _setup_styles(self):
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Subtitle.TLabel', font=('Helvetica', 12))
        style.configure('Info.TLabel', font=('Helvetica', 10))
        
        # Configure button styles
        style.configure('Action.TButton', 
                       font=('Helvetica', 11),
                       padding=5)
        style.configure('Primary.TButton',
                       font=('Helvetica', 11, 'bold'),
                       padding=8)

    def _create_ui(self):
        # Main container with padding
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left panel for controls (1/4 of width)
        control_panel = ttk.Frame(main_container)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Right panel for visualization (3/4 of width)
        self.viz_panel = ttk.Frame(main_container)
        self.viz_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._create_control_panel(control_panel)
        self._create_visualization_panel(self.viz_panel)

    def _create_control_panel(self, parent):
        # Title
        title = ttk.Label(parent, text="Control Panel", style='Title.TLabel')
        title.pack(pady=(0, 20))

        # Grid parameters section
        param_frame = ttk.LabelFrame(parent, text="Grid Parameters", padding="10")
        param_frame.pack(fill=tk.X, pady=(0, 10))

        # Grid size frame
        size_frame = ttk.Frame(param_frame)
        size_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(size_frame, text="Width:", style='Info.TLabel').pack(side=tk.LEFT)
        ttk.Entry(size_frame, textvariable=self.width_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(size_frame, text="Height:", style='Info.TLabel').pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(size_frame, textvariable=self.height_var, width=8).pack(side=tk.LEFT, padx=5)

        # Density frame
        density_frame = ttk.Frame(param_frame)
        density_frame.pack(fill=tk.X, pady=5)
        ttk.Label(density_frame, text="Obstacle Density (0-1):", style='Info.TLabel').pack(side=tk.LEFT)
        ttk.Entry(density_frame, textvariable=self.obstacle_density_var, width=8).pack(side=tk.LEFT, padx=5)

        # Beam width frame
        beam_frame = ttk.Frame(param_frame)
        beam_frame.pack(fill=tk.X, pady=5)
        ttk.Label(beam_frame, text="Beam Width (B*):", style='Info.TLabel').pack(side=tk.LEFT)
        ttk.Entry(beam_frame, textvariable=self.beam_width_var, width=8).pack(side=tk.LEFT, padx=5)

        # Beam search settings
        beam_frame = ttk.LabelFrame(param_frame, text="B* Settings", padding="5")
        beam_frame.pack(fill=tk.X, pady=5)
        
        beam_width_frame = ttk.Frame(beam_frame)
        beam_width_frame.pack(fill=tk.X, pady=2)
        ttk.Label(beam_width_frame, text="Base Beam Width:", style='Info.TLabel').pack(side=tk.LEFT)
        ttk.Entry(beam_width_frame, textvariable=self.beam_width_var, width=8).pack(side=tk.LEFT, padx=5)
        
        ttk.Checkbutton(beam_frame, text="Adaptive Beam Width", 
                       variable=self.adaptive_beam_var).pack(fill=tk.X)

        # Statistics frame
        self.stats_frame = ttk.LabelFrame(parent, text="Current Grid Stats", padding="10")
        self.stats_frame.pack(fill=tk.X, pady=(0, 10))
        self.stats_label = ttk.Label(self.stats_frame, text="No grid generated yet", 
                                   style='Info.TLabel', wraplength=250)
        self.stats_label.pack(fill=tk.X)

        # Create Grid button
        ttk.Button(parent, text="Generate New Grid", 
                  command=self._create_grid,
                  style='Primary.TButton').pack(fill=tk.X, pady=10)

        # Algorithm selection
        algo_frame = ttk.LabelFrame(parent, text="Algorithms", padding="10")
        algo_frame.pack(fill=tk.X, pady=(0, 10))

        for algo_name, display_name in [
            ("A*", "Run A* Search"),
            ("Greedy", "Run Greedy Best-First"),
            ("B*", "Run B* Search")
        ]:
            ttk.Button(algo_frame, text=display_name,
                      command=lambda n=algo_name: self._run_algorithm(n),
                      style='Action.TButton').pack(fill=tk.X, pady=2)

        ttk.Button(algo_frame, text="Compare All Algorithms",
                  command=self._compare_all,
                  style='Primary.TButton').pack(fill=tk.X, pady=(10, 2))

        # Results section
        results_frame = ttk.LabelFrame(parent, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Results text with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.results_text = tk.Text(text_frame, height=12, width=35,
                                  yscrollcommand=scrollbar.set,
                                  font=('Courier', 10))
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_text.yview)

    def _create_visualization_panel(self, parent):
        # Title
        title = ttk.Label(parent, text="Visualization", style='Title.TLabel')
        title.pack(pady=(0, 10))

        # Frame for matplotlib figure
        self.fig_frame = ttk.Frame(parent)
        self.fig_frame.pack(fill=tk.BOTH, expand=True)

    def _validate_params(self):
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            density = float(self.obstacle_density_var.get())
            beam_width = int(self.beam_width_var.get())

            if width <= 0 or height <= 0:
                raise ValueError("Width and height must be positive")
            if width > 100 or height > 100:
                raise ValueError("Maximum grid size is 100x100")
            if density < 0 or density >= 1:
                raise ValueError("Obstacle density must be between 0 and 1")
            if beam_width <= 0:
                raise ValueError("Beam width must be positive")

            return True
        except ValueError as e:
            messagebox.showerror("Invalid Parameters", str(e))
            return False

    def _create_grid(self):
        if not self._validate_params():
            return

        try:
            self.grid = GridWorld(
                width=int(self.width_var.get()),
                height=int(self.height_var.get()),
                obstacle_density=float(self.obstacle_density_var.get())
            )
            self.pathfinder = PathFinder(self.grid)
            self._display_grid()
            
            # Update statistics
            obstacle_count = sum(sum(row) for row in self.grid.grid)
            grid_size = self.grid.width * self.grid.height
            density = obstacle_count / grid_size
            
            stats_text = f"Grid Size: {self.grid.width}x{self.grid.height}\n"
            stats_text += f"Obstacles: {obstacle_count} ({density:.1%})\n"
            stats_text += f"Free Cells: {grid_size - obstacle_count} ({1-density:.1%})"
            self.stats_label.config(text=stats_text)
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "New grid created successfully!\n")
            self.results_text.insert(tk.END, stats_text + "\n")
            self.results_text.see(tk.END)
            
            # Reset last results
            self.last_results = {}
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create grid: {str(e)}")

    def _display_grid(self, path=None, explored=None, title="Grid World"):
        """Display grid with improved error handling and cleanup"""
        try:
            # Clear the current figure
            self.current_figure.clear()
            ax = self.current_figure.add_subplot(111)
            
            # Draw on the figure
            if self.grid:
                self.grid.visualize(path, explored, title, ax)
            
            # Update canvas
            if self.canvas:
                self.canvas.draw()
            else:
                self.canvas = FigureCanvasTkAgg(self.current_figure, self.fig_frame)
                self.canvas.draw()
                
                # Add toolbar if it doesn't exist
                if not self.toolbar:
                    self.toolbar = NavigationToolbar2Tk(self.canvas, self.fig_frame)
                    self.toolbar.update()
                
                self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("Visualization Error", f"Failed to display grid: {str(e)}")

    def _run_algorithm(self, algo_name):
        if not self.grid or not self.pathfinder:
            messagebox.showerror("Error", "Please create a grid first!")
            return

        try:
            self.root.config(cursor="watch")
            self.root.update()

            if algo_name == "A*":
                path, explored, nodes = self.pathfinder.astar_search()
            elif algo_name == "Greedy":
                path, explored, nodes = self.pathfinder.greedy_search()
            else:  # B*
                path, explored, nodes = self.pathfinder.beam_search(
                    beam_width=int(self.beam_width_var.get()),
                    adaptive=self.adaptive_beam_var.get()
                )

            self.last_results[algo_name] = {
                'path': path,
                'explored': explored,
                'nodes': nodes
            }

            self._display_grid(path, explored, f"{algo_name} Search Results")
            self._update_results(algo_name, path, explored, nodes)
            
            # Add comparison with other algorithms if available
            if len(self.last_results) > 1:
                self._show_comparison()
            
            self.root.config(cursor="")
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("Error", f"Algorithm failed: {str(e)}")

    def _show_comparison(self):
        """Show comparison between algorithms that have been run"""
        self.results_text.insert(tk.END, "\nComparison Summary:\n")
        self.results_text.insert(tk.END, "═" * 40 + "\n")
        
        # Find the best path length
        path_lengths = {name: len(data['path']) if data['path'] else float('inf')
                       for name, data in self.last_results.items()}
        best_length = min(path_lengths.values())
        
        for name, data in self.last_results.items():
            path_length = len(data['path']) if data['path'] else float('inf')
            relative_efficiency = (best_length / path_length * 100) if path_length else 0
            
            self.results_text.insert(tk.END, f"{name}:\n")
            self.results_text.insert(tk.END, f"├─ Path Length: {path_length if path_length != float('inf') else 'No path'}\n")
            self.results_text.insert(tk.END, f"├─ Relative Efficiency: {relative_efficiency:.1f}%\n")
            self.results_text.insert(tk.END, f"└─ Exploration Rate: {(len(data['explored'])/len(data['path'])) if data['path'] else 'N/A'}\n")
            self.results_text.insert(tk.END, "─" * 40 + "\n")
        
        self.results_text.see(tk.END)

    def _compare_all(self):
        """Compare all algorithms with improved error handling"""
        if not self.grid or not self.pathfinder:
            messagebox.showerror("Error", "Please create a grid first!")
            return

        try:
            self.root.config(cursor="watch")
            self.root.update()
            
            # Clear previous results
            self.results_text.delete(1.0, tk.END)
            
            algorithms = {
                "A*": self.pathfinder.astar_search,
                "Greedy": self.pathfinder.greedy_search,
                "B*": lambda: self.pathfinder.beam_search(
                    beam_width=int(self.beam_width_var.get()),
                    adaptive=self.adaptive_beam_var.get()
                )
            }

            results = []
            for name, algo in algorithms.items():
                path, explored, nodes = algo()
                results.append((name, path, explored, nodes))
                self._update_results(name, path, explored, nodes)

            # Create new figure for comparison
            self.current_figure.clear()
            gs = self.current_figure.add_gridspec(2, 3, height_ratios=[3, 1])
            
            # Plot paths in the top row
            for i, (name, path, explored, nodes) in enumerate(results):
                ax = self.current_figure.add_subplot(gs[0, i])
                self.grid.visualize(path, explored, f"{name} Search", ax)
                ax.set_title(f"{name}\nNodes: {nodes}\nPath: {len(path) if path else 'No path'}")
            
            # Plot performance comparison in bottom row
            ax_metrics = self.current_figure.add_subplot(gs[1, :])
            self._plot_performance_comparison(results, ax_metrics)
            
            self.current_figure.tight_layout()
            
            if self.canvas:
                self.canvas.draw()
            
            # Show detailed comparison
            self._show_detailed_comparison(results)
            
            self.root.config(cursor="")
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("Comparison Error", f"Failed to compare algorithms: {str(e)}")

    def _plot_performance_comparison(self, results, ax):
        """Plot performance metrics comparison"""
        # Extract metrics
        names = [r[0] for r in results]
        path_lengths = [len(r[1]) if r[1] else 0 for r in results]
        explored_counts = [len(r[2]) for r in results]
        nodes_expanded = [r[3] for r in results]
        
        # Calculate efficiency metrics
        min_path = min([l for l in path_lengths if l > 0]) if any(path_lengths) else 1
        path_efficiency = [min_path/l * 100 if l > 0 else 0 for l in path_lengths]
        exploration_efficiency = [l/e * 100 if e > 0 else 0 for l, e in zip(path_lengths, explored_counts)]
        
        # Set up bar positions
        x = np.arange(len(names))
        width = 0.2
        
        # Create grouped bar chart
        ax.bar(x - width, path_lengths, width, label='Path Length', color='#2ecc71', alpha=0.7)
        ax.bar(x, explored_counts, width, label='Nodes Explored', color='#3498db', alpha=0.7)
        ax.bar(x + width, nodes_expanded, width, label='Nodes Expanded', color='#e74c3c', alpha=0.7)
        
        # Add efficiency line plots
        ax2 = ax.twinx()
        ax2.plot(x, path_efficiency, 'g--', label='Path Efficiency %', linewidth=2, marker='o')
        ax2.plot(x, exploration_efficiency, 'b--', label='Exploration Efficiency %', linewidth=2, marker='s')
        
        # Customize the plot
        ax.set_ylabel('Counts')
        ax2.set_ylabel('Efficiency %')
        ax.set_title('Algorithm Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        
        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()

    def _show_detailed_comparison(self, results):
        """Show detailed comparison metrics in the results text area"""
        self.results_text.insert(tk.END, "\nDetailed Performance Analysis:\n")
        self.results_text.insert(tk.END, "═" * 40 + "\n\n")
        
        # Find best metrics
        valid_paths = [(r[0], len(r[1])) for r in results if r[1]]
        if valid_paths:
            best_path_length = min(p[1] for p in valid_paths)
            best_path_algo = min(valid_paths, key=lambda x: x[1])[0]
        else:
            best_path_length = float('inf')
            best_path_algo = "None"
            
        min_explored = min(len(r[2]) for r in results)
        min_expanded = min(r[3] for r in results)
        
        # Show summary
        self.results_text.insert(tk.END, "Best Performers:\n")
        self.results_text.insert(tk.END, f"├─ Shortest Path: {best_path_algo} ({best_path_length} steps)\n")
        self.results_text.insert(tk.END, f"├─ Least Exploration: {min_explored} nodes\n")
        self.results_text.insert(tk.END, f"└─ Most Efficient: {min_expanded} expansions\n\n")
        
        # Show detailed metrics for each algorithm
        for name, path, explored, nodes in results:
            path_length = len(path) if path else 0
            efficiency = (best_path_length / path_length * 100) if path_length > 0 else 0
            exploration_rate = (len(explored) / path_length if path_length > 0 else float('inf'))
            
            self.results_text.insert(tk.END, f"{name} Analysis:\n")
            self.results_text.insert(tk.END, f"├─ Path Length: {path_length if path_length > 0 else 'No path'}\n")
            self.results_text.insert(tk.END, f"├─ Nodes Explored: {len(explored)}\n")
            self.results_text.insert(tk.END, f"├─ Nodes Expanded: {nodes}\n")
            self.results_text.insert(tk.END, f"├─ Path Efficiency: {efficiency:.1f}%\n")
            self.results_text.insert(tk.END, f"└─ Exploration Rate: {exploration_rate:.2f} nodes/step\n")
            self.results_text.insert(tk.END, "─" * 40 + "\n")
        
        self.results_text.see(tk.END)

    def _update_results(self, name, path, explored, nodes):
        result_text = f"\n{name} Results:\n"
        result_text += f"├─ Nodes Expanded: {nodes}\n"
        result_text += f"├─ Path Length: {len(path) if path else 'No path'}\n"
        result_text += f"└─ Total Explored: {len(explored)}\n"
        result_text += "─" * 40 + "\n"
        
        self.results_text.insert(tk.END, result_text)
        self.results_text.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = PathfindingUI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nExiting application...")
        root.quit()
        root.destroy() 