#!/usr/bin/env python3
"""
Python Function Dependency Graph Analyzer

This tool parses a Python project and creates a NetworkX graph showing
function dependencies to help newcomers understand code structure.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, Set, List, Optional, Tuple
import networkx as nx
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_hex
import numpy as np


class FunctionVisitor(ast.NodeVisitor):
    """AST visitor to extract function definitions and their calls."""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.functions: Dict[str, Dict] = {}  # function_name -> {node, calls, class_name}
        self.current_class = None
        self.current_function = None
        self.imports: Dict[str, str] = {}  # alias -> full_name
        self.from_imports: Dict[str, str] = {}  # name -> module
        
    def visit_Import(self, node: ast.Import):
        """Handle 'import module' statements."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports[name] = alias.name
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Handle 'from module import name' statements."""
        if node.module:
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                self.from_imports[name] = node.module
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Handle class definitions."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Handle function definitions."""
        # Create function identifier
        if self.current_class:
            func_id = f"{self.module_name}.{self.current_class}.{node.name}"
        else:
            func_id = f"{self.module_name}.{node.name}"
        
        # Store function info
        self.functions[func_id] = {
            'node': node,
            'calls': set(),
            'class_name': self.current_class
        }
        
        # Visit function body to find calls
        old_function = self.current_function
        self.current_function = func_id
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Handle async function definitions."""
        self.visit_FunctionDef(node)  # Treat same as regular function
    
    def visit_Call(self, node: ast.Call):
        """Handle function calls."""
        if self.current_function:
            call_name = self._extract_call_name(node)
            if call_name:
                self.functions[self.current_function]['calls'].add(call_name)
        self.generic_visit(node)
    
    def _extract_call_name(self, node: ast.Call) -> Optional[str]:
        """Extract the name of a function being called."""
        if isinstance(node.func, ast.Name):
            # Simple function call: func()
            name = node.func.id
            
            # Check if it's an imported function
            if name in self.from_imports:
                return f"{self.from_imports[name]}.{name}"
            elif name in self.imports:
                return f"{self.imports[name]}.{name}"
            else:
                # Local function - assume it's in current module
                return f"{self.module_name}.{name}"
                
        elif isinstance(node.func, ast.Attribute):
            # Method call: obj.method() or module.func()
            return self._extract_attribute_call(node.func)
        
        return None
    
    def _extract_attribute_call(self, node: ast.Attribute) -> Optional[str]:
        """Extract attribute-based calls like obj.method() or module.func()."""
        if isinstance(node.value, ast.Name):
            obj_name = node.value.id
            method_name = node.attr
            
            # Check if it's a module call
            if obj_name in self.imports:
                return f"{self.imports[obj_name]}.{method_name}"
            else:
                # Could be object method call - for now, treat as external
                return f"{obj_name}.{method_name}"
                
        elif isinstance(node.value, ast.Attribute):
            # Nested attribute: obj.attr.method()
            base = self._extract_attribute_call(node.value)
            if base:
                return f"{base}.{node.attr}"
        
        return None


class DependencyAnalyzer:
    """Main class for analyzing Python project dependencies."""
    
    def __init__(self, source_dir: str = "src"):
        self.source_dir = Path(source_dir)
        self.graph = nx.DiGraph()
        self.all_functions: Dict[str, Dict] = {}
        
    def analyze_project(self) -> nx.DiGraph:
        """Analyze the entire project and build dependency graph."""
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory '{self.source_dir}' not found")
        
        # Find all Python files
        python_files = list(self.source_dir.rglob("*.py"))
        
        if not python_files:
            print(f"No Python files found in '{self.source_dir}'")
            return self.graph
        
        # Parse each Python file
        for py_file in python_files:
            if py_file.name == "__init__.py":
                continue  # Skip __init__ files for now
                
            try:
                self._analyze_file(py_file)
            except Exception as e:
                print(f"Error analyzing {py_file}: {e}")
        
        # Build the dependency graph
        self._build_graph()
        
        return self.graph
    
    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file."""
        # Get module name from file path
        relative_path = file_path.relative_to(self.source_dir)
        module_name = str(relative_path.with_suffix('')).replace(os.sep, '.')
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Extract functions and calls
            visitor = FunctionVisitor(module_name)
            visitor.visit(tree)
            
            # Store functions
            self.all_functions.update(visitor.functions)
            
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
    
    def _build_graph(self):
        """Build the NetworkX graph from collected function data."""
        # Add all functions as nodes
        for func_name in self.all_functions:
            self.graph.add_node(func_name, type='local')
        
        # Add edges for dependencies
        for func_name, func_data in self.all_functions.items():
            for called_func in func_data['calls']:
                # Add called function as node if not exists (external function)
                if not self.graph.has_node(called_func):
                    self.graph.add_node(called_func, type='external')
                
                # Add edge: called_func -> func_name (dependency direction)
                self.graph.add_edge(called_func, func_name)
    
    def print_stats(self):
        """Print basic statistics about the analyzed code."""
        local_funcs = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'local']
        external_funcs = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'external']
        
        print(f"\n=== Project Analysis Stats ===")
        print(f"Total functions found: {len(local_funcs)}")
        print(f"External dependencies: {len(external_funcs)}")
        print(f"Total edges (dependencies): {self.graph.number_of_edges()}")
        
        if local_funcs:
            print(f"\nLocal functions:")
            for func in sorted(local_funcs):
                in_degree = self.graph.in_degree(func)
                out_degree = self.graph.out_degree(func)
                print(f"  {func} (called by {out_degree}, calls {in_degree})")
        
        if external_funcs:
            print(f"\nTop external dependencies:")
            external_usage = [(func, self.graph.out_degree(func)) for func in external_funcs]
            external_usage.sort(key=lambda x: x[1], reverse=True)
            for func, usage in external_usage[:10]:
                print(f"  {func} (used {usage} times)")
    
    def save_graph(self, output_file: str = "dependency_graph.gexf"):
        """Save the graph in GEXF format for visualization."""
        nx.write_gexf(self.graph, output_file)
        print(f"Graph saved to {output_file}")
    
    def find_entry_points(self) -> List[str]:
        """Find functions that are likely entry points (high out-degree, low in-degree)."""
        entry_points = []
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'local':
                in_degree = self.graph.in_degree(node)
                out_degree = self.graph.out_degree(node)
                
                # Entry point heuristic: not called by many others but calls others
                if in_degree <= 1 and out_degree > 0:
                    entry_points.append((node, in_degree, out_degree))
        
        return sorted(entry_points, key=lambda x: x[2], reverse=True)
    
    def find_utilities(self) -> List[str]:
        """Find utility functions (called by many, call few)."""
        utilities = []
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'local':
                in_degree = self.graph.in_degree(node)
                out_degree = self.graph.out_degree(node)
                
                # Utility heuristic: called by many but doesn't call many others
                if out_degree >= 2 and in_degree <= out_degree // 2:
                    utilities.append((node, in_degree, out_degree))
        
        return sorted(utilities, key=lambda x: x[2], reverse=True)
    
    def plot_graph(self, output_file: str = None, figsize: Tuple[int, int] = (16, 12), 
                   layout: str = 'spring', show_labels: bool = True):
        """Create a beautiful visualization of the dependency graph."""
        if self.graph.number_of_nodes() == 0:
            print("No graph to plot - no functions found.")
            return
        
        # Set up the plot
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.patch.set_facecolor('white')
        
        # Choose layout algorithm
        pos = self._get_layout(layout)
        
        # Separate nodes by type and characteristics
        local_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'local']
        external_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'external']
        
        # Categorize local nodes by their role
        entry_points = [ep[0] for ep in self.find_entry_points()[:5]]
        utilities = [util[0] for util in self.find_utilities()[:5]]
        
        regular_locals = [n for n in local_nodes if n not in entry_points and n not in utilities]
        
        # Color scheme
        colors = {
            'entry_point': '#FF6B6B',      # Red - entry points
            'utility': '#4ECDC4',          # Teal - utilities
            'local': '#45B7D1',           # Blue - regular local functions
            'external': '#FFA07A',         # Light salmon - external dependencies
        }
        
        # Node sizes based on importance (degree)
        def get_node_size(node):
            degree = self.graph.degree(node)
            base_size = 300
            return base_size + (degree * 50)
        
        # Draw different types of nodes
        if entry_points:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=entry_points,
                                 node_color=colors['entry_point'],
                                 node_size=[get_node_size(n) for n in entry_points],
                                 alpha=0.8, ax=ax)
        
        if utilities:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=utilities,
                                 node_color=colors['utility'],
                                 node_size=[get_node_size(n) for n in utilities],
                                 alpha=0.8, ax=ax)
        
        if regular_locals:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=regular_locals,
                                 node_color=colors['local'],
                                 node_size=[get_node_size(n) for n in regular_locals],
                                 alpha=0.8, ax=ax)
        
        if external_nodes:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=external_nodes,
                                 node_color=colors['external'],
                                 node_size=[get_node_size(n) for n in external_nodes],
                                 alpha=0.6, ax=ax)
        
        # Draw edges with different styles for different types of connections
        local_edges = [(u, v) for u, v in self.graph.edges() 
                      if self.graph.nodes[u].get('type') == 'local' 
                      and self.graph.nodes[v].get('type') == 'local']
        
        external_edges = [(u, v) for u, v in self.graph.edges() 
                         if self.graph.nodes[u].get('type') == 'external']
        
        # Draw local-to-local edges (main dependencies)
        if local_edges:
            nx.draw_networkx_edges(self.graph, pos, edgelist=local_edges,
                                 edge_color='#2C3E50', alpha=0.6, arrows=True,
                                 arrowsize=20, arrowstyle='->', width=1.5, ax=ax)
        
        # Draw external dependencies (lighter)
        if external_edges:
            nx.draw_networkx_edges(self.graph, pos, edgelist=external_edges,
                                 edge_color='#BDC3C7', alpha=0.4, arrows=True,
                                 arrowsize=15, arrowstyle='->', width=1, ax=ax)
        
        # Add labels if requested
        if show_labels:
            # Create abbreviated labels for better readability
            labels = {}
            for node in self.graph.nodes():
                parts = node.split('.')
                if len(parts) > 2:
                    # Show last two parts for long names
                    labels[node] = f"{parts[-2]}.{parts[-1]}"
                else:
                    labels[node] = parts[-1]
            
            # Different label styles for different node types
            local_labels = {n: labels[n] for n in local_nodes if n in labels}
            external_labels = {n: labels[n] for n in external_nodes if n in labels}
            
            if local_labels:
                nx.draw_networkx_labels(self.graph, pos, local_labels, 
                                      font_size=8, font_weight='bold', 
                                      font_color='white', ax=ax)
            
            if external_labels:
                nx.draw_networkx_labels(self.graph, pos, external_labels,
                                      font_size=7, font_weight='normal',
                                      font_color='black', ax=ax)
        
        # Create legend
        legend_elements = [
            mpatches.Patch(color=colors['entry_point'], label='Entry Points'),
            mpatches.Patch(color=colors['utility'], label='Utility Functions'),
            mpatches.Patch(color=colors['local'], label='Local Functions'),
            mpatches.Patch(color=colors['external'], label='External Dependencies')
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        
        # Styling
        ax.set_title(f'Function Dependency Graph\n'
                    f'{len(local_nodes)} Local Functions, {len(external_nodes)} External Dependencies',
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Graph visualization saved to {output_file}")
        else:
            plt.show()
    
    def _get_layout(self, layout: str) -> Dict:
        """Get node positions using specified layout algorithm."""
        if layout == 'spring':
            return nx.spring_layout(self.graph, k=3, iterations=100, seed=42)
        elif layout == 'circular':
            return nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            return nx.kamada_kawai_layout(self.graph)
        elif layout == 'hierarchical':
            return self._hierarchical_layout()
        else:
            return nx.spring_layout(self.graph, k=3, iterations=50, seed=42)
    
    def _hierarchical_layout(self) -> Dict:
        """Create a hierarchical layout based on dependency levels."""
        try:
            # Try to create layers based on topological sort
            if nx.is_directed_acyclic_graph(self.graph):
                layers = list(nx.topological_generations(self.graph))
            else:
                # Fallback for cyclic graphs
                layers = [list(self.graph.nodes())]
            
            pos = {}
            y_step = 1.0 / max(len(layers), 1)
            
            for layer_idx, layer in enumerate(layers):
                y = 1.0 - (layer_idx * y_step)
                x_step = 1.0 / max(len(layer), 1) if layer else 1.0
                
                for node_idx, node in enumerate(layer):
                    x = node_idx * x_step + x_step / 2
                    pos[node] = (x, y)
            
            return pos
        except:
            # Fallback to spring layout
            return nx.spring_layout(self.graph, k=3, iterations=50, seed=42)
    
    def create_interactive_plot(self, output_file: str = "interactive_graph.html"):
        """Create an interactive HTML visualization using a simple template."""
        try:
            import json
            
            # Prepare data for interactive visualization
            nodes = []
            edges = []
            
            local_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'local']
            external_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'external']
            
            # Add nodes
            for node in self.graph.nodes():
                node_type = self.graph.nodes[node].get('type', 'unknown')
                degree = self.graph.degree(node)
                
                nodes.append({
                    'id': node,
                    'label': node.split('.')[-1],  # Short name
                    'title': node,  # Full name for tooltip
                    'group': node_type,
                    'size': min(10 + degree * 2, 30)
                })
            
            # Add edges
            for source, target in self.graph.edges():
                edges.append({
                    'from': source,
                    'to': target
                })
            
            # Create HTML content
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Function Dependency Graph</title>
    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet" type="text/css" />
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        #graph {{ width: 100%; height: 600px; border: 1px solid lightgray; }}
        .info {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Function Dependency Graph</h1>
    <div class="info">
        <p><strong>Local Functions:</strong> {len(local_nodes)} | <strong>External Dependencies:</strong> {len(external_nodes)}</p>
        <p>Click and drag nodes to explore the graph. Hover for full function names.</p>
    </div>
    <div id="graph"></div>
    
    <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(nodes)});
        var edges = new vis.DataSet({json.dumps(edges)});
        
        var container = document.getElementById('graph');
        var data = {{ nodes: nodes, edges: edges }};
        
        var options = {{
            nodes: {{
                shape: 'dot',
                scaling: {{ min: 10, max: 30 }},
                font: {{ size: 12, face: 'arial' }}
            }},
            edges: {{
                arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }},
                color: {{ inherit: false }},
                smooth: {{ enabled: true, type: 'continuous' }}
            }},
            groups: {{
                local: {{ color: {{ background: '#45B7D1', border: '#2C3E50' }} }},
                external: {{ color: {{ background: '#FFA07A', border: '#FF6347' }} }}
            }},
            physics: {{
                stabilization: {{ iterations: 150 }},
                barnesHut: {{ gravitationalConstant: -8000, springConstant: 0.001, springLength: 200 }}
            }},
            interaction: {{ hover: true }}
        }};
        
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>"""
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"Interactive graph saved to {output_file}")
            print(f"Open {output_file} in your web browser to explore the graph interactively.")
            
        except Exception as e:
            print(f"Could not create interactive plot: {e}")
            print("Falling back to static matplotlib visualization.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze Python project function dependencies")
    parser.add_argument("--src", default="src", help="Source directory (default: src)")
    parser.add_argument("--output", default="dependency_graph.gexf", 
                       help="Output graph file (default: dependency_graph.gexf)")
    parser.add_argument("--stats", action="store_true", help="Show detailed statistics")
    parser.add_argument("--plot", action="store_true", help="Generate graph visualization")
    parser.add_argument("--plot-file", help="Save plot to file (PNG/PDF/SVG)")
    parser.add_argument("--layout", choices=['spring', 'circular', 'kamada_kawai', 'hierarchical'],
                       default='spring', help="Layout algorithm for graph visualization")
    parser.add_argument("--figsize", nargs=2, type=int, default=[16, 12],
                       help="Figure size for plot (width height)")
    parser.add_argument("--interactive", action="store_true", 
                       help="Create interactive HTML visualization")
    parser.add_argument("--no-labels", action="store_true", help="Hide node labels in plot")
    
    args = parser.parse_args()
    
    try:
        # Create analyzer and process project
        analyzer = DependencyAnalyzer(args.src)
        graph = analyzer.analyze_project()
        
        if graph.number_of_nodes() == 0:
            print("No functions found to analyze.")
            return
        
        # Print basic info
        analyzer.print_stats()
        
        if args.stats:
            # Find and display entry points
            entry_points = analyzer.find_entry_points()
            if entry_points:
                print(f"\n=== Potential Entry Points ===")
                for func, in_deg, out_deg in entry_points[:5]:
                    print(f"  {func} (calls {in_deg} functions, called by {out_deg})")
            
            # Find and display utilities
            utilities = analyzer.find_utilities()
            if utilities:
                print(f"\n=== Utility Functions ===")
                for func, in_deg, out_deg in utilities[:5]:
                    print(f"  {func} (calls {in_deg} functions, called by {out_deg})")
        
        # Save graph data
        analyzer.save_graph(args.output)
        
        # Create visualizations
        if args.plot or args.plot_file:
            try:
                print("\nCreating graph visualization...")
                analyzer.plot_graph(
                    output_file=args.plot_file,
                    figsize=tuple(args.figsize),
                    layout=args.layout,
                    show_labels=not args.no_labels
                )
            except ImportError:
                print("Error: matplotlib is required for plotting. Install with: pip install matplotlib")
            except Exception as e:
                print(f"Error creating plot: {e}")
        
        if args.interactive:
            try:
                print("\nCreating interactive visualization...")
                analyzer.create_interactive_plot()
            except Exception as e:
                print(f"Error creating interactive plot: {e}")
        
        if not (args.plot or args.plot_file or args.interactive):
            print(f"\nGraph data saved to {args.output}")
            print("Use --plot to create visualization or --interactive for HTML graph")
            print("Install matplotlib for plotting: pip install matplotlib")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
