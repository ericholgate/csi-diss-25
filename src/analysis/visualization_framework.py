"""
Visualization Framework for CSI Analysis
=========================================

Unified visualization framework for creating publication-ready plots.
Provides consistent styling, color schemes, and plot templates for all analyses.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from dataclasses import dataclass


@dataclass
class PlotConfig:
    """Configuration for plot styling."""
    figure_size: Tuple[float, float] = (10, 6)
    dpi: int = 100
    font_size: int = 11
    title_size: int = 14
    label_size: int = 12
    legend_size: int = 10
    line_width: float = 2.0
    marker_size: float = 8.0
    grid_alpha: float = 0.3
    color_palette: str = 'husl'
    style: str = 'seaborn-v0_8-darkgrid'


class VisualizationFramework:
    """Unified visualization framework for CSI analysis."""
    
    # Color schemes for different analysis types
    COLOR_SCHEMES = {
        'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        'character_types': {
            'main': '#FFD700',      # Gold
            'recurring': '#87CEEB',  # Sky blue
            'guest': '#90EE90',      # Light green
            'minor': '#FFB6C1'       # Light pink
        },
        'performance': {
            'train': '#2E7D32',      # Green
            'validation': '#1976D2',  # Blue
            'test': '#C62828'        # Red
        },
        'embeddings': {
            'killer': '#DC143C',     # Crimson
            'suspect': '#FF8C00',    # Dark orange
            'innocent': '#4169E1',   # Royal blue
            'unknown': '#808080'     # Gray
        },
        'heatmap': 'RdYlBu_r',
        'diverging': 'coolwarm',
        'sequential': 'viridis'
    }
    
    def __init__(self, config: Optional[PlotConfig] = None):
        """Initialize visualization framework."""
        self.config = config or PlotConfig()
        self.setup_style()
        
    def setup_style(self):
        """Set up matplotlib style settings."""
        # Use style if available
        try:
            plt.style.use(self.config.style)
        except:
            plt.style.use('seaborn-darkgrid')
        
        # Set default parameters
        mpl.rcParams.update({
            'figure.figsize': self.config.figure_size,
            'figure.dpi': self.config.dpi,
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.title_size,
            'axes.labelsize': self.config.label_size,
            'xtick.labelsize': self.config.label_size - 1,
            'ytick.labelsize': self.config.label_size - 1,
            'legend.fontsize': self.config.legend_size,
            'lines.linewidth': self.config.line_width,
            'lines.markersize': self.config.marker_size,
            'axes.grid': True,
            'grid.alpha': self.config.grid_alpha,
            'figure.autolayout': True
        })
        
        # Set seaborn palette
        sns.set_palette(self.config.color_palette)
    
    def get_colors(self, color_type: str = 'default', n_colors: Optional[int] = None) -> Union[List[str], Dict[str, str], str]:
        """
        Get colors for plotting.
        
        Args:
            color_type: Type of color scheme
            n_colors: Number of colors needed (for list types)
        """
        if color_type in self.COLOR_SCHEMES:
            colors = self.COLOR_SCHEMES[color_type]
            if isinstance(colors, list):
                if n_colors and n_colors > len(colors):
                    # Generate more colors if needed
                    return sns.color_palette(self.config.color_palette, n_colors).as_hex()
                return colors[:n_colors] if n_colors else colors
            return colors
        return self.COLOR_SCHEMES['default']
    
    def create_figure(self, nrows: int = 1, ncols: int = 1, 
                     figsize: Optional[Tuple[float, float]] = None,
                     **kwargs) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """Create a figure with consistent styling."""
        figsize = figsize or (self.config.figure_size[0] * ncols, 
                             self.config.figure_size[1] * nrows)
        return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    
    def plot_training_curves(self, history: Dict[str, List[float]], 
                            title: str = "Training History",
                            save_path: Optional[Path] = None) -> plt.Figure:
        """Plot training and validation curves."""
        fig, axes = self.create_figure(1, 2, figsize=(14, 6))
        colors = self.get_colors('performance')
        
        # Loss curves
        ax = axes[0]
        if 'train_loss' in history:
            ax.plot(history['train_loss'], label='Train', 
                   color=colors['train'], linewidth=2)
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Validation', 
                   color=colors['validation'], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=self.config.grid_alpha)
        
        # Accuracy curves
        ax = axes[1]
        if 'train_acc' in history:
            ax.plot(history['train_acc'], label='Train', 
                   color=colors['train'], linewidth=2)
        if 'val_acc' in history:
            ax.plot(history['val_acc'], label='Validation', 
                   color=colors['validation'], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Curves')
        ax.legend()
        ax.grid(True, alpha=self.config.grid_alpha)
        
        plt.suptitle(title, fontsize=self.config.title_size + 2, fontweight='bold')
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig
    
    def plot_confusion_matrix(self, cm: np.ndarray, labels: List[str],
                             title: str = "Confusion Matrix",
                             normalize: bool = True,
                             save_path: Optional[Path] = None) -> plt.Figure:
        """Plot confusion matrix with annotations."""
        fig, ax = self.create_figure(figsize=(8, 6))
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=self.COLOR_SCHEMES['heatmap'],
                   xticklabels=labels, yticklabels=labels,
                   square=True, cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                   ax=ax)
        
        ax.set_xlabel('Predicted', fontsize=self.config.label_size)
        ax.set_ylabel('Actual', fontsize=self.config.label_size)
        ax.set_title(title, fontsize=self.config.title_size, fontweight='bold')
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig
    
    def plot_character_distribution(self, character_data: Dict[str, int],
                                   top_n: int = 20,
                                   title: str = "Character Distribution",
                                   save_path: Optional[Path] = None) -> plt.Figure:
        """Plot character frequency distribution."""
        fig, ax = self.create_figure(figsize=(12, 8))
        
        # Sort and get top N
        sorted_chars = sorted(character_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
        names, values = zip(*sorted_chars)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(names))
        colors = self.get_colors('default', len(names))
        
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Frequency', fontsize=self.config.label_size)
        ax.set_title(title, fontsize=self.config.title_size, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:,}', ha='left', va='center', fontsize=9)
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig
    
    def plot_embedding_scatter(self, embeddings: np.ndarray, labels: Optional[np.ndarray] = None,
                              names: Optional[List[str]] = None,
                              title: str = "Character Embeddings",
                              method: str = "tsne",
                              save_path: Optional[Path] = None) -> plt.Figure:
        """Plot 2D embedding visualization."""
        fig, ax = self.create_figure(figsize=(10, 8))
        
        # Determine colors
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = self.get_colors('embeddings')
            label_colors = [colors.get(str(l), colors['unknown']) for l in labels]
        else:
            label_colors = self.get_colors('default', 1)[0]
        
        # Create scatter plot
        scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1],
                           c=label_colors, alpha=0.6, s=50)
        
        # Add labels if provided
        if names:
            for i, name in enumerate(names[:20]):  # Limit to avoid clutter
                ax.annotate(name, (embeddings[i, 0], embeddings[i, 1]),
                          fontsize=8, alpha=0.7)
        
        ax.set_xlabel(f'{method.upper()} 1', fontsize=self.config.label_size)
        ax.set_ylabel(f'{method.upper()} 2', fontsize=self.config.label_size)
        ax.set_title(title, fontsize=self.config.title_size, fontweight='bold')
        
        # Add legend if labels provided
        if labels is not None:
            handles = []
            for label in unique_labels:
                handles.append(plt.scatter([], [], c=colors.get(str(label), colors['unknown']),
                                          label=str(label), s=50))
            ax.legend(handles=handles)
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig
    
    def plot_comparison_bars(self, data: Dict[str, float],
                           title: str = "Model Comparison",
                           ylabel: str = "Score",
                           highlight_best: bool = True,
                           save_path: Optional[Path] = None) -> plt.Figure:
        """Plot model comparison bar chart."""
        fig, ax = self.create_figure(figsize=(10, 6))
        
        names = list(data.keys())
        values = list(data.values())
        
        # Determine colors
        if highlight_best:
            best_idx = np.argmax(values)
            colors = ['gold' if i == best_idx else 'steelblue' for i in range(len(values))]
        else:
            colors = self.get_colors('default', len(names))
        
        bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel(ylabel, fontsize=self.config.label_size)
        ax.set_title(title, fontsize=self.config.title_size, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.1)
        
        # Rotate x-labels if needed
        if len(names) > 5:
            plt.xticks(rotation=45, ha='right')
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig
    
    def plot_timeline(self, timeline_data: Dict[str, List[float]],
                     title: str = "Timeline Analysis",
                     xlabel: str = "Time",
                     ylabel: str = "Value",
                     save_path: Optional[Path] = None) -> plt.Figure:
        """Plot timeline data for multiple series."""
        fig, ax = self.create_figure(figsize=(14, 6))
        
        colors = self.get_colors('default', len(timeline_data))
        
        for (name, values), color in zip(timeline_data.items(), colors):
            ax.plot(values, label=name, color=color, linewidth=2, marker='o',
                   markersize=4, alpha=0.8)
        
        ax.set_xlabel(xlabel, fontsize=self.config.label_size)
        ax.set_ylabel(ylabel, fontsize=self.config.label_size)
        ax.set_title(title, fontsize=self.config.title_size, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=self.config.grid_alpha)
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig
    
    def create_subplot_grid(self, plots: List[Dict[str, Any]],
                          title: str = "Analysis Grid",
                          save_path: Optional[Path] = None) -> plt.Figure:
        """Create a grid of subplots with different plot types."""
        n_plots = len(plots)
        ncols = min(3, n_plots)
        nrows = (n_plots + ncols - 1) // ncols
        
        fig, axes = self.create_figure(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        if nrows == 1 and ncols == 1:
            axes = [axes]
        elif nrows == 1 or ncols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for idx, plot_config in enumerate(plots):
            ax = axes[idx]
            plot_type = plot_config.get('type', 'line')
            
            if plot_type == 'line':
                ax.plot(plot_config['data'], **plot_config.get('kwargs', {}))
            elif plot_type == 'bar':
                ax.bar(range(len(plot_config['data'])), plot_config['data'], 
                      **plot_config.get('kwargs', {}))
            elif plot_type == 'scatter':
                ax.scatter(plot_config['x'], plot_config['y'], 
                         **plot_config.get('kwargs', {}))
            elif plot_type == 'hist':
                ax.hist(plot_config['data'], **plot_config.get('kwargs', {}))
            
            ax.set_title(plot_config.get('title', f'Plot {idx+1}'))
            ax.set_xlabel(plot_config.get('xlabel', ''))
            ax.set_ylabel(plot_config.get('ylabel', ''))
            ax.grid(True, alpha=self.config.grid_alpha)
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(title, fontsize=self.config.title_size + 2, fontweight='bold')
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig
    
    def save_figure(self, fig: plt.Figure, path: Path, dpi: Optional[int] = None,
                   formats: List[str] = ['png', 'pdf']):
        """Save figure in multiple formats."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        dpi = dpi or self.config.dpi * 3  # Higher DPI for saved figures
        
        for fmt in formats:
            save_path = path.with_suffix(f'.{fmt}')
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
    
    def create_latex_table(self, data: pd.DataFrame, caption: str = "Results",
                          label: str = "tab:results") -> str:
        """Generate LaTeX table from DataFrame."""
        latex = data.to_latex(index=False, escape=False, column_format='l' + 'c' * (len(data.columns) - 1))
        
        # Wrap in table environment
        latex_full = f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{latex}
\\end{{table}}"""
        
        return latex_full
    
    @staticmethod
    def set_publication_style():
        """Set style for publication-quality figures."""
        plt.style.use('seaborn-v0_8-whitegrid')
        mpl.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'lines.linewidth': 1.5,
            'lines.markersize': 6,
            'axes.linewidth': 0.8,
            'grid.alpha': 0.2,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],
            'text.usetex': False  # Set to True if LaTeX is available
        })


# Convenience functions
def quick_plot(data: Union[List, np.ndarray, Dict], plot_type: str = 'line',
              title: str = "", save: bool = False) -> plt.Figure:
    """Quick plotting function for rapid visualization."""
    viz = VisualizationFramework()
    fig, ax = viz.create_figure()
    
    if plot_type == 'line':
        if isinstance(data, dict):
            for name, values in data.items():
                ax.plot(values, label=name)
            ax.legend()
        else:
            ax.plot(data)
    elif plot_type == 'bar':
        if isinstance(data, dict):
            ax.bar(data.keys(), data.values())
        else:
            ax.bar(range(len(data)), data)
    elif plot_type == 'hist':
        ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
    
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if save:
        viz.save_figure(fig, Path(f"quick_plot_{plot_type}.png"))
    
    return fig


def create_publication_figure(plot_func, *args, **kwargs) -> plt.Figure:
    """Wrapper to create publication-quality figures."""
    VisualizationFramework.set_publication_style()
    fig = plot_func(*args, **kwargs)
    return fig


def main():
    """Demo of visualization framework."""
    import numpy as np
    
    # Initialize framework
    viz = VisualizationFramework()
    
    # Demo data
    epochs = 50
    train_loss = np.exp(-np.linspace(0, 3, epochs)) + np.random.normal(0, 0.01, epochs)
    val_loss = np.exp(-np.linspace(0, 2.5, epochs)) + np.random.normal(0, 0.02, epochs)
    
    history = {
        'train_loss': train_loss.tolist(),
        'val_loss': val_loss.tolist(),
        'train_acc': (1 - train_loss).tolist(),
        'val_acc': (1 - val_loss).tolist()
    }
    
    # Create training curves
    fig1 = viz.plot_training_curves(history, title="Model Training Progress")
    
    # Create comparison plot
    model_scores = {
        'Baseline': 0.65,
        'TF-IDF + SVM': 0.72,
        'BERT': 0.85,
        'Our Method': 0.89
    }
    fig2 = viz.plot_comparison_bars(model_scores, title="Model Performance Comparison",
                                   ylabel="Accuracy")
    
    # Create character distribution
    char_data = {f'Character_{i}': np.random.randint(10, 1000) for i in range(20)}
    fig3 = viz.plot_character_distribution(char_data, title="Character Speaking Frequency")
    
    plt.show()
    
    print("Visualization framework demo complete!")


if __name__ == "__main__":
    main()