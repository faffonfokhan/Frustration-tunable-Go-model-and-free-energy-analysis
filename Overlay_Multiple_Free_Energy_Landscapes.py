import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def free_energy(Qs, delta_Q=0.01, temperature=300):
    Qs = np.array(Qs)
    Qs = Qs[~np.isnan(Qs)]
    Qmin = np.min(Qs)
    Qmax = np.max(Qs)
    bins = np. arange(Qmin, Qmax + delta_Q, delta_Q)
    hist, bin_edges = np.histogram(Qs, bins=bins)
    Q = (bin_edges[:-1] + bin_edges[1:]) / 2
    prob = np.maximum(hist / np.sum(hist), 1e-10)
    R = 8.314e-3
    F = -R * temperature * np.log(prob)
    F = F - np.min(F)
    
    return Q, F

def plot_overlay_fel(csv_files, labels=None, delta_Q=0.01, temperature=300, 
                     figsize=(10, 6), font_size=14, axis_width=1.5,
                     output_file='FEL_overlay.png', colors=None):
    """
    Overlay multiple FEL curves on a single plot.
    
    Parameters:
    ----------
    csv_files : list of str
        List of paths to CSV files containing reaction coordinate data.
    labels : list of str, optional
    
        Labels for each curve.  If None, uses filenames.
    delta_Q :  float, optional
        Bin width for histogram calculation, default is 0.01.
    temperature : float, optional
        Temperature in Kelvin, default is 300K. 
    figsize : tuple, optional
        Figure size for the plot, default is (10, 6).
    font_size : int, optional
        Font size for axes labels and ticks, default is 14.
    axis_width : float, optional
        Width of axis lines, default is 1.5.
    output_file : str, optional
        Output filename for the plot. 
    colors : list, optional
        List of colors for each curve. If None, uses default color cycle.
    
    Returns:
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    """
    if colors is None:
        colors = ['#000000', '#1c0bb8', '#fca000', '#fa0a0a', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#91FF14', '#FB607F', '#FF0800', '#9966CC', '#32174D', '#76ABDF', '#4B0082', '#50C878','#FF1493', '#FF00FF']
    
    if labels is None:
        labels = [os.path.basename(f).replace('.csv', '') for f in csv_files]
    plt.rcParams.update({'font.size': font_size})
    fig, ax = plt.subplots(figsize=figsize)
    for i, csv_file in enumerate(csv_files):
        try:
            df = pd. read_csv(csv_file)
            Qs = df.iloc[:, 0].values
            Q, F = free_energy(Qs, delta_Q, temperature)
            color = colors[i % len(colors)]
            ax.plot(Q, F, '-', lw=2.5, color=color, label=labels[i])
            
            print(f"Plotted:  {labels[i]}")
            
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    ax.set_xlabel('Q', fontsize=font_size+2, fontweight='bold')
    ax.set_ylabel('F(Q) (kJ/mol)', fontsize=font_size+2, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.tick_params(axis='both', which='major', labelsize=font_size, width=axis_width)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=font_size-2, framealpha=0.9, loc='best')
    
    for spine in ax.spines.values():
        spine.set_linewidth(axis_width)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nOverlay plot saved to '{output_file}'")
    
    return fig


if __name__ == "__main__": 
    if len(sys. argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        print_usage()
        sys.exit(0)
    csv_files = []
    labels = None
    delta_Q = 0.01
    temperature = 300
    output_file = 'FEL_overlay.png'
    colors = None
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == '--labels': 
            labels = []
            i += 1
            while i < len(sys.argv) and not sys.argv[i].startswith('--'):
                labels.append(sys.argv[i])
                i += 1
        elif arg == '--delta':
            if i + 1 < len(sys.argv):
                try:
                    delta_Q = float(sys.argv[i+1])
                    i += 2
                except ValueError:
                    print(f"Error: Invalid value for delta: {sys.argv[i+1]}")
                    sys.exit(1)
            else:
                print("Error: --delta requires a value")
                sys.exit(1)
        elif arg == '--temp':
            if i + 1 < len(sys.argv):
                try:
                    temperature = float(sys.argv[i+1])
                    i += 2
                except ValueError:
                    print(f"Error: Invalid value for temperature: {sys.argv[i+1]}")
                    sys. exit(1)
            else:
                print("Error: --temp requires a value")
                sys. exit(1)
        elif arg == '--output':
            if i + 1 < len(sys.argv):
                output_file = sys.argv[i+1]
                i += 2
            else:
                print("Error: --output requires a filename")
                sys.exit(1)
        elif arg == '--colors': 
            colors = []
            i += 1
            while i < len(sys.argv) and not sys.argv[i].startswith('--'):
                colors.append(sys. argv[i])
                i += 1
        elif arg.endswith('.csv'):
            csv_files.append(arg)
            i += 1
        else: 
            print(f"Error:  Unknown option or invalid file: {arg}")
            print_usage()
            sys.exit(1)
    
    if len(csv_files) < 2:
        print("Error: At least 2 CSV files are required for overlay plot")
        print_usage()
        sys.exit(1)
    
    if labels is not None and len(labels) != len(csv_files):
        print(f"Warning: Number of labels ({len(labels)}) doesn't match number of files ({len(csv_files)})")
        print("Using filenames as labels instead")
        labels = None
    
    print(f"\nCreating overlay plot with {len(csv_files)} curves...")
    print(f"Temperature: {temperature} K")
    print(f"Delta Q: {delta_Q}")
    print()
    
    fig = plot_overlay_fel(
        csv_files,
        labels=labels,
        delta_Q=delta_Q,
        temperature=temperature,
        output_file=output_file,
        colors=colors
    )
    
    plt.show()
