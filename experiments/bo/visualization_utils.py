import matplotlib.pyplot as plt
import torch

# import tikzplotlib
import torch
import pub_ready_plots as prp
import math
import tikzplotlib

COLORS = [
    "blue",
    "purple",
    "orange",
    "cyan",
    "magenta",
    "yellow",
    "red",
    "grey",
    "black",
]

LINESTYLES = [
    "-",  # Solid line
    "--",  # Dashed line
    ":",  # Dotted line
    "-.",  # Dash-dot line
    (0, (1, 1)),  # Densely dotted
    (0, (5, 1)),  # Densely dashed
    (0, (5, 10)),  # Loosely dashed
    (0, (3, 5, 1, 5)),  # Dash-dot-dotted
    (0, (1, 10)),  # Loosely dotted
]


def plot_optimum_evolution(
    res_dict,
    x_range,
    title=None,
    save_folder="results/",
    output_file="optimum_evolution_nd",
    n_init=0,
    minimum=0,
    colors=None,
    linestyles=None,
    plot_legend=False,
    plot_axis=False,
    save_type="png",
):
    """
    Plots the evolution of the optimum value for various optimization methods.

    Args:
        res_dict (dict): Optimization results as {method_name: [replications]}.
        x_range (tuple): Range of input values to consider.
        title (str, optional): Title of the plot.
        save_folder (str): Directory to save the results.
        output_file (str): Output file name (without extension).
        n_init (int): Number of initial iterations to skip.
        minimum (float): Value to subtract from objectives for normalization.
        colors (list, optional): Custom colors for the lines.
        linestyles (list, optional): Custom line styles for the lines.
        plot_legend (bool): Whether to include a legend.
        plot_axis (bool): Whether to show axis labels and title.
        save_type (str): File format for saving the plot ('png' or 'pdf').

    Returns:
        None
    """
    n_init = n_init -1
    def compute_cumulative_minimum(values):
        """Compute the cumulative minimum along the first dimension."""
        return torch.cummin(values, dim=0).values

    # Use default colors and linestyles if not provided
    colors = colors or COLORS
    linestyles = linestyles or LINESTYLES

    # Plot setup using a custom context (assume prp.get_context is defined elsewhere)
    with prp.get_context(layout=prp.Layout.NEURIPS, width_frac=1, height_frac=0.4) as (fig, axs):
        for i, method_name in enumerate(res_dict.keys()):
            cumulative_minima = []

            for replication in res_dict[method_name]:
                if "GP" not in method_name and "BOPrO" not in method_name:
                    x = replication["batch"].xc[:, :, 1:]
                    y = replication["batch"].yc - minimum
                else:
                    x = replication[0][None, :, :]
                    y = replication[1][None, :, :] - minimum

                # Filter values within bounds
                within_bounds = (x >= x_range[0]) & (x <= x_range[1])
                within_bounds = within_bounds.all(dim=-1, keepdim=True)
                y[~within_bounds] = torch.inf

                cumulative_minima.append(compute_cumulative_minimum(y[0, :, 0])[None, :])

            # Convert cumulative minima to tensor and compute statistics
            cumulative_minima_tensor = torch.cat(cumulative_minima)
            mean = cumulative_minima_tensor.mean(dim=0)
            std_error = cumulative_minima_tensor.std(dim=0) / torch.sqrt(torch.tensor(len(cumulative_minima)))

            # Prepare data for plotting
            iterations = torch.arange(1, len(mean) + 1).detach().numpy()
            mean_np = mean.detach().numpy()
            std_error_np = std_error.detach().numpy()

            # Plot mean and uncertainty
            axs.plot(
                iterations[n_init:],
                mean_np[n_init:],
                color=colors[i],
                label=method_name,
                linestyle=linestyles[i],
            )
            axs.fill_between(
                iterations[n_init:],
                mean_np[n_init:] - std_error_np[n_init:],
                mean_np[n_init:] + std_error_np[n_init:],
                color=colors[i],
                alpha=0.3,
            )

        # Configure plot appearance
        if plot_axis:
            axs.set_title(title)
            if plot_legend:
                axs.legend()
        else:
            axs.set_xticklabels([])
            axs.set_yticklabels([])

        # Save the plot
        file_path = f"{save_folder}{output_file}.{save_type}"
        fig.savefig(file_path, format=save_type, dpi=300 if save_type == "png" else None)
        tikzplotlib.save(
            f"{save_folder}{output_file}.tex",
            axis_width="\\figurewidth",
            axis_height="\\figureheight",
        )

        print(f"Plot saved at {file_path}")


def plot_optimum_evolution_all(
    res_dicts,
    x_ranges,
    titles=None,
    save_folder="results/",
    output_file="optimum_evolution_nd",
    n_inits=None,
    minimums=None,
    colors=None,
    linestyles=None,
    plot_legend=False,
    plot_axis=False,
    save_type="png",
):
    """
    Plots the evolution of optimum values across multiple datasets in subplots.

    Args:
        res_dicts (list): List of dictionaries containing optimization results.
        x_ranges (list): List of tuples specifying x-ranges for each dataset.
        titles (list, optional): Titles for each subplot.
        save_folder (str): Directory to save the results.
        output_file (str): Output file name (without extension).
        n_inits (list): List of initial iterations to skip for each dataset.
        minimums (list): List of minimum values to subtract for normalization.
        colors (list, optional): Custom colors for the lines.
        linestyles (list, optional): Custom line styles for the lines.
        plot_legend (bool): Whether to include a legend.
        plot_axis (bool): Whether to show axis labels and titles.
        save_type (str): File format for saving the plot ('png' or 'pdf').

    Returns:
        None
    """

    def compute_cumulative_minimum(values, take_log=False):
        """Compute the cumulative minimum along the first dimension."""
        val = torch.cummin(values, dim=0).values
        return torch.log(val) if take_log else val

    nrows, ncols = calculate_rows_columns(len(res_dicts))

    # with prp.get_context(
    #     layout=None,
    #     width_frac=1,
    #     height_frac=0.15,
    #     nrows=nrows,
    #     ncols=ncols,
    #     sharey=False,  # Allow different y-scales for each plot
    #     sharex=True,   # Share x-axis
    # ) as (fig, axs):
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3))
    axs = axs.flatten()
    for n, res_dict in enumerate(res_dicts):
        for i, method_name in enumerate(res_dict.keys()):
            cumulative_minima = []
            minimum = minimums[n]
            x_range = x_ranges[n]
            n_init = n_inits[n] -1

            for replication in res_dict[method_name]["res"]:
                if "GP" not in method_name and "BOPrO" not in method_name:
                    x = replication["batch"].xc[:, :, 1:]
                    y = replication["batch"].yc - minimum
                else:
                    x = replication[0][None, :, :]
                    y = replication[1][None, :, :] - minimum

                # Filter values within bounds
                within_bounds = (x >= x_range[0]) & (x <= x_range[1])
                within_bounds = within_bounds.all(dim=-1, keepdim=True)
                y[~within_bounds] = torch.inf

                cumulative_minima.append(compute_cumulative_minimum(y[0, :, 0])[None, :])

            # Aggregate results and compute statistics
            cumulative_minima_tensor = torch.cat(cumulative_minima)
            mean = cumulative_minima_tensor.mean(dim=0)
            std_error = cumulative_minima_tensor.std(dim=0) / torch.sqrt(torch.tensor(len(cumulative_minima)))

            # Plot data
            iterations = torch.arange(1, len(mean) + 1).detach().numpy()
            mean_np = mean.detach().numpy()
            std_error_np = std_error.detach().numpy()

            axs[n].plot(
                iterations[n_init:],
                mean_np[n_init:],
                color=colors[n][i],
                label=method_name,
                linestyle=linestyles[n][i],
            )
            axs[n].fill_between(
                iterations[n_init:],
                mean_np[n_init:] - std_error_np[n_init:],
                mean_np[n_init:] + std_error_np[n_init:],
                color=colors[n][i],
                alpha=0.3,
            )

        # Add subplot-specific details
        if titles and n < len(titles):
            axs[n].set_title(titles[n], loc="right")

        if plot_legend and n == 0:
            legend = axs[n].legend()

    # Set shared x-label for the entire figure
    fig.text(0.5, 0.01, "Iterations", ha="center")

    # Set shared y-label but allow different y-scales
    fig.text(0.01, 0.5, "Optimum", va="center", rotation="vertical")

    # Create and save a shared legend
    handles, labels = axs[0].get_legend_handles_labels()
    sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i])
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_handles = [handles[i] for i in sorted_indices]
    fig_legend = plt.figure(figsize=(len(labels), 0.15))
    fig_legend.legend(sorted_handles, sorted_labels, loc="center", ncol=len(labels))

    # Save plots and legend
    
    fig_legend.savefig(f"{save_folder}legend_only.{save_type}", bbox_inches="tight", dpi=300)
    #legend.remove()
    fig.savefig(f"{save_folder}{output_file}.{save_type}", format=save_type, dpi=300)

    print(f"Plots saved at {save_folder}{output_file}.{save_type} and legend saved as {save_folder}legend_only.{save_type}")


       
def calculate_rows_columns(N):
    if N < 6:
        rows = 1
        cols = N
        return rows, cols
    for n in range(int(math.sqrt(N)), 0, -1):
        if N % n == 0:
            rows = n
            cols = N // n
            return rows, cols