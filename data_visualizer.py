from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column, row
from bokeh.models import HoverTool
import numpy as np
import matplotlib.pyplot as plt
import os


class PlotEngine:
    """
    Handles all visualization tasks for the assignment pipeline.

    Generates interactive Bokeh HTML plots and static matplotlib PNG exports
    for training data, ideal function matches, test point mappings, and deviations.

    """

    def __init__(self):
        # Initialize an empty list to hold any plots generated during the session
        self.chart_collection = []

    def build_training_charts(self, train_df, ideal_df, matched_funcs):
        """
        Builds one scatter+line plot per training function, overlaying
        its best-matched ideal function for visual comparison.

        """
        # Distinct colors for each of the 4 training functions
        palette = ['#e63946', '#2a9d8f', '#e9c46a', '#6a0572']
        chart_list = []

        for t_idx in range(1, 5):
            # Create a new Bokeh figure for this training function
            chart = figure(
                title=f"Training y{t_idx}  vs  Ideal y{matched_funcs[t_idx]}",
                x_axis_label='X Axis',
                y_axis_label='Y Axis',
                width=500,
                height=400,
                tools="pan,wheel_zoom,box_zoom,reset,save"
            )

            # Plot training data points as scattered dots
            chart.scatter(
                train_df['x'],
                train_df[f'y{t_idx}'],
                size=6,
                color=palette[t_idx - 1],
                alpha=0.7,
                legend_label=f'Training y{t_idx}'
            )

            # Overlay the matched ideal function as a continuous line
            i_idx = matched_funcs[t_idx]
            chart.line(
                ideal_df['x'],
                ideal_df[f'y{i_idx}'],
                line_width=2,
                color='#264653',       # Dark teal line for all ideal overlays
                alpha=0.85,
                legend_label=f'Ideal y{i_idx}'
            )

            # Place legend in top-left and allow clicking to toggle visibility
            chart.legend.location = "top_left"
            chart.legend.click_policy = "hide"

            chart_list.append(chart)

        return chart_list

    def build_mapping_chart(self, test_df, mappings, ideal_df, matched_funcs):
        """
        Builds a combined plot showing all matched ideal function curves
        alongside the test points mapped to each of them.


        """
        chart = figure(
            title="Test Point Assignments to Matched Ideal Functions",
            x_axis_label='X Axis',
            y_axis_label='Y Axis',
            width=1000,
            height=600,
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )

        # Assign a unique color to each matched ideal function
        color_lookup = {
            matched_funcs[1]: '#e63946',   # Red
            matched_funcs[2]: '#2a9d8f',   # Teal
            matched_funcs[3]: '#f4a261',   # Orange
            matched_funcs[4]: '#6a0572'    # Purple
        }

        # Draw each matched ideal function as a background reference line
        for t_idx, i_idx in matched_funcs.items():
            chart.line(
                ideal_df['x'],
                ideal_df[f'y{i_idx}'],
                line_width=2,
                color=color_lookup[i_idx],
                alpha=0.45,
                legend_label=f'Ideal y{i_idx}'
            )

        # Overlay test points grouped by which ideal function they were mapped to
        if not mappings.empty:
            for i_idx in mappings['ideal_function_index'].unique():
                group = mappings[mappings['ideal_function_index'] == i_idx]

                chart.scatter(
                    group['x'],
                    group['y'],
                    size=8,
                    color=color_lookup[i_idx],
                    alpha=0.9,
                    legend_label=f'Test → Ideal y{i_idx}'
                )

        # Add hover tooltip to inspect individual test point coordinates
        hover = HoverTool(tooltips=[
            ("X", "@x{0.00}"),
            ("Y", "@y{0.00}")
        ])
        chart.add_tools(hover)

        chart.legend.location = "top_left"
        chart.legend.click_policy = "hide"

        return chart

    def build_deviation_chart(self, mappings, max_deviations, matched_funcs):
        """
        Builds a scatter plot of test point deviations from their matched ideal function,
        with a dashed threshold line indicating the maximum allowable deviation (√2 * max_dev).


        """
        # Return an empty placeholder chart if there are no mappings to display
        if mappings.empty:
            return figure(
                title="No Deviation Data Available",
                width=800,
                height=400
            )

        chart = figure(
            title="Test Point Deviations from Matched Ideal Functions",
            x_axis_label='X Axis',
            y_axis_label='Deviation (ΔY)',
            width=1000,
            height=400,
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )

        # Same color scheme as the mapping chart for consistency
        color_lookup = {
            matched_funcs[1]: '#e63946',
            matched_funcs[2]: '#2a9d8f',
            matched_funcs[3]: '#f4a261',
            matched_funcs[4]: '#6a0572'
        }

        # Plot deviation of each test point, grouped by matched ideal function
        for i_idx in mappings['ideal_function_index'].unique():
            group = mappings[mappings['ideal_function_index'] == i_idx]

            chart.scatter(
                group['x'],
                group['deviation'],
                size=6,
                color=color_lookup[i_idx],
                alpha=0.65,
                legend_label=f'Ideal y{i_idx}'
            )

        # Draw a dashed horizontal threshold line for each matched function
        # Threshold = max_deviation * √2 as per assignment specification
        x_range = [mappings['x'].min(), mappings['x'].max()]
        for t_idx, i_idx in matched_funcs.items():
            threshold_val = max_deviations[t_idx] * np.sqrt(2)

            chart.line(
                x_range,
                [threshold_val, threshold_val],
                line_width=2,
                line_dash='dashed',
                color=color_lookup[i_idx],
                alpha=0.6,
                legend_label=f'Threshold y{i_idx}'
            )

        chart.legend.location = "top_right"
        chart.legend.click_policy = "hide"

        return chart

    def create_all_visualizations(self, train_df, test_df, ideal_df,
                                  matched_funcs, mappings, max_deviations):
        """
        Master method that builds all charts and saves them as a single
        interactive HTML file using Bokeh.


        """
        print("\n" + "=" * 60)
        print("RENDERING VISUALIZATIONS")
        print("=" * 60)

        # Set the output HTML file for all Bokeh plots
        output_file("assignment_visualizations.html")

        print("Building training vs ideal charts...")
        training_charts = self.build_training_charts(train_df, ideal_df, matched_funcs)

        print("Building test mapping chart...")
        mapping_chart = self.build_mapping_chart(test_df, mappings, ideal_df, matched_funcs)

        print("Building deviation chart...")
        deviation_chart = self.build_deviation_chart(mappings, max_deviations, matched_funcs)

        # Arrange all plots in a vertical layout with training charts in a 2x2 grid on top
        layout = column(
            row(training_charts[0], training_charts[1]),
            row(training_charts[2], training_charts[3]),
            mapping_chart,
            deviation_chart
        )

        # Save the full layout to HTML
        save(layout)

        print("Visualizations saved to: assignment_visualizations.html")
        print("Open this file in a web browser to explore interactive plots")
        print("=" * 60 + "\n")

    def export_pngs(self, train_df, mappings, ideal_df, matched_funcs, out_dir='visuals'):
        """
        Exports static PNG previews using matplotlib for quick offline viewing.

        Generates two files:
            - training_vs_ideal.png : 2x2 grid of training vs ideal overlays
            - mappings.png          : scatter plot of mapped test points


        """
        # Create the output directory if it doesn't already exist
        os.makedirs(out_dir, exist_ok=True)

        # ── Training vs Ideal 2x2 grid ──────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        # Use a warm matplotlib color cycle for the 4 subplots
        mpl_colors = ['#e63946', '#2a9d8f', '#e9c46a', '#6a0572']
        x_vals = ideal_df['x'].values

        for i in range(1, 5):
            ax = axes[i - 1]

            # Plot training scatter if column exists
            if f'y{i}' in train_df.columns:
                ax.scatter(
                    train_df['x'],
                    train_df[f'y{i}'],
                    s=8,
                    color=mpl_colors[i - 1],
                    alpha=0.65,
                    label=f'Training y{i}'
                )

            # Overlay matched ideal function line
            i_idx = matched_funcs.get(i)
            if i_idx and f'y{i_idx}' in ideal_df.columns:
                ax.plot(
                    x_vals,
                    ideal_df[f'y{i_idx}'].values,
                    color='#264653',
                    linewidth=1.5,
                    label=f'Ideal y{i_idx}'
                )

            ax.set_title(f'Training y{i}  vs  Ideal y{i_idx}', fontsize=10)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend(fontsize=7)

        out_path_1 = os.path.join(out_dir, 'training_vs_ideal.png')
        fig.tight_layout()
        fig.savefig(out_path_1, dpi=150)
        plt.close(fig)

        # ── Mapped test points scatter ───────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))

        if mappings is not None and not mappings.empty:
            # Color each point by its assigned ideal function index
            sc = ax.scatter(
                mappings['x'],
                mappings['y'],
                c=mappings['ideal_function_index'],
                cmap='plasma',          # Changed from tab10 to plasma for better contrast
                s=20,
                alpha=0.8
            )
            fig.colorbar(sc, ax=ax, label='Ideal Function Index')
        else:
            # Display a message if no test points were mapped
            ax.text(0.5, 0.5, 'No mappings available', ha='center', va='center', fontsize=12)

        ax.set_title('Mapped Test Points', fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        out_path_2 = os.path.join(out_dir, 'mappings.png')
        fig.savefig(out_path_2, dpi=150)
        plt.close(fig)

        return out_path_1, out_path_2