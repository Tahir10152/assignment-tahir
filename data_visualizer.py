from bokeh.plotting import figure, output_file, save, show
from bokeh.layouts import column, row
from bokeh.models import HoverTool, Legend
import numpy as np
import matplotlib.pyplot as plt
import os


class Visualizer:
	def __init__(self):
		self.plots = []

	def plot_training_and_ideal(self, training_data, ideal_functions, selected_functions):
		colors = ['red', 'blue', 'green', 'orange']
		plots = []

		for trainNumber in range(1, 5):
			p = figure(
				title=f"Training Function {trainNumber} vs Ideal Function {selected_functions[trainNumber]}",
				x_axis_label='X',
				y_axis_label='Y',
				width=500,
				height=400,
				tools="pan,wheel_zoom,box_zoom,reset,save"
			)

			p.scatter(
				training_data['x'],
				training_data[f'y{trainNumber}'],
				size=6,
				color=colors[trainNumber - 1],
				alpha=0.6,
				legend_label=f'Training {trainNumber}'
			)

			idealNumber = selected_functions[trainNumber]
			p.line(
				ideal_functions['x'],
				ideal_functions[f'y{idealNumber}'],
				line_width=2,
				color='black',
				alpha=0.8,
				legend_label=f'Ideal {idealNumber}'
			)

			p.legend.location = "top_left"
			p.legend.click_policy = "hide"

			plots.append(p)

		return plots

	def plot_test_mappings(self, test_data, mappings, ideal_functions, selected_functions):
		p = figure(
			title="Test Data Mappings to Ideal Functions",
			x_axis_label='X',
			y_axis_label='Y',
			width=1000,
			height=600,
			tools="pan,wheel_zoom,box_zoom,reset,save"
		)

		colors_map = {
			selected_functions[1]: 'red',
			selected_functions[2]: 'blue',
			selected_functions[3]: 'green',
			selected_functions[4]: 'orange'
		}

		for trainNumber, idealNumber in selected_functions.items():
			p.line(
				ideal_functions['x'],
				ideal_functions[f'y{idealNumber}'],
				line_width=2,
				color=colors_map[idealNumber],
				alpha=0.5,
				legend_label=f'Ideal Function {idealNumber}'
			)

		if not mappings.empty:
			for idealNumber in mappings['ideal_function_index'].unique():
				subset = mappings[mappings['ideal_function_index'] == idealNumber]
				p.scatter(
					subset['x'],
					subset['y'],
					size=8,
					color=colors_map[idealNumber],
					alpha=0.8,
					legend_label=f'Test → Ideal {idealNumber}'
				)

		hover = HoverTool(
			tooltips=[
				("X", "@x{0.00}"),
				("Y", "@y{0.00}"),
			]
		)
		p.add_tools(hover)

		p.legend.location = "top_left"
		p.legend.click_policy = "hide"

		return p

	def plot_deviations(self, mappings, max_deviations, selected_functions):
		if mappings.empty:
			p = figure(
				title="No Mappings to Display",
				width=800,
				height=400
			)
			return p

		p = figure(
			title="Deviations of Test Points from Ideal Functions",
			x_axis_label='X',
			y_axis_label='Deviation (Delta Y)',
			width=1000,
			height=400,
			tools="pan,wheel_zoom,box_zoom,reset,save"
		)

		colors_map = {
			selected_functions[1]: 'red',
			selected_functions[2]: 'blue',
			selected_functions[3]: 'green',
			selected_functions[4]: 'orange'
		}

		for idealNumber in mappings['ideal_function_index'].unique():
			subset = mappings[mappings['ideal_function_index'] == idealNumber]

			p.scatter(
				subset['x'],
				subset['deviation'],
				size=6,
				color=colors_map[idealNumber],
				alpha=0.6,
				legend_label=f'Ideal {idealNumber}'
			)

		for trainNumber, idealNumber in selected_functions.items():
			threshold = max_deviations[trainNumber] * np.sqrt(2)
			p.line(
				[mappings['x'].min(), mappings['x'].max()],
				[threshold, threshold],
				line_width=2,
				line_dash='dashed',
				color=colors_map[idealNumber],
				alpha=0.5,
				legend_label=f'Threshold {idealNumber}'
			)

		p.legend.location = "top_right"
		p.legend.click_policy = "hide"

		return p

	def create_all_visualizations(self, training_data, test_data, ideal_functions,
								  selected_functions, mappings, max_deviations):
		print("\n" + "=" * 60)
		print("CREATING VISUALIZATIONS")
		print("=" * 60)

		output_file("assignment_visualizations.html")

		print("Creating training vs ideal function plots...")
		training_plots = self.plot_training_and_ideal(
			training_data,
			ideal_functions,
			selected_functions
		)

		print("Creating test mappings plot...")
		mapping_plot = self.plot_test_mappings(
			test_data,
			mappings,
			ideal_functions,
			selected_functions
		)

		print("Creating deviations plot...")
		deviation_plot = self.plot_deviations(
			mappings,
			max_deviations,
			selected_functions
		)

		layout = column(
			row(training_plots[0], training_plots[1]),
			row(training_plots[2], training_plots[3]),
			mapping_plot,
			deviation_plot
		)

		save(layout)

		print("Visualizations saved to: assignment_visualizations.html")
		print("Open this file in your web browser to view interactive plots")
		print("=" * 60 + "\n")

	def export_pngs(self, training_data, mappings, ideal_functions, selected_functions, out_dir='visuals'):
		"""Create simple PNG exports using matplotlib for quick previews."""
		os.makedirs(out_dir, exist_ok=True)

		# Training vs ideal grid
		fig, axes = plt.subplots(2, 2, figsize=(12, 8))
		axes = axes.flatten()
		colors = ['C0', 'C1', 'C2', 'C3']
		x_ideal = ideal_functions['x'].values

		for i in range(1, 5):
			ax = axes[i - 1]
			if f'y{i}' in training_data.columns:
				ax.scatter(training_data['x'], training_data[f'y{i}'], s=8, color=colors[i - 1], alpha=0.6)
			idealNumber = selected_functions.get(i)
			if idealNumber and f'y{idealNumber}' in ideal_functions.columns:
				ax.plot(x_ideal, ideal_functions[f'y{idealNumber}'].values, color='k', linewidth=1.5)
			ax.set_title(f'Train y{i} vs Ideal {idealNumber}')
			ax.set_xlabel('x')
			ax.set_ylabel('y')

		out1 = os.path.join(out_dir, 'training_vs_ideal.png')
		fig.tight_layout()
		fig.savefig(out1)
		plt.close(fig)

		# Mappings plot
		fig, ax = plt.subplots(figsize=(8, 6))
		if mappings is not None and not mappings.empty:
			sc = ax.scatter(mappings['x'], mappings['y'], c=mappings['ideal_function_index'], cmap='tab10', s=20)
			fig.colorbar(sc, ax=ax, label='Ideal Function')
		else:
			ax.text(0.5, 0.5, 'No mappings', ha='center')

		ax.set_title('Mapped Test Points')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		out2 = os.path.join(out_dir, 'mappings.png')
		fig.savefig(out2)
		plt.close(fig)

		return out1, out2

