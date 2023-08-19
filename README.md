# 3Data-Visualizer

## Project Overview

The goal of this project is to showcase various data visualizations using Dash. It provides examples of different types of charts, including line plots, box plots, scatter plots and bar charts. The application allows users to interact with the visualizations by exploring data, applying filters, and customizing the display.

## Features
- Random Select: This feature allows you to randomly select data points from a given dataset.
- Color: Select graph color manually.
- Row range: You can manually select rows which is use for visualization.
- Scatter plots: Explore relationships between variables.
- Line plots: Visualize trends and patterns over time.
- Box Plots: Compare data across different categories.
- Bar Plots: Display categorical data using rectangular bars with lengths proportional to the values they represent.

## Usage

1. Run the application:
>python app.py

2. Open a web browser and navigate to `http://localhost:8050`.

3. Explore the different visualizations by interacting with the controls and filters provided.

## Project Structure

The project structure is organized as follows:

- `Project.py`: Main application file that defines the Dash app layout and callbacks.
- `data/`: Directory containing sample data files used for visualization.
- `assets/`: Directory for storing CSS files.

## Dependencies

The project relies on the following dependencies:

- Dash: Main framework for building the web application.
- Plotly: Library for creating interactive and dynamic visualizations.
- Pandas: Data manipulation and analysis library.
- Other dependencies listed in the `requirements.txt` file.

For detailed information about the dependencies and their versions, please refer to the `requirements.txt` file.



