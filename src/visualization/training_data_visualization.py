from src.utils.logging_util import load_metrics

import matplotlib.pyplot as plt
import json

def _plot_metrics(data):
    steps = json.loads(data['step'])

    for key, value in data.items():
        if key != 'step':  # Ensure step is always the x-axis
            plt.figure()
            values = json.loads(value)
            plt.plot(steps, values, marker='o', linestyle='-')
            plt.xlabel('step')
            plt.ylabel(key)
            plt.title(f'{key} vs step')
            plt.grid(True)
            plt.show()

def plot_training_data_from_file(model_name, file_name=None):
    data = load_metrics(model_name, file_name)
    _plot_metrics(data)

def plot_training_data(data:dict):
    _plot_metrics(data)