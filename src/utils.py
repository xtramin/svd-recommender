import plotly.graph_objects as go
import numpy as np


def plot_losses(losses: list):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.array(range(1, len(losses) + 1)), y=losses, mode="lines+markers"
        )
    )

    fig.update_layout(
        title="Root Mean Squared Error After Each Epoch",
        xaxis_title="Epoch",
        yaxis_title="RMSE",
    )

    fig.show()
