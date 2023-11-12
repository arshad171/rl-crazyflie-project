import pandas as pd
import plotly.express as px
import os


def plot():
    path = "./results/rl-train/plt/test"

    plot_attrs = [
        "x0",
        "y0",
        "z0",
    ]

    df = pd.DataFrame(columns=plot_attrs)
    for attr in plot_attrs:
        print(attr)
        df_temp = pd.read_csv(
            filepath_or_buffer=os.path.join(path, f"{attr}.csv"),
            names=["t", attr],
        )
        df[attr] = df_temp[attr]

    fig = px.line_3d()
    fig.update_layout(
        width=1000,
        height=1000,
        title_text="Trajectory",
        scene=dict(
            xaxis=dict(range=[-1.5, 1.5]),
            yaxis=dict(range=[-1.5, 1.5]),
            zaxis=dict(range=[0, 1.5]),
        ),
    )
    fig.add_scatter3d(connectgaps=False, x=[0], y=[0], z=[0], name="start")
    fig.add_scatter3d(connectgaps=False, x=[0], y=[0], z=[1], name="goal")
    fig.add_scatter3d(
        connectgaps=True,
        x=df["x0"],
        y=df["y0"],
        z=df["z0"],
        mode="lines",
        name="trajectory",
    )

    fig.show()
    # fig.write_image(os.path.join(path, "3d-plot.png"))


if __name__ == "__main__":
    plot()
