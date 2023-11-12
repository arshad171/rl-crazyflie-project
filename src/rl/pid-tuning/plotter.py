import pandas as pd
import plotly.express as px
import os

def plot():
    # save_path = "/Users/arshadjaveed/My Data/Learning/Lund Notes/Classes/Sem 2/Project/test-pybullet/results"
    save_path = "./results"

    # paths = {
    #     "init": "/Users/arshadjaveed/My Data/Learning/Lund Notes/Classes/Sem 2/Project/test-pybullet/results/save-flight-init.png-03.25.2023_17.40.32",
    #     "tuned": "/Users/arshadjaveed/My Data/Learning/Lund Notes/Classes/Sem 2/Project/test-pybullet/results/save-flight-tuned.png-03.25.2023_17.40.53",
    #     "optimal": "/Users/arshadjaveed/My Data/Learning/Lund Notes/Classes/Sem 2/Project/test-pybullet/results/save-flight-optimal.png-03.25.2023_17.41.07",
    # }
    paths = {
        "init": "./results/rl/pid-tuning/raw/init",
        "tuned": "./results/rl/pid-tuning/raw/tuned",
        "default": "./results/rl/pid-tuning/raw/optimal",
    }

    colors = {
        "init": "black",
        "tuned": "blue",
        "default": "green",
    }

    plot_attrs = [
        ["x0", "x_r0"],
        ["y0", "y_r0"],
        ["z0", "z_r0"],
        ["r0", "r_r0"],
        ["p0", "p_r0"],
        ["ya0", "ya_r0"],
    ]

    for (attr, attr_ref) in plot_attrs:
        print(attr, attr_ref)
        fig = px.line()
        fig.update_layout(
            title_text=attr,
            title_xanchor="center",
            title_x=0.5,
            xaxis_title="time",
            yaxis_title=attr,
        )
        df_ref = None
        for path in paths:
            df_ref = pd.read_csv(
                filepath_or_buffer=os.path.join(paths[path], f"{attr_ref}.csv"),
                names=["x", "y"],
            )
            df = pd.read_csv(
                filepath_or_buffer=os.path.join(paths[path], f"{attr}.csv"),
                names=["x", "y"],
            )

            fig.add_scatter(x=df["x"], y=df["y"], name=path, line=dict(color=colors[path]))
        fig.add_scatter(x=df_ref["x"], y=df_ref["y"], name="ref", line=dict(color="red"))

        # fig.write_image(os.path.join(save_path, f"{attr}.png"))
        fig.show()


if __name__ == "__main__":
    plot()