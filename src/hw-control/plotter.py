# %%
import pandas as pd
import plotly.express as px
import json
import os

# %%
# logging sampling interval (in ms)
SAMPLING_TIME = 1
PATH = "./results/"

# %%
with open(os.path.join(PATH, "hw-control.log"), "r") as file:
    lines = file.readlines()

lines = map(lambda line: line.replace("'", "\""), lines)
json_data = list(map(lambda line: json.loads(line), lines))

# new_json_data = []
# for i in range(0, len(json_data), 3):
#     measurement = {}
#     measurement.update(json_data[i])
#     measurement.update(json_data[i + 1])
#     measurement.update(json_data[i + 2])
#     new_json_data.append(measurement)
# json_data = new_json_data

# %%
df = pd.DataFrame(json_data)

# %%
fig = px.line_3d(df, x="stateEstimate.x", y="stateEstimate.y", z="stateEstimate.z")
fig.write_image(os.path.join(PATH, "hw/nav-dist", "3d-trajectory.png"))

# %%
cols = ["stateEstimate.x", "stateEstimate.y", "stateEstimate.z", "stabilizer.roll", "stabilizer.pitch", "stabilizer.yaw"]

for col in cols:
    fig = px.line()
    fig.update_layout(title_text=col,
            title_xanchor="center",
            title_x=0.5,
            xaxis_title="time",
            yaxis_title=col)
    fig.add_scatter(x=df.index * SAMPLING_TIME, y=df[col])
    fig.write_image(os.path.join(PATH, "hw/nav-dist", f"{col}.png"))


