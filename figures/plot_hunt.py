from stacks import Stack
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def plot_stack(st: Stack):
    color_map = {
        "houston": "blue",
        "north": "orange",
        "south": "red",
        "west": "green",
    }

    # create a DataFrame with "x_tick", "width", "height", "name" columns
    df = pd.DataFrame.from_dict(st.plot_data())
    df["color"] = df.name.map(color_map)

    # plot each group by zone
    fig, ax = plt.subplots()
    for name, grp in df.groupby("name"):
        ax.bar(
            x=grp.x_tick,
            height=grp.height,
            width=grp.width,
            color=grp.color,
            label=name
        )

    ax.legend()
    plt.show()

if __name__ == "__main__":
    # read in the sample data and make a stack for each region
    df = pd.read_csv("figures/hunt_aug2024_sample.csv")

    stacks = []
    for name, grp in df.groupby("zone"):
        stack = Stack.from_blocks(
            grp.mw.to_numpy(),
            grp.price.to_numpy(),
            name=name
        )
        stacks.append(stack)
    stack = reduce(np.add, stacks)

    plot_stack(stack)
