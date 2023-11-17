# Visualizing documents
import itertools
import pandas as pd

# Define colors for the visualization to iterate over
colors = itertools.cycle(
    [
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#bcf60c",
        "#fabebe",
        "#008080",
        "#e6beff",
        "#9a6324",
        "#fffac8",
        "#800000",
        "#aaffc3",
        "#808000",
        "#ffd8b1",
        "#000075",
        "#808080",
        "#ffffff",
        "#000000",
    ]
)
color_key = {
    str(topic): next(colors) for topic in set(topic_model.topics_) if topic != -1
}

# Prepare dataframe and ignore outliers
df = pd.DataFrame(
    {
        "x": reduced_embeddings_2d[:, 0],
        "y": reduced_embeddings_2d[:, 1],
        "Topic": [str(t) for t in topic_model.topics_],
    }
)
df["Length"] = [len(doc) for doc in docs]
df = df.loc[df.Topic != "-1"]
df = df.loc[(df.y > -10) & (df.y < 10) & (df.x < 10) & (df.x > -10), :]
df["Topic"] = df["Topic"].astype("category")

# Get centroids of clusters
mean_df = df.groupby("Topic").mean().reset_index()
mean_df.Topic = mean_df.Topic.astype(int)
mean_df = mean_df.sort_values("Topic")

import seaborn as sns
from matplotlib import pyplot as plt
from adjustText import adjust_text
import matplotlib.patheffects as pe

fig = plt.figure(figsize=(16, 16))
sns.scatterplot(
    data=df,
    x="x",
    y="y",
    c=df["Topic"].map(color_key),
    alpha=0.4,
    sizes=(0.4, 10),
    size="Length",
)

# Annotate top 50 topics
texts, xs, ys = [], [], []
for row in mean_df.iterrows():
    topic = row[1]["Topic"]
    name = " - ".join(list(zip(*topic_model.get_topic(int(topic))))[0][:3])
    if int(topic) <= 50:
        xs.append(row[1]["x"])
        ys.append(row[1]["y"])
        texts.append(
            plt.text(
                row[1]["x"],
                row[1]["y"],
                name,
                size=10,
                ha="center",
                color=color_key[str(int(topic))],
                path_effects=[pe.withStroke(linewidth=0.5, foreground="black")],
            )
        )

# Adjust annotations such that they do not overlap
adjust_text(
    texts,
    x=xs,
    y=ys,
    time_lim=1,
    force_text=(0.01, 0.02),
    force_static=(0.01, 0.02),
    force_pull=(0.5, 0.5),
)
plt.show()
plt.savefig("scatterplot_2M.png", dpi=600)
