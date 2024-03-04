from umap import UMAP
import plotly.graph_objects as go

"""
Utils for visualization
"""

def plot_trending_papers(trend_df, trend_embeddings):
    """
    Function that takes in a dataframe of trending topics and their embeddings like returned by `sciencenow.postprocessing.trends.TrendPostprocessor`
    and returns an interactive figure that can be displayed as a html file in a browser.
    """
    reduced_embeddings = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(trend_embeddings)

    trend_df["x"] = reduced_embeddings[:,0]
    trend_df["y"] = reduced_embeddings[:,1]

    unique_topics = set(trend_df.Topic)

    fig = go.Figure()
    # loop over individual topics since we cannot group by topics with plotly scattergl
    for topic in unique_topics:
        selection = trend_df.loc[trend_df.Topic == topic, :]
        selection["text"] = ""
        name = "_".join(selection.Representation.tolist()[0][0:5])
        fig.add_trace(
            go.Scattergl(
                x=selection.x,
                y=selection.y,
                hovertext=selection.title,
                hoverinfo="text",
                mode='markers+text',
                name=name,
                textfont=dict(size=12),
                marker=dict(size=5, opacity=0.5)
            )
        )

    # Add grid in a 'plus' shape
    x_range = (trend_df.x.min() - abs((trend_df.x.min()) * .15), trend_df.x.max() + abs((trend_df.x.max()) * .15))
    y_range = (trend_df.y.min() - abs((trend_df.y.min()) * .15), trend_df.y.max() + abs((trend_df.y.max()) * .15))
    fig.add_shape(type="line",
                    x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                    line=dict(color="#CFD8DC", width=2))
    fig.add_shape(type="line",
                    x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                    line=dict(color="#9E9E9E", width=2))
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

    # Stylize layout
    fig.update_layout(
        template="simple_white",
        title={
            'text': f"Trending Topics & Papers",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig