import plotly.express as px

def graph_fun(Y, y):
    fig = px.scatter(None, x = Y[0, :], y = Y[1, :],
                 labels={
                     "x": "Dimension 1",
                     "y": "Dimension 2",
                 },
                 opacity=1, color=y.astype(str))

    # Change the background color of a graph
    fig.update_layout(dict(plot_bgcolor = 'white'))

    # Update axis lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                 showline=True, linewidth=1, linecolor='black')

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                 showline=True, linewidth=1, linecolor='black')

    # Setting the name of the pattern
    fig.update_layout(title_text="distributed t-SNE")

    # Update marker size
    fig.update_traces(marker=dict(size=3))
    fig.show()

