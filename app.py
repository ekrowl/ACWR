import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# -----------------------------
# Load and prepare data
# -----------------------------

# Load workload data
workload_df = pd.read_csv("MasterWorkload.csv")
workload_df["Session Date"] = pd.to_datetime(workload_df["Session Date"])

# Load player positions
positions_df = pd.read_csv("positions.csv")
merged_df = workload_df.merge(positions_df, on="Player Name", how="right")

# Sort by player/date
merged_df = merged_df.sort_values(by=["Player Name", "Session Date"])

# Metrics to include
metrics = ["High Speed Running", "DSL"]

# -----------------------------
# Remove extreme outliers per player
# -----------------------------
def remove_extreme_highs(df, metric, multiplier=2.5):
    cleaned = pd.DataFrame()
    for player, group in df.groupby("Player Name"):
        Q1 = group[metric].quantile(0.25)
        Q3 = group[metric].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + multiplier * IQR
        # Keep all normal and low values; only remove unrealistically high spikes
        group = group[group[metric] <= upper]
        cleaned = pd.concat([cleaned, group])
    return cleaned

for metric in metrics:
    merged_df = remove_extreme_highs(merged_df, metric, multiplier=2.5)

# -----------------------------
# Calculate ACWR metrics
# -----------------------------
for metric in metrics:
    merged_df[f"Acute_{metric}"] = (
        merged_df.groupby("Player Name")[metric]
        .rolling(window=7, min_periods=1).mean()
        .reset_index(0, drop=True)
    )
    merged_df[f"Chronic_{metric}"] = (
        merged_df.groupby("Player Name")[metric]
        .rolling(window=28, min_periods=1).mean()
        .reset_index(0, drop=True)
    )
    merged_df[f"ACWR_{metric}"] = (
        merged_df[f"Acute_{metric}"] / merged_df[f"Chronic_{metric}"]
    )

# Get most recent entry per player
latest_df = (
    merged_df.sort_values("Session Date")
    .groupby("Player Name")
    .tail(1)
)

# -----------------------------
# Dash app
# -----------------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Workload ACWR Dashboard"),

    html.Label("Select Metric:"),
    dcc.Dropdown(
        id="metric-dropdown",
        options=[
            {"label": "High Speed Running", "value": "High Speed Running"},
            {"label": "DSL", "value": "DSL"}
        ],
        value="High Speed Running",
        clearable=False
    ),

    html.Label("Filter by Position:"),
    dcc.Dropdown(
        id="position-dropdown",
        options=[{"label": pos, "value": pos} for pos in positions_df["Position"].dropna().unique()],
        multi=False
    ),

    dcc.Graph(id="acwr-comparison-chart")
])

# -----------------------------
# Callback to update chart
# -----------------------------
@app.callback(
    Output("acwr-comparison-chart", "figure"),
    Input("position-dropdown", "value"),
    Input("metric-dropdown", "value")
)
def update_chart(selected_position, selected_metric):
    df = latest_df.copy()

    if selected_position:
        df = df[df["Position"] == selected_position]

    acute_col = f"Acute_{selected_metric}"
    chronic_col = f"Chronic_{selected_metric}"
    acwr_col = f"ACWR_{selected_metric}"

    # Handle NaNs
    df = df.dropna(subset=[acute_col, chronic_col], how="all")

    # Color based on ACWR thresholds
    bar_colors = df[acwr_col].apply(
        lambda x: "blue" if x < 0.8 else ("red" if x > 1.5 else "grey")
    )

    # Custom hover text showing both acute and chronic
    hover_texts = [
        f"<b>{name}</b><br>Acute: {acute:.1f}<br>Chronic: {chronic:.1f}"
        for name, acute, chronic in zip(df["Player Name"], df[acute_col], df[chronic_col])
    ]

    fig = go.Figure()

    # Acute bar (main)
    fig.add_trace(go.Bar(
        y=df["Player Name"],
        x=df[acute_col],
        orientation="h",
        name="Acute (7d avg)",
        marker_color=bar_colors,
        opacity=0.8,
        hovertext=hover_texts,
        hoverinfo="text"
    ))

    # Thin vertical chronic line
    for i, row in df.iterrows():
        fig.add_shape(
            type="line",
            x0=row[chronic_col],
            x1=row[chronic_col],
            y0=list(df["Player Name"]).index(row["Player Name"]) - 0.4,
            y1=list(df["Player Name"]).index(row["Player Name"]) + 0.4,
            line=dict(color="black", width=3)
        )

    fig.update_layout(
        title=f"{selected_metric} ACWR by Player",
        xaxis_title=selected_metric,
        yaxis_title="Player",
        barmode="overlay",
        height=700,
        template="plotly_white",
        showlegend=False
    )

    return fig


# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)

