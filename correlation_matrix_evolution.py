import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go

# --- App Config ---
st.set_page_config(
    page_title="Correlation Matrix Evolution",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# --- Dark Theme Styling ---
st.markdown(
    """
    <style>
    body, .stApp { background-color: #18191A; color: #F5F6FA; }
    .stSlider > div[data-baseweb="slider"] { color: #F5F6FA; }
    .stMetric { color: #F5F6FA; }
    .stPlotlyChart { background: #18191A !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar Controls ---
st.sidebar.title("Settings")
window_size = st.sidebar.slider(
    "Rolling Window Size (days)", min_value=10, max_value=100, value=30, step=1
)

# --- Synthetic Data Generation ---
np.random.seed(42)
num_assets = 5
num_days = 300
asset_names = [f"Asset {chr(65+i)}" for i in range(num_assets)]

# Create a random positive-definite covariance matrix
A = np.random.rand(num_assets, num_assets)
cov = np.dot(A, A.T)
# Normalize diagonal to 1 (unit variance)
cov = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))

# Generate correlated returns
data = np.random.multivariate_normal(
    mean=np.zeros(num_assets), cov=cov, size=num_days
)
df = pd.DataFrame(data, columns=asset_names)

# --- Rolling Correlation Calculation ---
def rolling_corr_matrices(df, window):
    corrs = []
    for i in range(window, len(df)+1):
        corr = df.iloc[i-window:i].corr().values
        corrs.append(corr)
    return np.array(corrs)

corr_matrices = rolling_corr_matrices(df, window_size)

# --- Average Correlation Metric ---
def avg_corr(mat):
    n = mat.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return mat[mask].mean()

current_corr = avg_corr(corr_matrices[-1])

# --- Plotly Animated Heatmap ---
frames = []
for t, mat in enumerate(corr_matrices):
    frames.append(
        go.Frame(
            data=[go.Heatmap(
                z=mat,
                x=asset_names,
                y=asset_names,
                zmin=-1, zmax=1,
                colorscale="RdBu",
                colorbar=dict(title="Corr", tickvals=[-1,0,1]),
                showscale=True
            )],
            name=str(t)
        )
    )

# Initial heatmap
data0 = corr_matrices[0]
heatmap = go.Heatmap(
    z=data0,
    x=asset_names,
    y=asset_names,
    zmin=-1, zmax=1,
    colorscale="RdBu",
    colorbar=dict(title="Corr", tickvals=[-1,0,1]),
    showscale=True
)

fig = go.Figure(
    data=[heatmap],
    layout=go.Layout(
        title=dict(
            text="<b>Correlation Matrix Evolution</b>",
            font=dict(size=28, color="#F5F6FA"),
            x=0.5
        ),
        autosize=True,
        width=800,
        height=800,
        margin=dict(l=80, r=80, t=100, b=80),
        paper_bgcolor="#18191A",
        plot_bgcolor="#18191A",
        font=dict(color="#F5F6FA", size=18),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.1,
                x=1.05,
                xanchor="right",
                yanchor="top",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 60, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0, "easing": "linear"}
                        }]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    )
                ]
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[[str(i)], {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate"
                        }],
                        label=str(i+window_size)
                    ) for i in range(len(corr_matrices))
                ],
                transition={"duration": 0},
                x=0.1,
                y=0,
                currentvalue={"prefix": "Day: ", "font": {"color": "#F5F6FA", "size": 20}},
                len=0.8
            )
        ]
    ),
    frames=frames
)

# --- Streamlit Layout ---
st.title(":rainbow[Correlation Matrix Evolution]")
st.markdown("""
    <div style='color:#F5F6FA; font-size:20px; margin-bottom:20px;'>
        Animated visualization of rolling correlation matrices for 5 synthetic assets.<br>
        <b>Color scale:</b> <span style='color:#FF4136;'>Red</span> = -1, <span style='color:#0074D9;'>Blue</span> = +1
    </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([4,1])
with col1:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
with col2:
    st.metric(
        label="Current Avg. Correlation",
        value=f"{current_corr:.2f}",
        delta=None,
        help="Average of off-diagonal elements in the latest correlation matrix."
    )
    st.markdown(f"<div style='margin-top:40px; color:#888;'>Window size: <b>{window_size}</b></div>", unsafe_allow_html=True)

st.markdown("""
    <style>
    .stMetric { font-size: 2.5em !important; }
    .stPlotlyChart { margin-top: 0px !important; }
    </style>
    """, unsafe_allow_html=True)
