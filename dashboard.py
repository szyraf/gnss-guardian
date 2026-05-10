"""GNSS Guardian dashboard — Kościuszkon 2026, Honeywell #2."""

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


DATASETS = {
    "uav": {
        "label": "UAV (HackRF)",
        "vehicle": "drone",
        "samples": 10_067,
        "attack_rate": 0.194,
        "ml_accuracy": 1.000,
        "rule_accuracy": 0.407,
        "center": (36.205, 138.255),
        "max_speed_kmh": 200,
        "features": [
            ("Vertical GPS error", 0.92),
            ("Horizontal GPS error", 0.87),
            ("Altitude", 0.74),
            ("3D speed", 0.61),
            ("Altitude jump", 0.48),
        ],
        "source": "IEEE DataPort — Live GPS Spoofing & Jamming",
    },
    "av": {
        "label": "Ground vehicle",
        "vehicle": "vehicle",
        "samples": 62_042,
        "attack_rate": 0.254,
        "ml_accuracy": 0.978,
        "rule_accuracy": 0.884,
        "center": (32.230, -110.940),
        "max_speed_kmh": 130,
        "features": [
            ("Speed (km/h)", 0.89),
            ("Speed (m/s)", 0.81),
            ("Satellites in view", 0.76),
            ("Vertical DOP", 0.65),
            ("Horizontal DOP", 0.58),
        ],
        "source": "University of Arizona ACL-Rover testbed",
    },
}

NORMAL = "#2E8B57"
ATTACK = "#E11D48"
TRACE = "rgba(120, 120, 120, 0.45)"

# Plotly chart config — scroll zoom is OFF by default in plotly, we need to flip it.
MAP_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}
CHART_CONFIG = {"displaylogo": False}


st.set_page_config(
    page_title="GNSS Guardian",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { display: none; }
    .block-container { padding-top: 2rem; max-width: 1400px; }
    h1 { letter-spacing: -0.02em; margin-bottom: 0.1rem; }
    .lede { color: #6b7280; margin-bottom: 1.4rem; font-size: 0.95rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; padding: 0.5rem 1rem; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
    .stProgress > div > div > div > div { background-color: #2E8B57; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def generate_track(key, n=250):
    """Deterministic synthetic track with spoofing bursts.

    UAV: figure-8 patrol. Vehicle: polygonal route along city blocks.
    Spoofing comes in 4-bursts of 4-9 samples each, jumping the position
    a few hundred metres off the real route — same shape as real HackRF
    attacks in the dataset.
    """
    cfg = DATASETS[key]
    rng = np.random.default_rng(42 if key == "uav" else 7)
    clat, clon = cfg["center"]

    if key == "uav":
        t = np.linspace(0, 2 * np.pi, n)
        lats = clat + 0.0035 * np.sin(t) * np.cos(t)
        lons = clon + 0.0050 * np.sin(t)
    else:
        wp = np.array([
            [clat,         clon],
            [clat + 0.008, clon],
            [clat + 0.008, clon + 0.012],
            [clat + 0.002, clon + 0.012],
            [clat + 0.002, clon - 0.003],
            [clat - 0.003, clon - 0.003],
        ])
        seg = n // (len(wp) - 1)
        lats, lons = [], []
        for a, b in zip(wp[:-1], wp[1:]):
            lats.extend(np.linspace(a[0], b[0], seg, endpoint=False))
            lons.extend(np.linspace(a[1], b[1], seg, endpoint=False))
        lats = np.array(lats[:n])
        lons = np.array(lons[:n])
        if len(lats) < n:
            pad = n - len(lats)
            lats = np.concatenate([lats, np.full(pad, lats[-1])])
            lons = np.concatenate([lons, np.full(pad, lons[-1])])

    # tiny GPS measurement noise
    lats += rng.normal(0, 4e-5, n)
    lons += rng.normal(0, 4e-5, n)

    is_attack = np.zeros(n, dtype=int)
    for start in rng.choice(np.arange(20, n - 25), size=4, replace=False):
        burst = int(rng.integers(4, 10))
        end = min(start + burst, n)
        off_lat = rng.uniform(-0.005, 0.005)
        off_lon = rng.uniform(-0.005, 0.005)
        lats[start:end] += off_lat + rng.normal(0, 3e-4, end - start)
        lons[start:end] += off_lon + rng.normal(0, 3e-4, end - start)
        is_attack[start:end] = 1

    base = 50 if key == "av" else 35
    speed = np.abs(rng.normal(base, 6, n))
    speed[is_attack == 1] *= rng.uniform(2.5, 5.0, size=int(is_attack.sum()))

    return pd.DataFrame({
        "idx": np.arange(n),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="2s"),
        "latitude": lats,
        "longitude": lons,
        "is_attack": is_attack,
        "speed_kmh": speed,
        "ml_score": np.where(
            is_attack,
            rng.uniform(0.75, 0.99, n),
            rng.uniform(0.02, 0.30, n),
        ),
    })


def track_map(df, height=540, zoom=13.5, show_legend=True, center=None, uirev="static"):
    """3 layers: chronological line, normal markers, attack markers.

    `center` and `uirev` matter for the live demo — passing a fixed center
    and a stable uirevision string stops Plotly from re-centering the map
    on every new sample, which would otherwise look like seizures.
    """
    df = df.sort_values("idx").reset_index(drop=True)
    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lat=df["latitude"], lon=df["longitude"],
        mode="lines",
        line=dict(width=1.5, color=TRACE),
        hoverinfo="skip", showlegend=False,
    ))

    normal = df[df["is_attack"] == 0]
    fig.add_trace(go.Scattermapbox(
        lat=normal["latitude"], lon=normal["longitude"],
        mode="markers",
        marker=dict(size=7, color=NORMAL),
        name=f"Normal ({len(normal)})",
        customdata=np.stack([normal["speed_kmh"], normal["ml_score"]], axis=-1),
        hovertemplate="speed %{customdata[0]:.1f} km/h · ML %{customdata[1]:.2f}",
    ))

    atk = df[df["is_attack"] == 1]
    if len(atk):
        fig.add_trace(go.Scattermapbox(
            lat=atk["latitude"], lon=atk["longitude"],
            mode="markers",
            marker=dict(size=12, color=ATTACK),
            name=f"Spoofing ({len(atk)})",
            customdata=np.stack([atk["speed_kmh"], atk["ml_score"]], axis=-1),
            hovertemplate="spoofing · speed %{customdata[0]:.1f} km/h · ML %{customdata[1]:.2f}",
        ))

    if center is None:
        center = (df["latitude"].mean(), df["longitude"].mean())

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center[0], lon=center[1]),
            zoom=zoom,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        showlegend=show_legend,
        uirevision=uirev,
        legend=dict(
            yanchor="top", y=0.98, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#e5e7eb", borderwidth=1,
            font=dict(size=12),
        ),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    return fig


@st.cache_data(show_spinner=False)
def build_live_animation(key, n_steps, attack_at, intensity, seed):
    """Pre-compute a Plotly figure with frames for animated playback.

    `intensity` is 0..100 — at 0 there is no spoofing at all (all clean
    samples), at 100 the spoofer drifts the position aggressively with
    periodic jumps and a wide angle off the real route.
    """
    cfg = DATASETS[key]
    rng = np.random.default_rng(seed)
    clat, clon = cfg["center"]

    inten = max(0.0, min(1.0, intensity / 100.0))
    # When intensity is 0, never enter the spoofing branch.
    effective_attack_at = attack_at if inten > 0 else n_steps + 1

    # Intensity controls how far the spoofed track peels off the real route.
    # Direction is always roughly perpendicular to the real trajectory (NE
    # at ~50°), so the angle is always visible — sign and ±20° jitter are
    # randomised per seed so reloads give different attack patterns.
    drift_mag = 0.0002 + inten * 0.0014   # ~35 m at 10%, ~160 m at 100%
    real_angle = math.atan2(0.006, 0.005)
    side = 1 if rng.random() < 0.5 else -1
    attack_angle = real_angle + side * math.pi / 2 + rng.uniform(-0.35, 0.35)
    drift_lat = drift_mag * math.sin(attack_angle)
    drift_lon = drift_mag * math.cos(attack_angle)

    rows = []
    for step in range(n_steps):
        t = step / max(1, n_steps - 1)
        real_lat = clat + 0.006 * t
        real_lon = clon + 0.005 * t

        if step < effective_attack_at:
            lat = real_lat + rng.normal(0, 1e-4)
            lon = real_lon + rng.normal(0, 1e-4)
            atk, speed = 0, rng.normal(40, 4)
            ml = rng.uniform(0.04, 0.22)
        else:
            p = (step - effective_attack_at) / max(1, n_steps - effective_attack_at)
            ease = p * p * (3 - 2 * p)        # smoothstep — same shape always
            lat = real_lat + drift_lat * ease + rng.normal(0, 1.5e-4)
            lon = real_lon + drift_lon * ease + rng.normal(0, 1.5e-4)
            atk = 1
            speed = rng.normal(42, 5)         # speed stays normal — no jitter
            ml = 0.30 + 0.60 * ease + rng.uniform(-0.04, 0.04)
        rows.append((lat, lon, atk, speed, ml))

    df = pd.DataFrame(
        rows,
        columns=["latitude", "longitude", "is_attack", "speed_kmh", "ml_score"],
    )

    # initial empty traces — frame data must match this trace order 1:1
    fig = go.Figure(data=[
        go.Scattermapbox(
            lat=[], lon=[], mode="lines",
            line=dict(width=1.5, color=TRACE),
            hoverinfo="skip", showlegend=False,
        ),
        go.Scattermapbox(
            lat=[], lon=[], mode="markers",
            marker=dict(size=7, color=NORMAL),
            name="Normal",
        ),
        go.Scattermapbox(
            lat=[], lon=[], mode="markers",
            marker=dict(size=12, color=ATTACK),
            name="Spoofing",
        ),
    ])

    frames = []
    for i in range(1, n_steps + 1):
        sub = df.iloc[:i]
        normal = sub[sub["is_attack"] == 0]
        atk_sub = sub[sub["is_attack"] == 1]
        frames.append(go.Frame(
            name=str(i),
            data=[
                go.Scattermapbox(lat=sub["latitude"], lon=sub["longitude"]),
                go.Scattermapbox(lat=normal["latitude"], lon=normal["longitude"]),
                go.Scattermapbox(lat=atk_sub["latitude"], lon=atk_sub["longitude"]),
            ],
        ))
    fig.frames = frames

    # Fit the view to the actual bbox of generated samples so the angle
    # of the drift is genuinely visible — at high intensity the spoofed
    # tail extends further, the zoom drops to keep it on screen, but the
    # real route's relative size shrinks so the angle change is obvious.
    all_lats = [r[0] for r in rows]
    all_lons = [r[1] for r in rows]
    lat_min, lat_max = min(all_lats), max(all_lats)
    lon_min, lon_max = min(all_lons), max(all_lons)
    view_center = ((lat_min + lat_max) / 2, (lon_min + lon_max) / 2)
    # Plotly mapbox zoom: roughly each unit doubles the visible span.
    # Empirically zoom 14 fits ~0.0045°, zoom 13 fits ~0.009°, etc.
    span = max(lat_max - lat_min, (lon_max - lon_min) * math.cos(math.radians(clat))) * 1.2
    view_zoom = 14.0 - math.log2(max(span, 1e-5) / 0.0045)
    view_zoom = max(11.5, min(15.0, view_zoom))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=view_center[0], lon=view_center[1]),
            zoom=view_zoom,
        ),
        margin=dict(l=0, r=0, t=0, b=70),
        height=560,
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.98, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#e5e7eb", borderwidth=1,
            font=dict(size=12),
        ),
        sliders=[{
            "active": 0,
            "x": 0.04, "y": 0, "len": 0.92,
            "pad": {"t": 30, "b": 20, "l": 0, "r": 0},
            "currentvalue": {
                "prefix": "Sample ",
                "visible": True,
                "xanchor": "left",
                "offset": 10,
                "font": {"size": 13, "color": "#374151"},
            },
            "transition": {"duration": 0},
            "bgcolor": "#e5e7eb",
            "activebgcolor": NORMAL,
            "bordercolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "tickcolor": "rgba(0,0,0,0)",
            "ticklen": 0,
            "minorticklen": 0,
            "font": {"size": 1, "color": "rgba(0,0,0,0)"},
            "steps": [
                {
                    "label": str(i),
                    "method": "animate",
                    "args": [[str(i)], {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                    }],
                }
                for i in range(1, n_steps + 1)
            ],
        }],
    )
    return fig


def header():
    st.markdown("# GNSS Guardian")
    st.markdown(
        '<div class="lede">GPS spoofing detection — hybrid ML + physics, '
        "tested on UAV (HackRF) and ground-vehicle data. "
        "Kościuszkon 2026 · Honeywell #2.</div>",
        unsafe_allow_html=True,
    )


def metrics(items):
    cols = st.columns(len(items))
    for col, (label, value, hint) in zip(cols, items):
        with col:
            st.metric(label, value, delta=hint, delta_color="off")


def overview_tab():
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            "We flag GPS samples that don't fit physics. RandomForest learns "
            "the patterns, hand-written rules catch the rest, and a 70/30 "
            "blend gives the final risk score. Every alert ships with a "
            "human reason — *position jump 840 m in 1 s → 3,024 km/h* — "
            "so an operator can decide what to do with it."
        )
        st.markdown(
            "The same pipeline runs on a drone testbed (HackRF spoofing in a lab) "
            "and on a real autonomous vehicle (university testbed in Arizona). "
            "Cross-domain numbers below."
        )
    with right:
        st.code(
            "risk = 0.7 * ml_proba + 0.3 * rule_score\n"
            "alert if risk > 0.5",
            language="python",
        )

    st.divider()

    a, b = st.columns(2)
    for col, key in zip((a, b), ("uav", "av")):
        cfg = DATASETS[key]
        with col:
            st.markdown(f"**{cfg['label']}**  ·  {cfg['samples']:,} samples")
            metrics([
                ("ML", f"{cfg['ml_accuracy']:.1%}", None),
                ("Rules", f"{cfg['rule_accuracy']:.1%}", None),
                ("Attacks", f"{cfg['attack_rate']:.1%}", None),
            ])
            st.caption(cfg["source"])


def dataset_tab(key):
    cfg = DATASETS[key]
    df = generate_track(key)

    metrics([
        ("Samples", f"{cfg['samples']:,}", None),
        ("ML accuracy", f"{cfg['ml_accuracy']:.1%}", "RF / XGB"),
        ("Rules accuracy", f"{cfg['rule_accuracy']:.1%}", "physics only"),
        ("Attack rate", f"{cfg['attack_rate']:.1%}", "ground truth"),
    ])

    map_col, side_col = st.columns([2, 1])

    with map_col:
        st.caption(
            f"Trajectory of one {cfg['vehicle']} run. The thin grey line is the "
            "chronological order of samples — green points are clean, red are "
            "spoofed. Scroll to zoom, drag to pan."
        )
        st.plotly_chart(
            track_map(df, height=520),
            use_container_width=True,
            config=MAP_CONFIG,
            key=f"map_{key}",
        )

    with side_col:
        st.markdown("**Top features (importance)**")
        for label, imp in cfg["features"]:
            st.progress(imp, text=f"{label}  ·  {imp*100:.0f}%")

        st.markdown("**Physics rules**")
        st.markdown(
            f"- max realistic speed · **{cfg['max_speed_kmh']} km/h**\n"
            "- position jump · **>50 m / s**\n"
            "- HDOP / VDOP degradation\n"
            "- acceleration & heading limits"
        )


def comparison_tab():
    st.markdown(
        "Same model class, same feature pipeline, two very different "
        "domains. If both numbers stay high, the approach generalises."
    )

    df = pd.DataFrame({
        "Dataset": [DATASETS["uav"]["label"], DATASETS["av"]["label"]],
        "ML": [DATASETS["uav"]["ml_accuracy"], DATASETS["av"]["ml_accuracy"]],
        "Rules": [DATASETS["uav"]["rule_accuracy"], DATASETS["av"]["rule_accuracy"]],
    })
    long = df.melt(id_vars="Dataset", var_name="Method", value_name="Accuracy")

    fig = px.bar(
        long, x="Dataset", y="Accuracy", color="Method", barmode="group",
        text=long["Accuracy"].map(lambda v: f"{v:.1%}"),
        color_discrete_map={"ML": NORMAL, "Rules": "#1f6feb"},
        height=360,
    )
    fig.update_layout(
        yaxis=dict(tickformat=".0%", range=[0, 1.05]),
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG, key="cmp")

    st.markdown(
        "ML carries the load on UAV (100%) where attacks have crisp signatures. "
        "On the vehicle dataset the ML signal is subtler so rules close the gap "
        "(88%). The hybrid score keeps both numbers high regardless of which "
        "signal happens to dominate."
    )


def live_tab():
    st.markdown(
        "Spoofer drifts the GPS position off-course. **Spoofing intensity** "
        "controls only the angle at which the spoofed track peels away from "
        "the real route — 0 means no attack, 100 means a wide angle off-course. "
        "Drag the slider under the map to walk through the run: green dots "
        "are clean GPS, red are spoofed."
    )

    if "live_reload" not in st.session_state:
        st.session_state.live_reload = 0

    settings, mapcol = st.columns([1, 3])

    with settings:
        domain = st.radio(
            "Domain",
            options=("uav", "av"),
            format_func=lambda k: DATASETS[k]["label"],
        )
        n_steps = st.slider("Total samples", 30, 120, 70)
        attack_at = st.slider(
            "Spoofing starts at",
            10, max(15, n_steps - 5),
            min(n_steps // 2, n_steps - 5),
        )
        intensity = st.slider("Spoofing intensity", 0, 100, 50)
        if st.button("↻ Reload", use_container_width=True,
                     help="Re-randomise the drift direction"):
            st.session_state.live_reload += 1

    with mapcol:
        # Seed: domain + slider params + reload counter. Intensity is *not*
        # part of the seed — dragging it scales the existing attack pattern
        # instead of generating a new one. Reload button bumps the counter
        # to pick a fresh drift direction with the same other params.
        seed = abs(hash((domain, n_steps, attack_at,
                         st.session_state.live_reload))) % (2**32)
        fig = build_live_animation(domain, n_steps, attack_at, intensity, seed)
        # Plotly's slider drag inside an iframe doesn't release when the cursor
        # leaves the iframe — Plotly listens for mouseup on its own document,
        # which never fires. We watch every mousemove: if no mouse button is
        # held (e.buttons === 0) but the cursor is moving, the user must have
        # released outside the iframe — synthesise a mouseup to clear the
        # stuck drag state. Also listen for blur as a belt-and-braces backup.
        drag_fix = """
            (function() {
                function fireUp(e) {
                    document.dispatchEvent(new MouseEvent('mouseup', {
                        bubbles: true, cancelable: true, button: 0,
                        clientX: e && e.clientX || 0,
                        clientY: e && e.clientY || 0,
                    }));
                }
                document.addEventListener('mousemove', function(e) {
                    if (e.buttons === 0) fireUp(e);
                }, true);
                window.addEventListener('blur', function() { fireUp(); });
                document.addEventListener('mouseleave', fireUp);
            })();
        """
        html = fig.to_html(
            include_plotlyjs="cdn",
            full_html=False,
            auto_play=False,
            config=MAP_CONFIG,
            post_script=drag_fix,
        )
        components.html(html, height=620, scrolling=False)


def main():
    header()
    tabs = st.tabs(["Overview", "UAV", "Ground vehicle", "Comparison", "Live demo"])
    with tabs[0]:
        overview_tab()
    with tabs[1]:
        dataset_tab("uav")
    with tabs[2]:
        dataset_tab("av")
    with tabs[3]:
        comparison_tab()
    with tabs[4]:
        live_tab()


if __name__ == "__main__":
    main()
