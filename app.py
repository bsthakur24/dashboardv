"""
Corporate Meeting Load & Collaboration Effectiveness — Dashboard
Rebuilt to use ONLY the KPIs/features actually engineered in the
Data Cleaning (Module 4) and EDA (Module 5) notebooks, computed
live from pq_data_cleaned.csv rather than hard-coded numbers.

Run locally:   streamlit run app.py
Deploy:        push this file + requirements.txt + pq_data_cleaned.csv to
                a GitHub repo and point Streamlit Community Cloud at app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Meeting Load & Collaboration Analytics",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stMetric {background-color:#fff; border:1px solid #dbe3ec; border-radius:8px; padding:10px 14px;}
    div[data-testid="stMetricValue"] {font-size:1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
CAT_NAN_TOKENS = {"nan", "Nan", "NaN", "NAN", "None", ""}


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # The cleaning notebook's astype(str) step turned real missing values
    # into the literal string "Nan" for a handful of categorical columns
    # (and produced a stray Org_Nan dummy column). Treat those as missing
    # rather than as a real category so filters/charts aren't polluted.
    cat_cols = ["Organization", "FunctionType", "Level", "LevelDesignation", "SupervisorIndicator"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].replace(list(CAT_NAN_TOKENS), np.nan)

    df["MetricDate"] = pd.to_datetime(df["MetricDate"], errors="coerce")

    # Reverse one-hot Organization -> Department (same approach as the EDA
    # notebook), skipping the stray Org_Nan dummy.
    org_cols = [c for c in df.columns if c.startswith("Org_") and c != "Org_Nan"]
    if org_cols:
        org_block = df[org_cols]
        has_org = org_block.sum(axis=1) > 0
        df["Department"] = np.where(has_org, org_block.idxmax(axis=1).str.replace("Org_", "", regex=False), np.nan)

    # Drop the single fully-corrupted "Nan" row that survives from the
    # cleaning-notebook bug (all categoricals literally "Nan").
    df = df.dropna(subset=["Department"]).copy()

    return df


DATA_PATH = "pq_data_cleaned.csv"
df_raw = load_data(DATA_PATH)

# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------
st.sidebar.header("🔍 Filters")
st.sidebar.caption(
    "Filters apply to the employee-collaboration view (pq_data_cleaned.csv). "
    "This build uses only the engineered KPIs from the Module 4/5 notebooks — "
    "no meeting-log (mt_data) view is included, since that source wasn't part "
    "of this capstone's cleaned dataset."
)

min_date, max_date = df_raw["MetricDate"].min(), df_raw["MetricDate"].max()
date_range = st.sidebar.date_input(
    "Date range (week of)",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date.date(), max_date.date()

dept_opts = sorted(df_raw["Department"].dropna().unique().tolist())
dept_sel = st.sidebar.multiselect("Department", dept_opts, default=dept_opts)

level_opts = sorted(df_raw["Level"].dropna().unique().tolist())
level_sel = st.sidebar.multiselect("Level", level_opts, default=level_opts)

func_opts = sorted(df_raw["FunctionType"].dropna().unique().tolist())
func_sel = st.sidebar.multiselect("Function Type", func_opts, default=func_opts)

role_opts = sorted(df_raw["SupervisorIndicator"].dropna().unique().tolist())
role_sel = st.sidebar.multiselect("Role", role_opts, default=role_opts)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Data source: `{DATA_PATH}` "
    f"({df_raw.shape[0]:,} employee-weeks, {len(dept_opts)} departments, "
    f"{min_date.date()} – {max_date.date()})."
)

# --------------------------------------------------------------------------
# Apply filters
# --------------------------------------------------------------------------
mask = (
    (df_raw["MetricDate"].dt.date >= start_date)
    & (df_raw["MetricDate"].dt.date <= end_date)
    & (df_raw["Department"].isin(dept_sel))
    & (df_raw["Level"].isin(level_sel))
    & (df_raw["FunctionType"].isin(func_sel))
    & (df_raw["SupervisorIndicator"].isin(role_sel))
)
df = df_raw.loc[mask].copy()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown("## 📊 Corporate Meeting Load & Collaboration Effectiveness")
st.caption(
    f"Showing **{df['PersonId'].nunique() if 'PersonId' in df.columns else '—'} employees** · "
    f"**{len(df):,} employee-weeks** · {start_date} → {end_date}"
)

if df.empty:
    st.warning("No rows match the current filters. Adjust filters in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------
# KPI cards — exactly the features engineered in the cleaning/EDA notebooks
# --------------------------------------------------------------------------
c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric(
    "Avg Meeting Hrs / Week",
    f"{df['Meeting_hours'].mean():.1f}",
    help="Meeting_hours — mean weekly meeting hours (Module 4).",
)
c2.metric(
    "Avg Collaboration Hrs",
    f"{df['Collaboration_hours'].mean():.1f}",
    help="Collaboration_hours — mean weekly total collaboration hours (core EDA metric).",
)
c3.metric(
    "Employee Experience Score",
    f"{df['Employee_Experience_Score'].mean():.2f} / 5",
    help="Employee_Experience_Score — mean of eSat, Wellbeing, Work_Life_Balance (Module 4 composite).",
)
c4.metric(
    "Low-Value Meeting Ratio",
    f"{df['Low_Value_Meeting_Ratio'].mean() * 100:.1f}%",
    help="Low_Value_Meeting_Ratio — (Conflicting + Large-and-long) hours ÷ total meeting hours.",
)
c5.metric(
    "Overload Rate",
    f"{df['Overload_Flag'].mean() * 100:.1f}%",
    help="Overload_Flag — share of employee-weeks above their department's 75th-percentile meeting load.",
)
c6.metric(
    "High-Load, Low-Value Rate",
    f"{df['High_Load_Low_Value'].mean() * 100:.1f}%",
    help="High_Load_Low_Value — Overload_Flag AND Employee_Experience_Score below the org-wide median.",
)

# --------------------------------------------------------------------------
# Key risk callout — computed live from filtered data (matches EDA Section 3 heatmap)
# --------------------------------------------------------------------------
corr_cols = ["Meeting_hours", "Collaboration_hours", "Employee_Experience_Score",
             "After_Hours_Ratio", "Multitasking_Rate"]
corr = df[corr_cols].corr()
r_afterhours_wellbeing = df[["After_Hours_Ratio"]].join(df["Employee_Experience_Score"]).corr().iloc[0, 1]
r_meeting_experience = corr.loc["Meeting_hours", "Employee_Experience_Score"]
r_afterhours_experience = corr.loc["After_Hours_Ratio", "Employee_Experience_Score"]

st.markdown(
    f"""
    <div style="background:#fdecea;border-left:4px solid #d9534f;padding:12px 16px;border-radius:4px;">
    ⚠️ <b>Key risk:</b> After-hours collaboration ratio correlates at r ≈ {r_afterhours_experience:.2f}
    with Employee Experience Score, vs. r ≈ {r_meeting_experience:.2f} for raw meeting hours —
    after-hours load tracks employee experience more closely than volume alone.
    Use the tabs below to drill into where this is concentrated.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# --------------------------------------------------------------------------
# Tabs — same flow as the original dashboard (Trends / Department Comparison
# / Drill-Down / Data). "Meeting Log" tab dropped: no mt_data source in this
# cleaned dataset.
# --------------------------------------------------------------------------
tab_trends, tab_dept, tab_drill, tab_data = st.tabs(
    ["📈 Trends", "🏢 Department Comparison", "🔎 Drill-Down", "🗂️ Data"]
)

white_layout = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font_color="#333",
    margin=dict(l=10, r=10, t=40, b=10),
)

# ---------------- TRENDS TAB ----------------
with tab_trends:
    st.markdown("### Meeting Load & Sentiment Over Time")

    weekly = (
        df.groupby("Week_Number", as_index=False)
        .agg(
            Meeting_hours=("Meeting_hours", "mean"),
            After_Hours_Ratio=("After_Hours_Ratio", "mean"),
            Employee_Experience_Score=("Employee_Experience_Score", "mean"),
            Low_Value_Meeting_Ratio=("Low_Value_Meeting_Ratio", "mean"),
        )
        .sort_values("Week_Number")
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(weekly, x="Week_Number", y="Meeting_hours", markers=True,
                       title="Average Weekly Meeting Hours Over Time",
                       color_discrete_sequence=["royalblue"])
        fig.update_layout(**white_layout)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(weekly, x="Week_Number", y="After_Hours_Ratio", markers=True,
                       title="Average After-Hours Ratio Over Time",
                       color_discrete_sequence=["firebrick"])
        fig.update_layout(**white_layout, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.line(weekly, x="Week_Number", y="Employee_Experience_Score", markers=True,
                       title="Average Employee Experience Score Over Time",
                       color_discrete_sequence=["seagreen"])
        fig.update_layout(**white_layout)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.line(weekly, x="Week_Number", y="Low_Value_Meeting_Ratio", markers=True,
                       title="Average Low-Value Meeting Ratio Over Time",
                       color_discrete_sequence=["darkorange"])
        fig.update_layout(**white_layout, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Distributions")
    col5, col6, col7 = st.columns(3)
    with col5:
        fig = px.histogram(df, x="Meeting_hours", nbins=30, title="Distribution of Meeting Hours",
                            color_discrete_sequence=["royalblue"])
        fig.update_layout(**white_layout)
        st.plotly_chart(fig, use_container_width=True)
    with col6:
        fig = px.histogram(df, x="Employee_Experience_Score", nbins=20,
                            title="Distribution of Employee Experience Score",
                            color_discrete_sequence=["purple"])
        fig.update_layout(**white_layout)
        st.plotly_chart(fig, use_container_width=True)
    with col7:
        fig = px.box(df, x="Multitasking_Rate", title="Spread of Multitasking Rate",
                      color_discrete_sequence=["orange"])
        fig.update_layout(**white_layout)
        st.plotly_chart(fig, use_container_width=True)

# ---------------- DEPARTMENT COMPARISON TAB ----------------
with tab_dept:
    st.markdown("### Where Is the Meeting Burden Concentrated?")

    col1, col2 = st.columns(2)
    with col1:
        dept_avg = df.groupby("Department", as_index=False)["Meeting_hours"].mean().sort_values("Meeting_hours", ascending=False)
        fig = px.bar(dept_avg, x="Department", y="Meeting_hours",
                      title="Average Meeting Hours by Department",
                      color_discrete_sequence=["teal"])
        fig.update_layout(**white_layout)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, x="Department", y="Employee_Experience_Score",
                      title="Employee Experience Score by Department",
                      color_discrete_sequence=["mediumseagreen"])
        fig.update_layout(**white_layout)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        level_order = {"Junior Ic": 1, "Senior Ic": 2, "Senior Manager": 3, "Executive": 4}
        lvl_avg = (
            df.dropna(subset=["LevelDesignation"])
            .groupby("LevelDesignation", as_index=False)["Meeting_hours"].mean()
        )
        lvl_avg["order"] = lvl_avg["LevelDesignation"].map(level_order)
        lvl_avg = lvl_avg.sort_values("order")
        fig = px.bar(lvl_avg, x="LevelDesignation", y="Meeting_hours",
                      title="Meeting Hours by Seniority Level",
                      color_discrete_sequence=["coral"])
        fig.update_layout(**white_layout)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        role_avg = df.groupby("SupervisorIndicator", as_index=False)["Overload_Flag"].mean()
        fig = px.bar(role_avg, x="SupervisorIndicator", y="Overload_Flag",
                      title="Overload Rate: Managers vs. Individual Contributors",
                      color_discrete_sequence=["indianred"])
        fig.update_layout(**white_layout, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- DRILL-DOWN TAB ----------------
with tab_drill:
    st.markdown("### Connecting Load to Value")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdBu", zmin=-1, zmax=1,
            text=corr.round(2).values, texttemplate="%{text}",
        ))
        fig.update_layout(title="Correlation Matrix: Load vs. Experience", **white_layout)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(df, x="Meeting_hours", y="Employee_Experience_Score",
                          color="Overload_Flag", opacity=0.5,
                          title="Meeting Load vs. Experience Score",
                          color_discrete_map={True: "crimson", False: "steelblue"})
        fig.update_layout(**white_layout)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        hlv = df[df["High_Load_Low_Value"] == True]
        if not hlv.empty:
            hlv_counts = hlv["Department"].value_counts().reset_index()
            hlv_counts.columns = ["Department", "Count"]
            fig = px.bar(hlv_counts, x="Department", y="Count",
                          title='Volume of "High-Load, Low-Value" Weeks by Department',
                          color_discrete_sequence=["darkred"])
            fig.update_layout(**white_layout)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No High-Load, Low-Value rows in the current filter selection.")
    with col4:
        fig = px.box(df, x="Department", y="After_Hours_Ratio", color="SupervisorIndicator",
                      title="After-Hours Ratio by Department and Role")
        fig.update_layout(**white_layout, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- DATA TAB ----------------
with tab_data:
    st.markdown("### Filtered Data")
    st.caption(f"{len(df):,} rows × {df.shape[1]} columns after filters.")
    display_cols = [
        "PersonId", "MetricDate", "Department", "Level", "LevelDesignation", "FunctionType",
        "SupervisorIndicator", "Meeting_hours", "Collaboration_hours",
        "Low_Value_Meeting_Ratio", "After_Hours_Ratio", "Multitasking_Rate",
        "Employee_Experience_Score", "Overload_Flag", "High_Load_Low_Value",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True, height=500)

    csv = df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data as CSV", csv, "filtered_pq_data.csv", "text/csv")

st.markdown("---")
st.caption(
    "Data Analysis Capstone Project — Corporate Meeting Load & Collaboration Effectiveness Analytics. "
    "Built directly from the engineered KPIs in the Module 4 (Data Cleaning) and Module 5 (EDA) notebooks — "
    "Meeting_hours, Collaboration_hours, Low_Value_Meeting_Ratio, After_Hours_Ratio, Multitasking_Rate, "
    "Employee_Experience_Score, Overload_Flag, High_Load_Low_Value — all computed live from pq_data_cleaned.csv."
)
