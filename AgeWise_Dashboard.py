import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\Jia Yi\Downloads\AgeWise\Dataset\final_datausage_cleaned.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Load model
model = joblib.load(r"C:\Users\Jia Yi\Downloads\AgeWise\random_forest_pipeline.pkl")

# Page title
st.title("AgeWise: Malaysia Healthcare and Aging Dashboard")
st.markdown("This dashboard helps researchers and policymakers explore Malaysia’s aging trends and plan healthcare resources.")

# Sidebar: Controls
st.sidebar.header("Controls")
states = sorted(df["state"].dropna().unique())
selected_state = st.sidebar.selectbox("Select State", states)

# Include future years
existing_years = df["year"].dropna().unique()
all_years = sorted(np.unique(np.concatenate([existing_years, np.arange(2024, 2031)])))
selected_year = st.sidebar.selectbox("Select Year", all_years)

# Determine strata automatically
if selected_year <= df["year"].max():
    strata_options = sorted(
        df[(df["state"] == selected_state) & (df["year"] == selected_year)]["strata"].dropna().unique()
    )
    selected_strata = strata_options[0] if strata_options else "Urban"
else:
    selected_strata = "Urban"  # default for future years

# Show strata as static info
st.sidebar.markdown(f"**Area Type (Strata):** {selected_strata}")

if selected_year >= 2024:
    st.sidebar.warning("Manual entry required for year 2024 and beyond.")
    population_input = st.sidebar.number_input("Population", min_value=10000, value=300000)
    urban_pct = st.sidebar.slider("Urban Population (%)", 0.0, 100.0, 75.0)
    elderly_ratio = st.sidebar.slider("Elderly Ratio (%)", 0.0, 50.0, 15.0)
else:
    matched_rows = df[
        (df["state"] == selected_state) &
        (df["year"] == selected_year) &
        (df["strata"] == selected_strata)
    ]

    if matched_rows.empty:
        st.warning("No data available for the selected state, year, and strata combination.")
        st.stop()

    population_input = matched_rows["population"].sum()
    urban_pct = matched_rows["urban_population_(%_of_total_population)"].mean()
    elderly_count = matched_rows[matched_rows["is_elderly"] == True]["population"].sum()
    elderly_ratio = (elderly_count / population_input) * 100 if population_input > 0 else 0

    st.sidebar.markdown("Autofilled values from dataset:")
    st.sidebar.write(f"Population: {int(population_input):,}")
    st.sidebar.write(f"Urban Population (%): {urban_pct:.2f}")
    st.sidebar.write(f"Elderly Ratio (%): {elderly_ratio:.2f}")

# Age Group Distribution
st.subheader("1. Population by Age Group")
age_df = df[(df["state"] == selected_state) & (df["year"] == selected_year)]
if not age_df.empty:
    age_grouped = age_df.groupby("age")["population"].sum().reset_index()
    age_order = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39',
                 '40-44', '45-49', '50-54', '55-59', '60-64', '65-69',
                 '70-74', '75-79', '80-84', '85+']
    age_grouped["age"] = pd.Categorical(age_grouped["age"], categories=age_order, ordered=True)
    age_grouped = age_grouped.sort_values("age")

    fig_line = px.line(age_grouped, x="age", y="population", markers=True,
                       title=f"Age Group Distribution in {selected_state} ({selected_year})")
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.info("Age group data not available for this year.")

# Elderly Ratio by State
st.subheader("2. Elderly Ratio by State")
elderly_df = df[df["is_elderly"] == True]
elderly_summary = (
    elderly_df.groupby("state")["population"]
    .sum()
    .reset_index()
    .rename(columns={"population": "elderly_population"})
)
total_pop = df.groupby("state")["population"].sum().reset_index().rename(columns={"population": "total_population"})
aging_ratio = pd.merge(elderly_summary, total_pop, on="state")
aging_ratio["elderly_ratio"] = (aging_ratio["elderly_population"] / aging_ratio["total_population"]) * 100

fig_bar = px.bar(aging_ratio.sort_values("elderly_ratio", ascending=False),
                 x="elderly_ratio", y="state", orientation="h",
                 labels={"elderly_ratio": "Elderly Population (%)", "state": "State"},
                 title="Elderly Population Ratio by State")
st.plotly_chart(fig_bar, use_container_width=True)

# Urban vs Rural Resource Comparison
st.subheader("3. Urban vs Rural Resource Comparison")
comparison_df = df.groupby("strata")[["beds_per_1000", "staff_per_1000"]].mean().reset_index()
fig_compare = px.bar(comparison_df, x="strata",
                     y=["beds_per_1000", "staff_per_1000"],
                     barmode="group",
                     title="Beds and Staff per 1000 People by Area Type")
st.plotly_chart(fig_compare, use_container_width=True)

# Population vs Staff Count
st.subheader("4. Population vs Staff Count")
sample_df = df.dropna(subset=["population", "staff_count"])
fig_scatter = px.scatter(sample_df, x="population", y="staff_count", color="strata",
                         title="Population vs Healthcare Staff Count",
                         labels={"population": "Population", "staff_count": "Staff Count"})
st.plotly_chart(fig_scatter, use_container_width=True)

# Hospital Bed Prediction
st.subheader("5. Predict Hospital Bed Needs")
st.markdown("Prediction uses real values from data (2023 and before) or manual inputs (2024+).")

# Construct input for model
input_df = pd.DataFrame({
    "year": [selected_year],
    "state": [selected_state],
    "strata": [selected_strata],
    "population": [population_input],
    "Urban population (% of total population)": [urban_pct],
    "elderly_ratio": [elderly_ratio]
})

if input_df.isnull().values.any():
    st.error("Prediction aborted: Some required features are missing (NaN).")
    st.write("Model Input Preview:", input_df)
    st.stop()

# Prediction
prediction = model.predict(input_df)[0]
st.success(f"Predicted Hospital Beds Required: {int(prediction):,}")
