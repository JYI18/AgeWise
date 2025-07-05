import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Load data and model
df = pd.read_csv(r"C:\Users\Jia Yi\Downloads\AgeWise\Dataset\final_datausage_cleaned.csv")
model = joblib.load(r"C:\Users\Jia Yi\Downloads\AgeWise\random_forest_pipeline.pkl")

# Title
st.title("AgeWise: Malaysia Healthcare and Aging Dashboard")
st.markdown("This dashboard helps researchers and policymakers explore Malaysia’s aging trends and plan healthcare resources.")

# Sidebar Controls
st.sidebar.header("Controls")
selected_state = st.sidebar.selectbox("Select State", sorted(df["state"].unique()))
selected_year = st.sidebar.selectbox("Select Year", sorted(df["year"].unique()))
urban_pct = st.sidebar.slider("Urban Population (%)", 50, 100, 80)
elderly_ratio = st.sidebar.slider("Elderly Ratio (%)", 0, 50, 15)
population_input = st.sidebar.number_input("Total Population", min_value=10000, value=300000)

# Section 1: Population by Age Group
st.subheader("5. Population by Age Group")
age_df = df[(df["state"] == selected_state) & (df["year"] == selected_year)]
age_grouped = age_df.groupby("age")["population"].sum().reset_index()

fig_line = px.line(age_grouped, x="age", y="population", markers=True,
                   title=f"Age Group Distribution in {selected_state} ({selected_year})")
st.plotly_chart(fig_line, use_container_width=True)

# Section 2: Elderly Ratio by State
st.subheader("1. Elderly Ratio by State")
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


# Section 3: Urban vs Rural Comparison
st.subheader("3. Urban vs Rural Resource Comparison")
comparison_df = df.groupby("strata")[["beds_per_1000", "staff_per_1000"]].mean().reset_index()
fig_compare = px.bar(comparison_df, x="strata",
                     y=["beds_per_1000", "staff_per_1000"],
                     barmode="group",
                     title="Beds and Staff per 1000 People by Strata")
st.plotly_chart(fig_compare, use_container_width=True)

# Section 4: Population vs Staff
st.subheader("4. Population vs Staff Count")
sample_df = df.dropna(subset=["population", "staff_count"])
fig_scatter = px.scatter(sample_df, x="population", y="staff_count", color="strata",
                         title="Population vs Healthcare Staff Count",
                         labels={"population": "Population", "staff_count": "Staff Count"})
st.plotly_chart(fig_scatter, use_container_width=True)



# Section 5: Hospital Bed Prediction Panel
st.subheader("2. Predict Hospital Bed Needs")
st.markdown("Use the inputs in the sidebar to simulate future infrastructure needs.")

input_df = pd.DataFrame({
    "year": [selected_year],
    "state": [selected_state],
    "strata": ["Urban"],
    "population": [population_input],
    "Urban population (% of total population)": [urban_pct],
    "elderly_ratio": [elderly_ratio / 100.0]
})

prediction = model.predict(input_df)[0]
st.success(f"Predicted Hospital Beds Required: {int(prediction):,}")
