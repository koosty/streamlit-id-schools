# Indonesian Schools & Population Data Dashboard

This Streamlit dashboard visualizes insights from Indonesian schools and population data, providing an interactive interface to explore the datasets.

## Features

The dashboard includes:

1. **Overview Tab**: Shows key metrics and distributions of the data
   - Population grids and school counts
   - Average enrollment and population density
   - Distribution plots for key metrics

2. **Population Tab**: Analyzes population density patterns
   - Population density by province
   - Top populated grid cells
   - Geographic distribution insights

3. **Schools Tab**: Examines educational institutions
   - School distribution by status (public/private)
   - School levels and counts
   - Average enrollment by school level

4. **Correlations Tab**: Explores relationships between variables
   - School enrollment vs. population density
   - Student-teacher ratios by school level
   - Gender distribution in education

5. **Anomalies Tab**: Highlights unusual patterns
   - Schools with no students but teachers
   - Schools with students but no teachers
   - Schools with extremely high ratios
   - Largest schools by enrollment

## How to Use

1. **Filters**: Use the sidebar to select specific provinces for focused analysis
2. **Tabs**: Navigate between different analysis perspectives
3. **Visualizations**: Hover over charts for additional details
4. **Data Tables**: Scroll through detailed data tables for specific information

## Technical Implementation

The dashboard uses:
- Streamlit for the web interface
- Pandas for data manipulation
- PyArrow for reading parquet files
- Matplotlib for visualizations
- Caching to improve performance

## Running the Dashboard

To run the dashboard locally:

```bash
pip install streamlit pandas pyarrow matplotlib seaborn
streamlit run dashboard_app.py
```

The dashboard will be available at `http://localhost:8501`

## Data Sources

- `data/popgrid.parquet`: Contains population density data with province, regency, and location information
- `data/sekolah.parquet`: Contains school information including enrollment, staff, accreditation, and location data

## Key Insights

The dashboard reveals several important patterns:
- High concentration of private schools in dense urban areas
- Significant variation in student-teacher ratios across school types
- Gender imbalances in teaching staff
- Geographic disparities in education infrastructure
- Population density correlations with school enrollment