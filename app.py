import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
from shapely import wkb

# Set page configuration
st.set_page_config(
    page_title="Indonesian Schools & Population Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🎓 Indonesian Schools & Population Data Dashboard")
st.markdown("""
## Understanding Indonesia's Educational Landscape

This interactive dashboard explores the relationship between **educational infrastructure** and **population distribution** across Indonesia's diverse archipelago. By analyzing comprehensive datasets of schools and population density, we can uncover patterns that inform education policy and resource allocation decisions.

### 📊 **What You'll Discover:**
- **Geographic disparities** in school distribution and population density
- **Quality indicators** through student-teacher ratios and accreditation patterns  
- **Access patterns** comparing public vs private education availability
- **Anomalies** that may indicate areas needing policy attention

### 🗺️ **Data Coverage:**
- **Schools Dataset**: 500,000+ schools with location, enrollment, staffing, and accreditation data
- **Population Dataset**: High-resolution population density grids across Indonesia's 34 provinces
- **Geographic Scope**: From dense urban centers like Jakarta to remote regions in Papua

### 📚 **Data Source:**
**Schools and population data provided by [Datawan Labs](https://github.com/datawan-labs/schools)**  
*A comprehensive collection of Indonesian educational and demographic datasets for research and analysis.*

### 🎯 **How to Use This Dashboard:**
1. **Start with filters** → Use the sidebar to select provinces and school types of interest
2. **Explore tabs systematically** → Each tab reveals different aspects of the data
3. **Look for patterns** → Notice relationships between population density and school distribution
4. **Investigate anomalies** → Use the Anomalies tab to find schools needing attention
5. **Compare regions** → Switch between provinces to understand regional differences

*💡 **Pro tip**: Start with "Select All Provinces" for national overview, then focus on specific regions for detailed analysis.*
""")

st.divider()

# Function to extract coordinates from WKB data
def extract_coordinates_from_wkb(wkb_bytes):
    try:
        # Handle null or empty values
        if wkb_bytes is None or pd.isna(wkb_bytes):
            return None, None
        
        # Convert WKB to geometry object
        geom = wkb.loads(wkb_bytes)
        
        # Extract x (longitude) and y (latitude)
        lon, lat = geom.x, geom.y
        
        # Validate coordinate bounds (Indonesia roughly: 95°E to 141°E, 6°S to 6°N)
        if lon < -180 or lon > 180 or lat < -90 or lat > 90:
            return None, None
            
        # Basic sanity check for Indonesia region
        if lon < 90 or lon > 145 or lat < -15 or lat > 10:
            # Still return coordinates but they might be questionable
            pass
            
        return lon, lat
    except Exception as e:
        # Return None for any parsing errors
        return None, None

# Load data
@st.cache_data
def load_data():
    # Load popgrid data
    popgrid_path = 'data/popgrid.parquet'
    popgrid_df = pq.read_table(popgrid_path).to_pandas()
    
    # Load sekolah data
    sekolah_path = 'data/sekolah.parquet'
    sekolah_df = pq.read_table(sekolah_path).to_pandas()
    
    # Calculate additional metrics
    sekolah_df['student_teacher_ratio'] = sekolah_df['pd'] / sekolah_df['ptk'].replace(0, np.nan)
    
    # Extract coordinates from location data
    sekolah_df['longitude'], sekolah_df['latitude'] = zip(*sekolah_df['location'].apply(extract_coordinates_from_wkb))
    
    # Extract coordinates from popgrid data
    popgrid_df['longitude'], popgrid_df['latitude'] = zip(*popgrid_df['location'].apply(extract_coordinates_from_wkb))
    
    return popgrid_df, sekolah_df

popgrid_df, sekolah_df = load_data()

# Sidebar filters
st.sidebar.header("Filters")

# Province selection (for both datasets)
# Remove null values before combining provinces
popgrid_provinces = [p for p in popgrid_df['provinsi'].unique() if pd.notna(p)]
sekolah_provinces = [p for p in sekolah_df['provinsi'].unique() if pd.notna(p)]
provinces = sorted(list(set(popgrid_provinces).union(set(sekolah_provinces))))

# Add "All Provinces" option
st.sidebar.subheader("📍 Province Filter")
select_all_provinces = st.sidebar.checkbox("Select All Provinces", value=False)

# Better default: start with key provinces for better performance
key_provinces = ['DKI Jakarta', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur', 'Sumatera Utara']
default_provinces = [p for p in key_provinces if p in provinces]
if not default_provinces:  # Fallback if key provinces not available
    default_provinces = provinces[:5]

# Handle "All Provinces" checkbox
if select_all_provinces:
    selected_provinces = provinces
else:
    selected_provinces = st.sidebar.multiselect(
        "Select Specific Provinces", 
        provinces, 
        default=default_provinces,
        help="Choose specific provinces to analyze. Uncheck 'Select All Provinces' above to use this filter."
    )

# School filters
st.sidebar.subheader("🏫 School Filters")

# Status filter for schools
school_status = ['All'] + ['negeri', 'swasta']
selected_status = st.sidebar.multiselect(
    "School Status", 
    school_status, 
    default=['All'],
    help="Filter by public (negeri) or private (swasta) schools"
)

# Location status filter
location_status_options = ['All'] + sekolah_df['location_status'].dropna().unique().tolist()
selected_location_status = st.sidebar.multiselect(
    "Location Verification Status", 
    location_status_options, 
    default=['All'],
    help="Filter by geographic coordinate verification status"
)

# Display filter summary
st.sidebar.info(f"📊 **Filter Summary:**\n"
               f"• Provinces: {len(selected_provinces)} selected\n"
               f"• School Status: {', '.join(selected_status)}\n"
               f"• Location Status: {', '.join(selected_location_status)}")

# Data source attribution in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
**📚 Data Source:**  
[Datawan Labs](https://github.com/datawan-labs/schools)

*Indonesian Schools & Population Datasets*
""")

# Apply province filter with proper empty selection handling
if selected_provinces:
    filtered_popgrid = popgrid_df[popgrid_df['provinsi'].isin(selected_provinces)]
    filtered_sekolah = sekolah_df[sekolah_df['provinsi'].isin(selected_provinces)]
else:
    # Handle case where no provinces selected - show empty datasets
    filtered_popgrid = popgrid_df.head(0)
    filtered_sekolah = sekolah_df.head(0)

# Apply additional filters with proper handling
if selected_status and 'All' not in selected_status:
    filtered_sekolah = filtered_sekolah[filtered_sekolah['status'].isin(selected_status)]

if selected_location_status and 'All' not in selected_location_status:
    filtered_sekolah = filtered_sekolah[filtered_sekolah['location_status'].isin(selected_location_status)]

# Main dashboard
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "🌍 Population", "🏫 Schools", "🗺️ Maps", "📈 Correlations", "🔍 Anomalies"])

with tab1:
    st.header("Data Overview")
    
    st.markdown("""
    ### 📈 Key Insights from Your Selection
    
    This overview provides essential metrics for understanding educational infrastructure in your selected regions. 
    The data reveals the scale of Indonesia's education system and helps identify patterns in resource distribution.
    
    **Understanding the metrics:**
    - **Population Grids**: 1km×1km cells showing population density patterns
    - **Schools**: Educational institutions from elementary to vocational levels
    - **Average Enrollment**: Students per school, indicating institution size
    - **Population Density**: People per grid cell, showing settlement patterns
    """)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Population Grids", value=f"{len(filtered_popgrid):,}")
        
    with col2:
        st.metric(label="Schools", value=f"{len(filtered_sekolah):,}")
        
    with col3:
        avg_enrollment = filtered_sekolah['pd'].mean() if len(filtered_sekolah) > 0 else 0
        st.metric(label="Avg. Enrollment", value=f"{avg_enrollment:.1f}")
        
    with col4:
        avg_pop_density = filtered_popgrid['value'].mean() if len(filtered_popgrid) > 0 else 0
        st.metric(label="Avg. Pop Density", value=f"{avg_pop_density:.1f}")
    
    # Narrative explanation of metrics
    st.markdown("""
    ### 💡 **What These Numbers Tell Us**
    
    The metrics above reveal key characteristics of your selected regions:
    
    - **High enrollment averages** (>150 students/school) suggest urban areas with larger institutions
    - **Low enrollment averages** (<50 students/school) may indicate rural areas or specialized schools
    - **Population density patterns** help explain school distribution and resource needs
    - **Grid cell counts** show the geographic coverage of our analysis
    
    Use the distribution charts below to understand how these values are spread across your selection.
    """)
    
    # Interactive distribution plots
    st.subheader("Distribution of Key Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        # Population density distribution
        if len(filtered_popgrid) > 0:
            fig_pop = px.histogram(filtered_popgrid, x='value', nbins=50, title='Distribution of Population Density')
            fig_pop.update_layout(xaxis_title='Population Density', yaxis_title='Frequency')
            st.plotly_chart(fig_pop, use_container_width=True)
            
            # Add interpretation
            median_pop = filtered_popgrid['value'].median()
            st.caption(f"📊 **Interpretation**: Most areas have relatively low population density (median: {median_pop:.1f} people/km²), with few high-density urban centers creating the long tail in this distribution.")
        else:
            st.warning("No population data available for selected filters.")
        
    with col2:
        # School enrollment distribution
        if len(filtered_sekolah) > 0:
            enrollment_data = filtered_sekolah[filtered_sekolah['pd'] < 500]
            if len(enrollment_data) > 0:
                fig_enroll = px.histogram(
                    enrollment_data, 
                    x='pd', 
                    nbins=50, 
                    title='Distribution of School Enrollment (Limited to <500)'
                )
                fig_enroll.update_layout(xaxis_title='Number of Students', yaxis_title='Frequency')
                st.plotly_chart(fig_enroll, use_container_width=True)
                
                # Add interpretation
                median_enrollment = enrollment_data['pd'].median()
                small_schools = len(enrollment_data[enrollment_data['pd'] < 50])
                total_schools = len(enrollment_data)
                st.caption(f"📊 **Interpretation**: Median enrollment is {median_enrollment:.0f} students. {small_schools:,} of {total_schools:,} schools ({small_schools/total_schools*100:.1f}%) have fewer than 50 students, indicating many small rural schools or specialized institutions.")
            else:
                st.info("No schools with enrollment < 500 in selected filters.")
        else:
            st.warning("No school data available for selected filters.")

with tab2:
    st.header("Population Analysis")
    
    st.markdown("""
    ### 🌍 Understanding Population Distribution Patterns
    
    Indonesia's population is highly concentrated in urban centers, particularly on Java island, while vast areas remain sparsely populated. 
    This analysis helps identify where people live and how density varies across regions.
    
    **Key patterns to look for:**
    - **Java dominance**: Highest population densities typically in Java provinces
    - **Urban concentration**: Cities show extreme density spikes in grid cells
    - **Rural-urban divide**: Sharp contrasts between populated and empty areas
    - **Island differences**: Outer islands generally have lower, more scattered populations
    """)
    
    if len(filtered_popgrid) > 0:
        # Population by province - interactive bar chart
        st.subheader("Average Population Density by Province")
        pop_by_prov = filtered_popgrid.groupby('provinsi')['value'].mean().sort_values(ascending=False).reset_index()
        
        fig_pop_prov = px.bar(
            pop_by_prov, 
            x='value', 
            y='provinsi', 
            orientation='h',
            title='Average Population Density by Province',
            labels={'value': 'Average Population Density', 'provinsi': 'Province'}
        )
        st.plotly_chart(fig_pop_prov, use_container_width=True)
        
        # Add interpretation
        highest_density = pop_by_prov.iloc[0]
        lowest_density = pop_by_prov.iloc[-1]
        st.markdown(f"""
        **📊 Key Insights:**
        - **Highest density**: {highest_density['provinsi']} ({highest_density['value']:.1f} people/km²)
        - **Lowest density**: {lowest_density['provinsi']} ({lowest_density['value']:.1f} people/km²)
        - **Ratio difference**: {highest_density['value']/lowest_density['value']:.1f}x variation between provinces
        
        This stark difference reflects Indonesia's geographic diversity and development patterns.
        """)
        
        # Top populated grid cells
        st.subheader("Top 20 Most Populated Grid Cells")
        top_populated = filtered_popgrid.nlargest(20, 'value')[['provinsi', 'kabupaten', 'value']]
        st.dataframe(top_populated)
        
        st.markdown("""
        **💡 Understanding Population Hotspots:**
        These grid cells represent the most densely populated 1km² areas in your selection. 
        Urban centers, city cores, and dense residential areas typically appear here. 
        Compare these locations with school distribution to assess educational access in high-density areas.
        """)
    else:
        st.warning("No population data available for selected filters.")

with tab3:
    st.header("School Analysis")
    
    st.markdown("""
    ### 🏫 Educational Infrastructure Overview
    
    Indonesia's education system includes both public (negeri) and private (swasta) institutions across multiple levels. 
    Understanding this distribution helps identify access patterns and resource allocation.
    
    **Key factors to consider:**
    - **Public vs Private balance**: Shows government vs private sector role in education
    - **Location verification**: Quality of geographic data affects spatial analysis accuracy
    - **School levels**: Different educational stages serve different community needs
    - **Enrollment patterns**: Reveal institution sizes and capacity utilization
    """)
    
    if len(filtered_sekolah) > 0:
        # Interactive charts for school count by status and location validation
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Schools by Status")
            status_counts = filtered_sekolah['status'].value_counts().reset_index()
            status_counts.columns = ['status', 'count']
            fig_status = px.pie(status_counts, values='count', names='status', title='Distribution of Public vs Private Schools')
            st.plotly_chart(fig_status, use_container_width=True)
            
            # Add interpretation
            private_pct = (status_counts[status_counts['status'] == 'swasta']['count'].sum() / status_counts['count'].sum() * 100) if len(status_counts) > 0 else 0
            st.caption(f"📊 **Insight**: {private_pct:.1f}% of schools are private. Higher private school percentages often indicate urban areas or regions with strong economic development.")
        
        with col2:
            st.subheader("Location Status Distribution")
            location_counts = filtered_sekolah['location_status'].value_counts().reset_index()
            location_counts.columns = ['location_status', 'count']
            fig_location = px.pie(location_counts, values='count', names='location_status', title='School Location Verification Status')
            st.plotly_chart(fig_location, use_container_width=True)
    else:
        st.warning("No school data available for selected filters.")
    
    # Interactive pivot table and visualization
    st.subheader("Schools by Level and Status")
    if len(filtered_sekolah) > 0:
        level_status_pivot = pd.crosstab(filtered_sekolah['jenjang'], filtered_sekolah['status']).head(10)
        st.dataframe(level_status_pivot)
        
        # Interactive enrollment by school level
        st.subheader("Average Enrollment by School Level")
        enrollment_by_level = filtered_sekolah.groupby('jenjang')['pd'].mean().sort_values(ascending=False).reset_index()
        
        fig_enroll_level = px.bar(
            enrollment_by_level, 
            x='pd', 
            y='jenjang', 
            orientation='h',
            title='Average Enrollment by School Level',
            labels={'pd': 'Average Enrollment', 'jenjang': 'School Level'}
        )
        st.plotly_chart(fig_enroll_level, use_container_width=True)
    else:
        st.info("No school data available to display level analysis.")

with tab4:
    st.header("Geographic Visualization")
    
    st.markdown("""
    ### 🗺️ Spatial Distribution Patterns
    
    Geographic visualization reveals how schools and population are distributed across Indonesia's landscape. 
    These maps help identify spatial patterns, accessibility issues, and planning opportunities.
    
    **What to observe:**
    - **Clustering patterns**: Schools often cluster in populated areas
    - **Accessibility gaps**: Remote areas with population but few schools
    - **Urban concentration**: Dense school networks in cities
    - **Geographic barriers**: Islands, mountains affecting distribution
    
    *Note: Maps show representative samples for performance. Use province filters to focus on specific regions.*
    """)
    
    # Prepare data for maps using filtered datasets
    # Filter out rows with null coordinates
    sekolah_with_coords = filtered_sekolah.dropna(subset=['latitude', 'longitude'])
    popgrid_with_coords = filtered_popgrid.dropna(subset=['latitude', 'longitude'])
    
    # Create data for Streamlit map
    if not sekolah_with_coords.empty:
        st.subheader("School Locations")
        
        # Better sampling strategy with random state for reproducibility
        sample_size = min(len(sekolah_with_coords), 1000)
        if sample_size < len(sekolah_with_coords):
            school_sample = sekolah_with_coords.sample(sample_size, random_state=42)
        else:
            school_sample = sekolah_with_coords
            
        school_map_data = pd.DataFrame({
            'lat': school_sample['latitude'],
            'lon': school_sample['longitude']
        })
        
        st.map(school_map_data, zoom=5)
        st.caption(f"Showing {len(school_sample)} of {len(sekolah_with_coords)} schools")
        
        # Map by location status
        st.subheader("School Locations by Verification Status")
        location_status_map_options = ['All'] + filtered_sekolah['location_status'].dropna().unique().tolist()
        location_status = st.selectbox("Select Location Status for Map", options=location_status_map_options)
        
        if location_status != 'All':
            status_schools = sekolah_with_coords[sekolah_with_coords['location_status'] == location_status]
            if not status_schools.empty:
                status_sample_size = min(1000, len(status_schools))
                if status_sample_size < len(status_schools):
                    status_sample = status_schools.sample(status_sample_size, random_state=42)
                else:
                    status_sample = status_schools
                    
                status_map_data = pd.DataFrame({
                    'lat': status_sample['latitude'],
                    'lon': status_sample['longitude']
                })
                st.map(status_map_data, zoom=5, use_container_width=True)
                st.caption(f"Showing {len(status_sample)} of {len(status_schools)} schools with {location_status} status")
            else:
                st.warning(f"No schools found with {location_status} status in selected filters.")
    else:
        st.warning("No school coordinate data available for selected filters.")
    
    # Population density map
    if not popgrid_with_coords.empty:
        st.subheader("Population Grid Locations (Sampled)")
        
        # Better sampling strategy for population data
        pop_sample_size = min(len(popgrid_with_coords), 2000)
        if pop_sample_size < len(popgrid_with_coords):
            pop_sample = popgrid_with_coords.sample(pop_sample_size, random_state=42)
        else:
            pop_sample = popgrid_with_coords
        
        # Create a dataframe with coordinates and population values (for color coding)
        pop_map_data = pd.DataFrame({
            'lat': pop_sample['latitude'],
            'lon': pop_sample['longitude'],
            'value': pop_sample['value']
        })
        
        # Create a sample map with all points
        st.map(pop_map_data, zoom=5)
        st.caption(f"Showing {len(pop_sample)} of {len(popgrid_with_coords)} population grid cells")
        
        # More detailed view with school locations by province
        if len(selected_provinces) > 0:
            st.subheader("School Distribution by Province")
            selected_prov_map = st.selectbox("Select Province for School Map", options=selected_provinces)
            
            if selected_prov_map:
                prov_schools = sekolah_with_coords[sekolah_with_coords['provinsi'] == selected_prov_map]
                if not prov_schools.empty:
                    prov_sample_size = min(len(prov_schools), 500)
                    if prov_sample_size < len(prov_schools):
                        prov_sample = prov_schools.sample(prov_sample_size, random_state=42)
                    else:
                        prov_sample = prov_schools
                        
                    prov_map_data = pd.DataFrame({
                        'lat': prov_sample['latitude'],
                        'lon': prov_sample['longitude']
                    })
                    st.map(prov_map_data, zoom=7)
                    st.caption(f"Showing {len(prov_sample)} of {len(prov_schools)} schools in {selected_prov_map}")
                else:
                    st.warning(f"No schools with coordinates found in {selected_prov_map}")
    else:
        st.warning("No population grid data available for selected filters.")

with tab5:
    st.header("Correlations and Relationships")
    
    st.markdown("""
    ### 📈 Understanding Data Relationships
    
    Correlation analysis reveals how different variables interact and influence each other. 
    These relationships help explain patterns and guide policy decisions.
    
    **Key relationships to explore:**
    - **Population vs Schools**: Do densely populated areas have more schools?
    - **Enrollment vs Teacher Ratios**: How do class sizes vary by region and level?
    - **Gender Patterns**: Are there systematic differences in male/female participation?
    - **Geographic Effects**: How does location influence educational characteristics?
    
    Strong correlations (>0.7) suggest predictable relationships, while weak correlations (<0.3) indicate complex interactions.
    """)
    
    # Interactive correlation plot
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("School Enrollment vs. Population Density by Province")
        
        # Calculate averages by province using filtered data
        avg_enrollment_by_prov = filtered_sekolah.groupby('provinsi')['pd'].mean()
        avg_pop_by_prov = filtered_popgrid.groupby('provinsi')['value'].mean()
        
        # Merge the two series ensuring same index
        combined_prov_stats = pd.DataFrame({
            'avg_enrollment': avg_enrollment_by_prov,
            'avg_pop_density': avg_pop_by_prov
        }).dropna()
        
        # Reset index to have province as a column
        combined_prov_stats = combined_prov_stats.reset_index()
        
        if len(combined_prov_stats) > 1:
            correlation = combined_prov_stats['avg_enrollment'].corr(combined_prov_stats['avg_pop_density'])
            
            fig_corr = px.scatter(
                combined_prov_stats, 
                x='avg_pop_density', 
                y='avg_enrollment',
                hover_name='provinsi',
                title=f'School Enrollment vs. Population Density by Province\nCorrelation: {correlation:.3f}'
            )
            fig_corr.update_layout(xaxis_title='Average Population Density', yaxis_title='Average School Enrollment')
            
            # Add trend line
            fig_corr.add_traces(
                px.scatter(
                    combined_prov_stats, 
                    x='avg_pop_density', 
                    y='avg_enrollment', 
                    trendline="ols"
                ).data
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        st.subheader("Student-Teacher Ratio by School Level")
        ratio_by_level = sekolah_df.groupby('jenjang')['student_teacher_ratio'].mean().sort_values(ascending=False).reset_index()
        
        fig_ratio = px.bar(
            ratio_by_level, 
            x='student_teacher_ratio', 
            y='jenjang', 
            orientation='h',
            title='Average Student-Teacher Ratio by School Level',
            labels={'student_teacher_ratio': 'Average Student-Teacher Ratio', 'jenjang': 'School Level'}
        )
        st.plotly_chart(fig_ratio, use_container_width=True)
    
    # Interactive gender distribution analysis
    st.subheader("Gender Distribution in Education")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Teachers**")
        total_male_teachers = sekolah_df['ptk_laki'].sum()
        total_female_teachers = sekolah_df['ptk_perempuan'].sum()
        
        fig_teachers = px.pie(
            names=['Male', 'Female'], 
            values=[total_male_teachers, total_female_teachers], 
            title='Gender Distribution of Teachers'
        )
        st.plotly_chart(fig_teachers, use_container_width=True)
    
    with col2:
        st.write("**Students**")
        total_male_students = sekolah_df['pd_laki'].sum()
        total_female_students = sekolah_df['pd_perempuan'].sum()
        
        fig_students = px.pie(
            names=['Male', 'Female'], 
            values=[total_male_students, total_female_students], 
            title='Gender Distribution of Students'
        )
        st.plotly_chart(fig_students, use_container_width=True)
    
    # Interactive accreditation analysis
    st.subheader("Accreditation Distribution by School Level")
    accred_by_level = pd.crosstab(sekolah_df['jenjang'], sekolah_df['akreditasi']).head(10)
    
    # Convert to long format for Plotly
    accred_long = accred_by_level.reset_index().melt(id_vars='jenjang', var_name='akreditasi', value_name='count')
    
    fig_accred = px.bar(
        accred_long, 
        x='count', 
        y='jenjang', 
        color='akreditasi',
        orientation='h',
        title='Accreditation Distribution by School Level',
        labels={'count': 'Number of Schools', 'jenjang': 'School Level', 'akreditasi': 'Accreditation'}
    )
    st.plotly_chart(fig_accred, use_container_width=True)

with tab6:
    st.header("Anomalies and Special Cases")
    
    st.markdown("""
    ### 🔍 Identifying Data Quality Issues and Outliers
    
    Anomaly detection helps identify data quality issues, unusual patterns, and schools that may need special attention. 
    These outliers often reveal important policy challenges or data collection problems.
    
    **Types of anomalies to investigate:**
    - **Staffing mismatches**: Schools with students but no teachers (or vice versa)
    - **Extreme ratios**: Unusually high or low student-teacher ratios
    - **Geographic inconsistencies**: Location data that doesn't match administrative boundaries
    - **Enrollment outliers**: Schools with unexpectedly high or low enrollment
    
    **Policy implications:**
    - Resource allocation needs
    - Data quality improvement priorities  
    - Special intervention requirements
    - Infrastructure planning opportunities
    """)
    
    # Location validation issues
    st.subheader("Location Validation Issues")
    location_issues = sekolah_df[sekolah_df['location_status'].isin(['potential-mismatch', 'invalid', 'not-mapped'])]
    st.write(f"Count: {len(location_issues)} schools with location validation issues")
    if len(location_issues) > 0:
        st.dataframe(location_issues[['nama', 'provinsi', 'kabupaten', 'location_status']].head(10))
    
    # Schools with no students but with teachers
    st.subheader("Schools with Teachers but No Students")
    no_stud_with_teach = sekolah_df[(sekolah_df['pd'] == 0) & (sekolah_df['ptk'] > 0)]
    st.write(f"Count: {len(no_stud_with_teach)} schools")
    if len(no_stud_with_teach) > 0:
        st.dataframe(no_stud_with_teach[['nama', 'jenjang', 'provinsi', 'pd', 'ptk']].head(10))
    
    # Schools with students but no teachers
    st.subheader("Schools with Students but No Teachers")
    stud_no_teach = sekolah_df[(sekolah_df['pd'] > 0) & (sekolah_df['ptk'] == 0)]
    st.write(f"Count: {len(stud_no_teach)} schools")
    if len(stud_no_teach) > 0:
        st.dataframe(stud_no_teach[['nama', 'jenjang', 'provinsi', 'pd', 'ptk']].head(10))
    
    # High student-teacher ratio schools
    st.subheader("Schools with Highest Student-Teacher Ratios")
    high_ratio_schools = sekolah_df[sekolah_df['student_teacher_ratio'] > 50].nlargest(10, 'student_teacher_ratio')
    st.dataframe(high_ratio_schools[['nama', 'jenjang', 'provinsi', 'pd', 'ptk', 'student_teacher_ratio']])
    
    # Large schools by enrollment
    st.subheader("Largest Schools by Enrollment")
    largest_schools = sekolah_df.nlargest(10, 'pd')[['nama', 'jenjang', 'provinsi', 'pd', 'status']]
    st.dataframe(largest_schools)

# Footer with proper attribution
st.markdown("---")
st.markdown("""
### 📋 **Data Attribution & Credits**

**Data Source:** [Datawan Labs - Indonesian Schools Dataset](https://github.com/datawan-labs/schools)  
**Repository:** `https://github.com/datawan-labs/schools`  
**Datasets Used:** `sekolah.parquet` (schools data) and `popgrid.parquet` (population grids)

**About the Data:**
- School location data with enrollment, staffing, and accreditation information
- High-resolution population density grids across Indonesia
- Geographic validation against 514 district/city administrative boundaries

*This dashboard is built for educational and research purposes using publicly available datasets.*

---
*Built with Streamlit • Data processed with Pandas & PyArrow • Maps powered by Plotly*
""")