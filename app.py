import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# RAG module (clean separation)
from rag.mental_health_rag import MentalHealthRAG, RAG_AVAILABLE

# Page Configuration
st.set_page_config(
    page_title="Student Mental Health Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    h1 {
        color: #2d3748;
        font-family: 'Helvetica Neue', sans-serif;
        text-align: center;
        padding: 20px;
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h2 {
        color: #4a5568;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e2e8f0;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load Dataset
@st.cache_data
def load_data():
    """Load the mental health dataset and normalize column names"""
    try:
        # Prefer cleaned file; fall back to original if needed
        try:
            df = pd.read_csv('Student_Mental_Health_CLEANED.csv')
        except FileNotFoundError:
            df = pd.read_csv('Student Mental health.csv')

        # Normalize legacy column names to the current schema
        df = df.rename(columns={
            'Choose your gender': 'Gender',
            'What is your course?': 'Course',
            'Your current year of Study': 'StudyYear',
            'Year of Study': 'StudyYear',
            'What is your CGPA?': 'CGPA',
            'Do you have Depression?': 'Depression',
            'Do you have Anxiety?': 'Anxiety',
            'Do you have Panic attack?': 'PanicAttack',
            'Panic Attack': 'PanicAttack',
            'Did you seek any specialist for a treatment?': 'Treatment'
        })
        
        # Normalize StudyYear to lowercase for consistency
        if 'StudyYear' in df.columns:
            df['StudyYear'] = df['StudyYear'].str.lower().str.strip()

    except FileNotFoundError:
        st.error("Dataset not found. Place 'Student_Mental_Health_CLEANED.csv' (or the original 'Student Mental health.csv') in this directory.")
        st.info("Download from: https://www.kaggle.com/datasets/aminasalamat/mental-health-of-students-dataset")
        st.stop()
    return df

# Main Application
def main():
    # Header
    st.markdown("<h1>🧠 Student Mental Health Analytics Dashboard</h1>", unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/mental-health.png", width=100)
        st.title("📊 Dashboard Controls")
        st.markdown("---")
        
        # Display key metrics
        st.subheader("📈 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Students", len(df))
        with col2:
            st.metric("Features", len(df.columns))
        
        if 'Depression' in df.columns:
            depression_rate = (df['Depression'] == 'Yes').sum() / len(df) * 100
            st.metric("Depression Rate", f"{depression_rate:.1f}%")
        
        if 'Anxiety' in df.columns:
            anxiety_rate = (df['Anxiety'] == 'Yes').sum() / len(df) * 100
            st.metric("Anxiety Rate", f"{anxiety_rate:.1f}%")
        
        st.markdown("---")
        st.info("💡 Navigate through tabs to explore different visualizations")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "🎓 Demographics",
        "🧠 Mental Health",
        "📈 Correlations",
        "🎯 Performance Analysis",
        "🤖 RAG Q&A"
    ])
    
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.subheader("Dataset Info")
            st.write(f"**Rows:** {df.shape[0]}")
            st.write(f"**Columns:** {df.shape[1]}")
            st.write("**Columns:**")
            for col in df.columns:
                st.write(f"- {col}")
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab2:
        st.header("Demographics Analysis")
        
        # Age distribution
        fig = px.histogram(df, x='Age', nbins=30,
                           title='Age Distribution of Students',
                           color_discrete_sequence=['#667eea'],
                           template='plotly_white')
        fig.update_traces(marker_line_color='white', marker_line_width=1.5)
        fig.update_layout(
            title_font_size=20,
            title_font_color='#2d3748',
            showlegend=False,
            xaxis_title="Age",
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Gender and Course distribution
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Gender Distribution', 'Course Distribution'),
            specs=[[{"type": "pie"}, {"type": "pie"}]]
        )
        
        gender_counts = df['Gender'].value_counts()
        fig.add_trace(
            go.Pie(labels=gender_counts.index, values=gender_counts.values,
                   marker_colors=['#667eea', '#764ba2', '#f093fb']),
            row=1, col=1
        )
        
        course_counts = df['Course'].value_counts().head(10)
        fig.add_trace(
            go.Pie(labels=course_counts.index, values=course_counts.values,
                   marker_colors=px.colors.qualitative.Set3),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Demographics Overview",
            title_font_size=20,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Year of Study distribution
        year_counts = df['StudyYear'].value_counts()
        year_order = ['year 1', 'year 2', 'year 3', 'year 4']
        year_counts = year_counts.reindex([y for y in year_order if y in year_counts.index])
        
        fig = go.Figure(data=[
            go.Bar(x=year_counts.index, y=year_counts.values,
                   marker_color='#667eea',
                   marker_line_color='white',
                   marker_line_width=1.5,
                   text=year_counts.values,
                   textposition='outside')
        ])
        
        fig.update_layout(
            title='Distribution of Students by Year of Study',
            title_font_size=20,
            xaxis_title="Year of Study",
            yaxis_title="Number of Students",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # CGPA Analysis by gender
        cgpa_stats = df.groupby('Gender')['CGPA'].agg(['mean', 'std', 'min', 'max']).reset_index()
        
        fig = go.Figure()
        colors = ['#667eea', '#764ba2', '#f093fb']
        for idx, (_, row) in enumerate(cgpa_stats.iterrows()):
            fig.add_trace(go.Bar(
                x=[row['Gender']],
                y=[row['mean']],
                error_y=dict(
                    type='data',
                    array=[row['std']],
                    visible=True
                ),
                marker_color=colors[idx % len(colors)],
                name=row['Gender'],
                text=f"Avg: {row['mean']:.2f}",
                textposition="outside"
            ))
        
        fig.update_layout(
            title='Average CGPA by Gender (with Std Dev Range)',
            title_font_size=20,
            xaxis_title="Gender",
            yaxis_title="CGPA",
            showlegend=True,
            template='plotly_white',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Mental Health Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Depression prevalence
            if 'Depression' in df.columns:
                depression_counts = df['Depression'].value_counts()
                
                fig = go.Figure(data=[
                    go.Pie(labels=depression_counts.index,
                           values=depression_counts.values,
                           hole=0.4,
                           marker_colors=['#84fab0', '#ff6b6b'],
                           textinfo='label+percent')
                ])
                
                fig.update_layout(
                    title='Depression Prevalence Among Students',
                    title_font_size=20,
                    height=500,
                    annotations=[dict(text='Depression', x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Anxiety prevalence
            if 'Anxiety' in df.columns:
                anxiety_counts = df['Anxiety'].value_counts()
                
                fig = go.Figure(data=[
                    go.Pie(labels=anxiety_counts.index,
                           values=anxiety_counts.values,
                           hole=0.4,
                           marker_colors=['#8fd3f4', '#ffa07a'],
                           textinfo='label+percent')
                ])
                
                fig.update_layout(
                    title='Anxiety Prevalence Among Students',
                    title_font_size=20,
                    height=500,
                    annotations=[dict(text='Anxiety', x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Mental health by year of study
        if 'Depression' in df.columns and 'Anxiety' in df.columns and 'StudyYear' in df.columns:
            mental_health_year = df.groupby('StudyYear')[['Depression', 'Anxiety']].apply(
                lambda x: (x == 'Yes').sum()
            ).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Depression',
                x=mental_health_year['StudyYear'],
                y=mental_health_year['Depression'],
                marker_color='#667eea'
            ))
            fig.add_trace(go.Bar(
                name='Anxiety',
                x=mental_health_year['StudyYear'],
                y=mental_health_year['Anxiety'],
                marker_color='#764ba2'
            ))
            
            fig.update_layout(
                title='Mental Health Issues by Year of Study',
                title_font_size=20,
                barmode='group',
                xaxis_title="Year of Study",
                yaxis_title="Number of Cases",
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # CGPA vs Depression
        if 'Depression' in df.columns:
            fig = px.violin(df, y='CGPA', x='Depression', color='Depression',
                            box=True, points='all',
                            title='CGPA Distribution by Depression Status',
                            color_discrete_sequence=['#84fab0', '#ff6b6b'],
                            template='plotly_white')
            fig.update_layout(title_font_size=20)
            st.plotly_chart(fig, use_container_width=True)
        
        # Treatment seeking analysis
        if 'Treatment' in df.columns:
            treatment_counts = df['Treatment'].value_counts()
            
            fig = go.Figure(data=[
                go.Bar(x=treatment_counts.index,
                       y=treatment_counts.values,
                       marker_color=['#667eea', '#764ba2'],
                       text=treatment_counts.values,
                       textposition='outside')
            ])
            
            fig.update_layout(
                title='Students Seeking Treatment',
                title_font_size=20,
                xaxis_title="Treatment Status",
                yaxis_title="Number of Students",
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Lifestyle & Stress Factors")
        col3, col4 = st.columns(2)
        
        with col3:
            # Sleep Quality
            if 'Sleep Quality' in df.columns:
                counts = df['Sleep Quality'].value_counts(dropna=False)
                fig = px.bar(
                    x=counts.index.astype(str),
                    y=counts.values,
                    title='Sleep Quality Distribution',
                    color_discrete_sequence=['#667eea'],
                    template='plotly_white'
                )
                fig.update_traces(text=counts.values, textposition='outside')
                fig.update_layout(
                    xaxis_title='Sleep Quality',
                    yaxis_title='Count'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Exercise Frequency
            if 'Exercise Frequency' in df.columns:
                counts = df['Exercise Frequency'].value_counts(dropna=False)
                fig = px.bar(
                    x=counts.index.astype(str),
                    y=counts.values,
                    title='Exercise Frequency Distribution',
                    color_discrete_sequence=['#8fd3f4'],
                    template='plotly_white'
                )
                fig.update_traces(text=counts.values, textposition='outside')
                fig.update_layout(
                    xaxis_title='Exercise Frequency',
                    yaxis_title='Count'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Social Support
            if 'Social Support' in df.columns:
                counts = df['Social Support'].value_counts(dropna=False)
                fig = px.bar(
                    x=counts.index.astype(str),
                    y=counts.values,
                    title='Social Support Levels',
                    color_discrete_sequence=['#84fab0'],
                    template='plotly_white'
                )
                fig.update_traces(text=counts.values, textposition='outside')
                fig.update_layout(
                    xaxis_title='Social Support',
                    yaxis_title='Count'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Financial Stress
            if 'Financial Stress' in df.columns:
                counts = df['Financial Stress'].value_counts(dropna=False)
                fig = px.bar(
                    x=counts.index.astype(str),
                    y=counts.values,
                    title='Financial Stress Levels',
                    color_discrete_sequence=['#ff9f43'],
                    template='plotly_white'
                )
                fig.update_traces(text=counts.values, textposition='outside')
                fig.update_layout(
                    xaxis_title='Financial Stress',
                    yaxis_title='Count'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Correlation Analysis")
        
        # Create correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(12, 8))
            correlation = df[numeric_cols].corr()
            
            sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                        square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                        fmt='.2f', ax=ax)
            plt.title('Correlation Heatmap of Mental Health Factors', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Not enough numeric columns for a correlation heatmap.")
        
        st.subheader("Key Insights")
        if len(numeric_cols) >= 2:
            correlation = df[numeric_cols].corr()
            st.write("Strongest positive correlations:")
            correlations_unstack = correlation.unstack()
            sorted_corr = correlations_unstack[correlations_unstack < 1].sort_values(ascending=False)
            st.write(sorted_corr.head(5))
    
    with tab5:
        st.header("Performance Analysis")
        st.write("Explore how academic performance relates to various lifestyle and stress factors.")
        
        # Filter Section
        if 'Academic Pressure' in df.columns:
            st.markdown("### 🎚️ Filters")
            col_filter1, col_filter2 = st.columns([3, 1])
            
            with col_filter1:
                pressure_levels = df['Academic Pressure'].dropna().unique().tolist()
                pressure_levels.sort()
                selected_pressures = st.multiselect(
                    "Select Academic Pressure Levels to Display:",
                    options=pressure_levels,
                    default=pressure_levels,
                    help="Filter the data by specific academic pressure levels"
                )
            
            with col_filter2:
                st.metric("Filtered Students", len(df[df['Academic Pressure'].isin(selected_pressures)]))
            
            # Apply filter
            if selected_pressures:
                df_filtered = df[df['Academic Pressure'].isin(selected_pressures)].copy()
                st.markdown("---")
            else:
                st.warning("⚠️ Please select at least one Academic Pressure level.")
                df_filtered = df.copy()
        else:
            df_filtered = df.copy()
        
        # Academic Stress and CGPA by Course
        st.subheader("📚 Academic Stress vs CGPA by Course")
        if 'Academic Pressure' in df_filtered.columns and 'Course' in df_filtered.columns and 'CGPA' in df_filtered.columns:
            # Convert Academic Pressure to numeric
            pressure_map = {
                "Low": 1,
                "Moderate": 2,
                "High": 3,
                "Very High": 4
            }
            df_temp = df_filtered.copy()
            df_temp["Academic_Pressure_Num"] = df_temp["Academic Pressure"].map(pressure_map)
            
            # Select top 5 courses by student count
            top_courses = df_temp["Course"].value_counts().head(5).index
            df_top = df_temp[df_temp["Course"].isin(top_courses)]
            
            # Group data
            grouped = df_top.groupby("Course", as_index=False).agg({
                "Academic_Pressure_Num": "mean",
                "CGPA": "mean"
            })
            
            # Create grouped bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Academic Pressure',
                x=grouped['Course'],
                y=grouped['Academic_Pressure_Num'],
                marker_color='#667eea',
                text=[f"{val:.2f}" for val in grouped['Academic_Pressure_Num']],
                textposition='outside'
            ))
            
            fig.add_trace(go.Bar(
                name='Average CGPA',
                x=grouped['Course'],
                y=grouped['CGPA'],
                marker_color='#764ba2',
                text=[f"{val:.2f}" for val in grouped['CGPA']],
                textposition='outside'
            ))
            
            fig.update_layout(
                title='Academic Stress and CGPA by Course (Top 5)',
                title_font_size=20,
                xaxis_title="Course",
                yaxis_title="Average Value",
                barmode='group',
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 **Insight:** Compare how academic pressure levels and CGPA vary across different courses. Higher bars indicate higher stress or better performance.")
        else:
            st.warning("Required columns not found for this analysis.")
        
        # Social Support vs Academic Pressure
        st.subheader("🤝 Social Support vs Academic Pressure")
        if 'Social Support' in df_filtered.columns and 'Academic Pressure' in df_filtered.columns:
            # Define order for better visualization
            support_order = ['Low', 'Moderate', 'High']
            pressure_order = ['Low', 'Moderate', 'High', 'Very High']
            
            # Create cross-tabulation
            cross_tab = pd.crosstab(df_filtered['Social Support'], df_filtered['Academic Pressure'])
            
            # Reindex to ensure proper order
            cross_tab = cross_tab.reindex(
                index=[s for s in support_order if s in cross_tab.index],
                columns=[p for p in pressure_order if p in cross_tab.columns],
                fill_value=0
            )
            
            # Create grouped bar chart
            fig = go.Figure()
            
            colors = ['#84fab0', '#8fd3f4', '#ffa07a', '#ff6b6b']
            
            for idx, pressure_level in enumerate(cross_tab.columns):
                fig.add_trace(go.Bar(
                    name=pressure_level,
                    x=cross_tab.index,
                    y=cross_tab[pressure_level],
                    marker_color=colors[idx % len(colors)],
                    text=cross_tab[pressure_level],
                    textposition='auto'
                ))
            
            fig.update_layout(
                title='Social Support vs Academic Pressure',
                title_font_size=20,
                xaxis_title="Social Support Level",
                yaxis_title="Number of Students",
                barmode='group',
                template='plotly_white',
                height=500,
                legend_title="Academic Pressure"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 **Insight:** This shows how academic pressure is distributed across different levels of social support. Students with higher social support may experience different pressure patterns.")
        else:
            st.warning("Required columns not found for this analysis.")

        # Financial Stress & Marital Status vs Depression
        st.subheader("💍 Financial Stress & Marital Status vs Depression")
        financial_col = 'Financial Stress'
        depression_col = 'Depression'
        marital_candidates = ['Marital Status', 'Marital status', 'Marital_Status', 'MaritalStatus']
        marital_col = next((c for c in marital_candidates if c in df_filtered.columns), None)
        if marital_col is None:
            marital_col = next((c for c in df_filtered.columns if 'marital' in c.lower()), None)
        
        if marital_col and financial_col in df_filtered.columns and depression_col in df_filtered.columns:
            df_temp = df_filtered[[marital_col, financial_col, depression_col]].dropna()
            
            if not df_temp.empty:
                # Calculate depression rate (% Yes) for each combination
                grouped = (
                    df_temp
                    .groupby([marital_col, financial_col])[depression_col]
                    .apply(lambda s: (s == 'Yes').mean() * 100)
                    .reset_index(name='DepressionRate')
                )
                
                if not grouped.empty:
                    # Pivot for heatmap
                    finance_order = ['Low', 'Moderate', 'High', 'Very High']
                    pivot = grouped.pivot(index=marital_col, columns=financial_col, values='DepressionRate')
                    pivot = pivot.reindex(columns=[c for c in finance_order if c in pivot.columns])
                    pivot = pivot.sort_index()

                    fig = go.Figure(data=go.Heatmap(
                        x=pivot.columns,
                        y=pivot.index,
                        z=pivot.values,
                        colorscale='RdBu',
                        reversescale=True,
                        colorbar=dict(title='Depression %'),
                        hovertemplate=(
                            f"{marital_col}: %{{y}}<br>Financial Stress: %{{x}}<br>Depression Rate: %{{z:.1f}}%<extra></extra>"
                        )
                    ))

                    # Add value annotations
                    for y_idx, y_val in enumerate(pivot.index):
                        for x_idx, x_val in enumerate(pivot.columns):
                            val = pivot.iloc[y_idx, x_idx]
                            if pd.notna(val):
                                fig.add_annotation(x=x_val, y=y_val, text=f"{val:.1f}%", showarrow=False, font=dict(color='white'))

                    fig.update_layout(
                        title='Depression Rate by Marital Status and Financial Stress',
                        title_font_size=20,
                        xaxis_title='Financial Stress',
                        yaxis_title='Marital Status',
                        template='plotly_white',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("💡 **Insight:** Heatmap shows the percentage of students reporting depression across financial stress levels and marital status. Darker colors indicate higher depression rates.")
                else:
                    st.warning("Not enough data for this analysis.")
            else:
                st.warning("Not enough data for this analysis.")
        else:
            st.warning("Required columns not found for this analysis.")
    
    with tab6:
        st.header("🤖 RAG-Based Q&A System")
        st.write("Ask questions about the mental health dataset!")
        
        if RAG_AVAILABLE:
            # Initialize RAG system
            if 'rag_system' not in st.session_state:
                with st.spinner("Initializing RAG system..."):
                    st.session_state.rag_system = MentalHealthRAG(df)
            
            # Question input
            question = st.text_input("Enter your question:", 
                                     placeholder="e.g., What is the average CGPA of students?")
            
            if st.button("Get Answer", type="primary"):
                if question:
                    with st.spinner("Searching knowledge base..."):
                        response = st.session_state.rag_system.query(question)
                        st.success("Answer:")
                        st.write(response)
                else:
                    st.warning("Please enter a question!")
            
            # Sample questions
            st.subheader("💡 Sample Questions")
            sample_questions = [
                "What is the distribution of students by gender?",
                "What are the mental health statistics?",
                "What is the average CGPA?",
                "How many students are in each year of study?"
            ]
            for q in sample_questions:
                if st.button(q, key=q):
                    with st.spinner("Searching..."):
                        response = st.session_state.rag_system.query(q)
                        st.success("Answer:")
                        st.write(response)
        else:
            st.error("RAG system not available. Please install required libraries:")
            st.code("pip install sentence-transformers chromadb")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>🎓 Student Mental Health Analytics Dashboard | Built with Streamlit & Python</p>
            <p>Data Source: Kaggle - Mental Health of Students Dataset</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()