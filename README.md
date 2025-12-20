# 🧠 Student Mental Health Analytics Dashboard

A powerful **Streamlit web application** that transforms student mental health data into actionable insights. Featuring interactive visualizations, intelligent analytics, and conversational AI for exploring mental health patterns across student populations.

## 🎯 What It Does

This application transforms raw student mental health data into visual stories. It helps educators, researchers, and counselors understand the mental health landscape of their student population—covering depression, anxiety, academic pressure, social support, and treatment-seeking behaviors.

**Who should use this?**
- University mental health departments
- Educational researchers
- Institutional decision-makers
- Data science professionals

## 🚀 Key Capabilities

### 📊 Visual Analytics
- Beautiful, interactive charts (histograms, pie charts, bar charts, heatmaps)
- Real-time filtering and exploration
- Professional gradient styling
- Mobile-responsive design

### 🧠 Insight Generation
- Track depression and anxiety patterns
- Discover variable relationships
- Analyze lifestyle factors (sleep, exercise, support systems)
- Identify at-risk student populations

### 📈 Performance Insights
- Connect academic pressure to CGPA
- Measure social support impact
- Visualize financial stress effects
- Compare outcomes across courses

### 🤖 Intelligent Q&A
- Ask natural language questions about the data
- Get AI-powered answers from a knowledge base
- Explore patterns without writing code
- Sample questions included for learning

---

## 📁 Project Layout

```
ai-project-main/
│
├── app.py                              Main dashboard application
├── rag/                                AI question-answering module
│   ├── __init__.py
│   ├── mental_health_rag.py           Core RAG logic
│   └── README.md
│
├── Student_Mental_Health_CLEANED.csv  Ready-to-use dataset
├── Student Mental health.csv           Original data (backup)
│
├── requirements.txt                    Dependencies
├── .env                                Config (create this yourself)
├── README.md                           You're reading it
│
└── ai/                                 Python environment
    └── (preinstalled with all packages)
```

---

## 🚀 Getting Started

### 1. Activate Python Environment

**Windows PowerShell:**
```powershell
E:\ai-project-main\ai\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
E:\ai-project-main\ai\Scripts\activate.bat
```

### 2. Create Configuration File

In the project folder, create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

(Most are pre-installed in the virtual environment)

### 4. Run the Dashboard

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

---

## 🎮 Dashboard Walkthrough

### Tab 1: Overview
**See the big picture**
- Dataset summary (row count, column count)
- Statistical overview of numeric variables
- First 10 data rows

### Tab 2: Demographics
**Understand your student population**
- Age distribution
- Gender breakdown
- Popular courses
- Year-wise student counts
- CGPA patterns by gender

### Tab 3: Mental Health
**Explore health patterns**
- Depression prevalence (%)
- Anxiety statistics
- Mental health by study year
- Sleep quality, exercise, social support, financial stress
- Connection between CGPA and depression
- Treatment-seeking behavior

### Tab 4: Correlations
**Find relationships**
- Correlation heatmap of all numeric variables
- Top 5 strongest correlations
- Identify which factors influence each other

### Tab 5: Performance Analysis
**Academic insights**
- Filter by academic pressure level
- Academic stress vs CGPA (top 5 courses)
- Social support impact on academic pressure
- Financial stress & marital status effects on depression

### Tab 6: AI Q&A
**Ask questions, get answers**
- Type natural language questions
- Click sample questions to explore
- AI searches the dataset and provides intelligent responses

---

## 🛠️ Technical Stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | Streamlit, HTML/CSS |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Data Processing** | Pandas, NumPy |
| **AI & Search** | Sentence Transformers, ChromaDB |
| **Language Model** | Groq (LLaMA 3.3 70B) |
| **Config** | Python-dotenv |

---

## 📊 Data Overview

### Source
Kaggle: [Mental Health of Students Dataset](https://www.kaggle.com/datasets/aminasalamat/mental-health-of-students-dataset)

### Main Columns

| Data | Meaning |
|------|---------|
| Age | Student age |
| Gender | Male/Female/Other |
| Course | Degree program |
| StudyYear | Year 1-4 |
| CGPA | Grade point average |
| Depression | Clinical indicator |
| Anxiety | Stress levels |
| Treatment | Professional help sought |
| Sleep Quality | Health metric |
| Exercise Frequency | Lifestyle indicator |
| Social Support | Relationship strength |
| Financial Stress | Economic pressure |
| Academic Pressure | Study burden |

---

## 🤖 Smart Q&A System

How does the AI understand your questions?

1. **You ask a question** - "What percentage of students have anxiety?"
2. **AI finds relevant info** - Searches a knowledge base 
3. **Context is added** - Retrieves matching statistics and patterns
4. **LLM generates answer** - Uses Groq's fast language model
5. **You get insights** - Conversational, accurate response

The system works even without internet (uses local embeddings), and gracefully handles API outages.

### app.py (~700 lines)
**The main dashboard**
- Clean, inline code for easy reading
- 6 interactive tabs with visualizations
- Data loading and normalization
- Real-time chart generation
- Sidebar with quick metrics

### rag/ folder
**AI-powered insights module**
- `mental_health_rag.py` - Embedding and question-answering logic
- `__init__.py` - Module exports
- Separate from main app for clean architecture

### Configuration Files
- `requirements.txt` - All Python packages
- `.env` - Your API key (you create this)
- `README.md` - Documentation (this file)

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| **App won't start** | Activate venv first: `Activate.ps1` |
| **"Module rag not found"** | Run from project root: `cd ai-project-main` |
| **No GROQ API key error** | Create `.env` file with your API key |
| **Slow first load** | RAG system building embeddings (only first time) |
| **Dataset not found** | Check file is in project root |
| **Port 8501 in use** | Run on different port: `streamlit run app.py --server.port 8502` |

---

## 💡 Quick Tips

✅ **First run slower?** Normal—RAG system initializes on first load  
✅ **Want to try Q&A?** Click sample questions first  
✅ **Filters not working?** Make sure at least one option is selected  
✅ **Charts look odd?** Refresh browser or restart app  
✅ **Confused by numbers?** Check Dataset section above for column meanings




