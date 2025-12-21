"""RAG logic for the Student Mental Health app.
Encapsulates embeddings, vector store setup, and LLM-backed query.
"""
from __future__ import annotations

import os
import warnings
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment variables for API keys, etc.
load_dotenv()

# Attempt to import heavy/optional RAG dependencies
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    from groq import Groq
    RAG_AVAILABLE = True
except Exception:
    # Keep the module importable even if RAG deps are missing
    RAG_AVAILABLE = False
    SentenceTransformer = None  # type: ignore
    chromadb = None  # type: ignore
    Settings = None  # type: ignore
    Groq = None  # type: ignore


class MentalHealthRAG:
    def __init__(self, df: pd.DataFrame):
        if not RAG_AVAILABLE:
            self.available = False
            return

        self.available = True
        self.df = df
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize Groq API client
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key or api_key == 'your_groq_api_key_here':
            st.warning("⚠️ GROQ_API_KEY not found in .env file. Please add your API key.")
            self.groq_client = None
        else:
            try:
                self.groq_client = Groq(api_key=api_key)
            except Exception as e:
                st.error(f"Failed to initialize Groq client: {e}")
                self.groq_client = None

        # Initialize ChromaDB (in-memory unless persistent dir configured externally)
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True,
        ))

        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(name="mental_health_docs")
        except Exception:
            # Create new collection if it doesn't exist
            self.collection = self.client.create_collection(
                name="mental_health_docs",
                metadata={"hnsw:space": "cosine"},
            )
            self._create_knowledge_base()

    def _create_knowledge_base(self) -> None:
        """Create comprehensive embeddings from the entire dataset."""
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        # 1. Overall Dataset Statistics
        stats_doc = f"""Dataset Overview:
Total Students: {len(self.df)}
Total Columns: {len(self.df.columns)}
Columns: {', '.join(self.df.columns)}

Numeric Statistics:
{self.df.describe().to_string()}
"""
        documents.append(stats_doc)
        metadatas.append({"type": "overview"})
        ids.append("overview")

        # 2. Column-wise Detailed Insights
        for col in self.df.columns:
            if self.df[col].dtype == 'object' or self.df[col].dtype.name == 'category':
                # Categorical columns
                value_counts = self.df[col].value_counts()
                doc = f"""Column: {col}
Type: Categorical
Total Values: {len(value_counts)}
Distribution:
{value_counts.to_string()}

Percentages:
{(value_counts / len(self.df) * 100).round(2).to_string()}%
"""
            else:
                # Numerical columns
                doc = f"""Column: {col}
Type: Numerical
Mean: {self.df[col].mean():.2f}
Median: {self.df[col].median():.2f}
Std Dev: {self.df[col].std():.2f}
Min: {self.df[col].min():.2f}
Max: {self.df[col].max():.2f}
"""
            documents.append(doc)
            metadatas.append({"type": "column_stats", "column": col})
            ids.append(f"col_{col}")

        # 3. Mental Health Specific Statistics
        mental_health_cols = ['Depression', 'Anxiety', 'PanicAttack', 'Treatment']
        for col in mental_health_cols:
            if col in self.df.columns:
                counts = self.df[col].value_counts()
                percentages = (counts / len(self.df) * 100).round(2)
                doc = f"""Mental Health Indicator: {col}
{counts.to_string()} students
Percentage: {percentages.to_string()}%

Analysis: Out of {len(self.df)} students surveyed:
"""
                for val, cnt in counts.items():
                    doc += f"\n- {cnt} students ({percentages[val]:.1f}%) reported {col}: {val}"

                documents.append(doc)
                metadatas.append({"type": "mental_health", "indicator": col})
                ids.append(f"mh_{col}")

        # 4. Cross-tabulation insights
        if 'Gender' in self.df.columns and 'Depression' in self.df.columns:
            cross_tab = pd.crosstab(self.df['Gender'], self.df['Depression'], normalize='index') * 100
            doc = f"""Gender vs Depression Analysis:
{cross_tab.round(2).to_string()}

This shows the percentage of students with depression across different genders.
"""
            documents.append(doc)
            metadatas.append({"type": "cross_analysis", "fields": "Gender_Depression"})
            ids.append("cross_gender_depression")

        # 5. Course-wise analysis
        if 'Course' in self.df.columns:
            course_stats = self.df.groupby('Course').size().sort_values(ascending=False).head(10)
            doc = f"""Top 10 Courses by Student Count:
{course_stats.to_string()}

This represents the most popular courses among the surveyed students.
"""
            documents.append(doc)
            metadatas.append({"type": "course_analysis"})
            ids.append("course_stats")

        # 6. CGPA Analysis
        if 'CGPA' in self.df.columns:
            cgpa_bins = pd.cut(self.df['CGPA'], bins=[0, 2.5, 3.0, 3.5, 4.0], labels=['Low', 'Medium', 'Good', 'Excellent'])
            cgpa_dist = cgpa_bins.value_counts()
            doc = f"""CGPA Distribution:
{cgpa_dist.to_string()}

Average CGPA: {self.df['CGPA'].mean():.2f}
Median CGPA: {self.df['CGPA'].median():.2f}
"""
            documents.append(doc)
            metadatas.append({"type": "cgpa_analysis"})
            ids.append("cgpa_stats")

        # 7. Age Distribution
        if 'Age' in self.df.columns:
            age_stats = self.df['Age'].describe()
            doc = f"""Age Distribution:
Mean Age: {age_stats['mean']:.1f} years
Median Age: {age_stats['50%']:.1f} years
Age Range: {age_stats['min']:.0f} to {age_stats['max']:.0f} years
Most students are between {age_stats['25%']:.0f} and {age_stats['75%']:.0f} years old.
"""
            documents.append(doc)
            metadatas.append({"type": "age_analysis"})
            ids.append("age_stats")

        # Generate embeddings for all documents
        embeddings = self.model.encode(documents, show_progress_bar=False)

        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def query(self, question: str) -> str:
        """Query the RAG system with optional LLM integration."""
        if not self.available:
            return "RAG system not available. Please install required libraries: pip install sentence-transformers chromadb groq"

        # Generate embedding for question
        question_embedding = self.model.encode([question])

        # Query ChromaDB for relevant context
        results = self.collection.query(
            query_embeddings=question_embedding.tolist(),
            n_results=5,  # Get top 5 most relevant documents
        )

        # Prepare context from retrieved documents
        context = "\n\n---\n\n".join(results['documents'][0])

        # If Groq client is available, use LLM for better answers
        if getattr(self, 'groq_client', None):
            try:
                prompt = f"""You are a helpful data analyst assistant specializing in student mental health data. 
You have access to a comprehensive dataset about students' mental health, including information about depression, anxiety, panic attacks, and treatment-seeking behavior.

Context from the dataset:
{context}

User Question: {question}

Instructions:
- Answer the question accurately based on the provided context
- Use specific numbers and statistics from the context
- If the question asks for trends or patterns, explain them clearly
- If the context doesn't contain enough information to fully answer, say so
- Be concise but informative
- Format your response in a clear, easy-to-read manner

Answer:"""

                response = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a helpful data analyst assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=1024,
                )

                return response.choices[0].message.content

            except Exception as e:
                st.error(f"Error calling Groq API: {e}")
                # Fallback to simple context-based response
                return f"**Retrieved Information:**\n\n{context}\n\n**Question:** {question}\n\n*Note: LLM integration failed. Showing raw context.*"
        else:
            # Fallback without LLM
            return (
                f"**Retrieved Information from Dataset:**\n\n{context}\n\n"
                f"**Your Question:** {question}\n\n"
                f"*💡 Tip: Add your GROQ_API_KEY to the .env file for AI-powered answers!*"
            )
