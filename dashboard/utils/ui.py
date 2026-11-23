
import streamlit as st


def apply_custom_ui():

    st.markdown("""
    <style>

    .stApp {
        background-color: #F8FAFC;
    }

    .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background-color: #0F172A;
    }

    [data-testid="stSidebar"] * {
        color: #E2E8F0;
    }

    [data-testid="stSidebarNav"] {
        padding-top: 0.5rem;
    }

    [data-testid="stSidebarNav"] a {
        border-radius: 8px;
        margin: 2px 0;
    }

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #E2E8F0;
        padding: 18px;
        border-radius: 12px;
    }

    .stButton button {
        border-radius: 8px;
        font-weight: 600;
    }

    h1, h2, h3 {
        color: #0F172A;
    }

    /* Sidebar branding */
    [data-testid="stSidebar"] .sidebar-brand {
        padding: 10px 16px 20px 16px;
        margin-bottom: 8px;
    }

    [data-testid="stSidebar"] .sidebar-brand-title {
        font-size: 20px;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: #FFFFFF;
        margin-bottom: 4px;
    }

    [data-testid="stSidebar"] .sidebar-brand-subtitle {
        font-size: 12px;
        color: #94A3B8;
        letter-spacing: 0.3px;
    }

    </style>
    """, unsafe_allow_html=True)

    # Sidebar branding - navigation'ın üstünde
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-title">FRAUD INTELLIGENCE</div>
                <div class="sidebar-brand-subtitle">
                    Detection & Investigation Platform
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


