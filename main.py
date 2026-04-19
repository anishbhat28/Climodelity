import streamlit as st

pages = [
    st.Page("app.py", title="Upload", icon="📤", default=True),
    st.Page("dashboard.py", title="Dashboard", icon="📊"),
]

pg = st.navigation(pages)
pg.run()
