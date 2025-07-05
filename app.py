import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from mlxtend.frequent_patterns import apriori, association_rules

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("APS Dataset.csv")
    return df

df = load_data()

st.set_page_config(page_title="Airline Passenger Satisfaction Dashboard", layout="wide")
st.title("✈️ Airline Passenger Satisfaction Analytics Dashboard")
st.markdown("""
A data insight platform for the Operations and Marketing Heads.
Explore satisfaction drivers, customer segments, and actionable insights to elevate passenger experience.
""")

# Sidebar Filters
st.sidebar.header("Filters")
class_options = st.sidebar.multiselect("Class", options=df["Class"].unique(), default=df["Class"].unique())
gender_options = st.sidebar.multiselect("Gender", options=df["Gender"].unique(), default=df["Gender"].unique())
travel_type = st.sidebar.multiselect("Type of Travel", options=df["Type of Travel"].unique(), default=df["Type of Travel"].unique())

filtered_df = df[
    (df["Class"].isin(class_options)) &
    (df["Gender"].isin(gender_options)) &
    (df["Type of Travel"].isin(travel_type))
]

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", "Macro Trends", "Micro Insights", "Predictive Analytics", "Clustering & Segmentation", "Association Mining"
])

# 1. OVERVIEW TAB
with tab1:
    st.header("Dashboard Overview")
    st.markdown("""
    This dashboard offers a comprehensive view of passenger satisfaction. Navigate through the tabs for:
    - Macro trends in satisfaction & demographics
    - Micro-level service analysis
    - Predictive ML models
    - Customer segmentation
    - Service association mining
    """)
    st.markdown("#### Quick Stats")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Passengers", len(filtered_df))
    col2.metric("Satisfied (%)", f"{filtered_df['satisfaction'].eq('satisfied').mean()*100:.1f}%")
    col3.metric("Avg. Age", int(filtered_df['Age'].mean()))
    col4.metric("Avg. Flight Distance", int(filtered_df['Flight Distance'].mean()))

    st.markdown("#### Satisfaction Distribution")
    fig = px.pie(filtered_df, names='satisfaction', title="Passenger Satisfaction Breakdown")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Class Distribution")
    fig = px.histogram(filtered_df, x="Class", color="satisfaction", barmode="group", title="Class Distribution by Satisfaction")
    st.plotly_chart(fig, use_container_width=True)

# 2. MACRO TRENDS TAB
with tab2:
    st.header("Macro Trends in Passenger Experience")
    st.markdown("Visualize high-level patterns in passenger demographics and satisfaction.")

    st.subheader("Satisfaction by Gender")
    fig = px.histogram(filtered_df, x="Gender", color="satisfaction", barmode="group", title="Satisfaction by Gender")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Satisfaction by Customer Type")
    fig = px.histogram(filtered_df, x="Customer Type", color="satisfaction", barmode="group", title="Satisfaction by Customer Type")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Satisfaction by Age")
    fig = px.box(filtered_df, x="satisfaction", y="Age", color="satisfaction", title="Age Distribution by Satisfaction")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Satisfaction by Type of Travel")
    fig = px.histogram(filtered_df, x="Type of Travel", color="satisfaction", barmode="group", title="Satisfaction by Type of Travel")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Satisfaction by Flight Distance")
    fig = px.histogram(filtered_df, x="Flight Distance", color="satisfaction", nbins=40, barmode="overlay", title="Satisfaction by Flight Distance")
    st.plotly_chart(fig, use_container_width=True)

# 3. MICRO INSIGHTS TAB
with tab3:
    st.header("Micro Insights: Service Ratings & Delays")
    st.markdown("Analyze specific service factors and their impact on satisfaction.")

    service_cols = [
        "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
        "Gate location", "Food and drink", "Online boarding", "Seat comfort",
        "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
        "Checkin service", "Inflight service", "Cleanliness"
    ]
    # Service ratings distribution
    st.subheader("Service Ratings Heatmap")
    avg_ratings = filtered_df.groupby("satisfaction")[service_cols].mean()
    fig, ax = plt.subplots(figsize=(12,5))
    sns.heatmap(avg_ratings, annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig, use_container_width=True)

    st.subheader("Delay Analysis")
    fig = px.box(filtered_df, x="satisfaction", y="Departure Delay in Minutes", color="satisfaction", title="Departure Delays by Satisfaction")
    st.plotly_chart(fig, use_container_width=True)
    fig = px.box(filtered_df, x="satisfaction", y="Arrival Delay in Minutes", color="satisfaction", title="Arrival Delays by Satisfaction")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pairplot: Age vs. Flight Distance vs. Satisfaction")
    fig = px.scatter(filtered_df, x="Age", y="Flight Distance", color="satisfaction", size="Flight Distance", hover_data=["Class"])
    st.plotly_chart(fig, use_container_width=True)

# 4. PREDICTIVE ANALYTICS TAB
with tab4:
    st.header("Predictive Analytics: Satisfaction Classification & Feature Importance")
    st.markdown("Train a classification model to predict satisfaction and reveal the most important drivers.")

    st.markdown("**Feature Importance:** Shows which features most influence satisfaction predictions.")

    # Prepare data
    data_ml = filtered_df.copy()
    data_ml = data_ml.drop(['id', 'Unnamed: 0'], axis=1, errors='ignore')
    data_ml = data_ml.dropna(subset=["satisfaction"])
    data_ml["satisfaction"] = data_ml["satisfaction"].map({'satisfied': 1, 'neutral or dissatisfied': 0})

    categorical_cols = data_ml.select_dtypes(include='object').columns
    data_ml = pd.get_dummies(data_ml, columns=categorical_cols, drop_first=True)

    X = data_ml.drop("satisfaction", axis=1)
    y = data_ml["satisfaction"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Model Accuracy:** {acc*100:.2f}%")

    st.markdown("**Confusion Matrix:**")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Dissatisfied","Satisfied"], yticklabels=["Dissatisfied","Satisfied"])
    st.pyplot(fig, use_container_width=True)

    st.markdown("**Feature Importance:**")
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)[:15]
    fig = px.bar(importances, orientation="h", title="Top 15 Feature Importances")
    st.plotly_chart(fig, use_container_width=True)

# 5. CLUSTERING & SEGMENTATION TAB
with tab5:
    st.header("Clustering & Passenger Segmentation")
    st.markdown("Uncover distinct passenger segments based on travel patterns and service preferences.")

    cluster_features = ["Age", "Flight Distance"] + service_cols
    cluster_data = filtered_df[cluster_features].dropna()
    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(cluster_data)

    # Choose k via slider
    k = st.slider("Select Number of Clusters (K)", 2, 8, 4)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(cluster_scaled)
    cluster_data["Cluster"] = clusters

    st.markdown(f"**PCA Visualization of {k} Clusters:**")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(cluster_scaled)
    cluster_data["pca1"] = pca_result[:,0]
    cluster_data["pca2"] = pca_result[:,1]

    fig = px.scatter(cluster_data, x="pca1", y="pca2", color=cluster_data["Cluster"].astype(str), title="Passenger Segments (PCA projection)", hover_data=["Age", "Flight Distance"])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Cluster Profiles:**")
    st.write(cluster_data.groupby("Cluster").mean())

# 6. ASSOCIATION MINING TAB
with tab6:
    st.header("Association Rule Mining")
    st.markdown("Discover frequent patterns among passenger preferences and service feedback.")

    st.markdown("""
    Toggle to display association rules based on either all passengers, or just dissatisfied ones.
    """)
    rule_focus = st.radio("Rule Focus:", ["All Passengers", "Only Dissatisfied"])

    # Prepare data for association rules
    assoc_df = filtered_df.copy()
    assoc_df = assoc_df[service_cols + ["satisfaction"]].dropna()
    for col in service_cols:
        assoc_df[col] = (assoc_df[col] >= 4).astype(int) # 1=High, 0=Low/Med

    if rule_focus == "Only Dissatisfied":
        assoc_df = assoc_df[filtered_df["satisfaction"] == "neutral or dissatisfied"]

    # Apply Apriori
    frequent_items = apriori(assoc_df, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_items, metric="confidence", min_threshold=0.6)
    st.write(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))

    if not rules.empty:
        st.markdown("**Association Rule Graph:**")
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        for idx, row in rules.iterrows():
            ant = ','.join(list(row['antecedents']))
            cons = ','.join(list(row['consequents']))
            G.add_edge(ant, cons, weight=row['confidence'])

        plt.figure(figsize=(10,6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=2500, node_color='skyblue', font_size=8, width=2, arrowsize=20)
        st.pyplot(plt)

# End of App
st.markdown("---")
st.caption("© 2024 Airline Passenger Satisfaction Dashboard | Powered by Streamlit and Data Science")
