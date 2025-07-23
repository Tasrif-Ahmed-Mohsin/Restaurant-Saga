import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from folium.plugins import HeatMap

# Page config with emoji and layout
st.set_page_config(page_title="📍 Restaurant Success Predictor", layout="wide")

# Add custom styling
# Custom CSS for background and overall style
st.markdown("""
    <style>
        .main { background-color: #f9f9fb; }
        .title { font-size: 3em; font-weight: bold; color: #0A5C59; margin-bottom: 0.5em; }
        .subtitle { font-size: 1.5em; color: #444; }
        .metric { font-size: 1.2em; }
        .footer { text-align: center; margin-top: 2rem; font-size: 0.9em; color: #777; }
        .footer a { text-decoration: none; color: #0A5C59; font-weight: bold; }
        .stMetricDelta { color: #0A5C59 !important; }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?export=download&id=1c8elHtu79a5FGImyMKjZ4kP2Bu0Nf7I7"
    df = pd.read_csv(url)
    return df

df = load_data()

# Drop unnecessary columns
drop_cols = [
    'mapurlwithcor', 'name', 'imgsrc', 'address', 'closetime', 'isopen',
    'pricerange', 'price_range', 'serving1', 'serving2', 'serving3',
    'serving_options'
]
df_model = df.drop(columns=drop_cols)

# Prepare features and target
y = df_model['success_rate'] / 10.0
X = df_model.drop(columns=['success_rate'])

# Identify columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Pipelines for preprocessing
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

@st.cache_resource
def train_model():
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()
max_success_rate = df['success_rate'].max()

def predict_success(lat, lon):
    distances = np.sqrt((X['latitude'] - lat)**2 + (X['longitude'] - lon)**2)
    nearest_index = distances.idxmin()
    nearest_sample = X.loc[nearest_index].copy()
    nearest_sample['latitude'] = lat
    nearest_sample['longitude'] = lon
    input_df = pd.DataFrame([nearest_sample])
    prediction_normalized = model.predict(input_df)[0]
    prediction_raw = prediction_normalized * 10
    percentage_score = (prediction_raw / max_success_rate) * 100
    return round(prediction_raw, 2), round(percentage_score, 2)

# Sidebar - info section
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This interactive map uses ML to predict **restaurant success** based on coordinates in **Dhaka**.

    **How to use:**
    - Click a location on the map
    - See the predicted success instantly
    """)

# Title Section
st.markdown('<div class="title">📍 Restaurant Success Predictor</div>', unsafe_allow_html=True)

# Container for output
output_container = st.container()

# Coordinates and prediction values
lat, lon, raw, percent = None, None, None, None

# Map setup
from folium import Map, Rectangle

# Define Dhaka bounding box
dhaka_bounds = [[23.65, 90.25], [23.95, 90.55]]

# Create map restricted to Dhaka
m = Map(
    location=[23.8103, 90.4125],
    zoom_start=12,
    min_zoom=11,           # Prevent zooming too far out
    max_zoom=16,           # Optional: prevent zooming in too much
    max_bounds=True        # Prevent panning outside
)

# Optionally add a visible bounding box (debug or visual purpose)
Rectangle(bounds=dhaka_bounds, fill=False, color='blue').add_to(m)

# Fit map to bounds of Dhaka
m.fit_bounds(dhaka_bounds)


heat_data = df[['latitude', 'longitude', 'success_rate']].dropna().values.tolist()
HeatMap(heat_data, radius=15, blur=20, min_opacity=0.5).add_to(m)
map_data = st_folium(m, width=1000, height=520)


# Show result if user clicked map
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    with st.spinner("🔎 Calculating..."):
        raw, percent = predict_success(lat, lon)

    with output_container:
        st.markdown('<div class="subtitle">🎯 Prediction Result</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**📌 Coordinates**")
            st.code(f"Lat: {lat:.5f}  \nLng: {lon:.5f}", language='markdown')
        with col2:
            st.markdown("**📈 Success Rate**")
            st.metric(
                label="Predicted Success Rate",
                value=f"{percent:.2f}%",
                delta=f"Raw: {raw:.2f} / Max: {max_success_rate:.2f}"
            )

# Footer immediately after map
st.markdown("---")
st.markdown("""
<div class="footer">
    Built with ❤️ by <a href="https://github.com/Tasrif-Ahmed-Mohsin" target="_blank">Tasrif</a>
</div>
""", unsafe_allow_html=True)
