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

st.set_page_config(page_title="Restaurant Success Predictor", layout="wide")

@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?export=download&id=1c8elHtu79a5FGImyMKjZ4kP2Bu0Nf7I7"
    df = pd.read_csv(url)
    return df

df = load_data()

drop_cols = [
    'mapurlwithcor', 'name', 'imgsrc', 'address', 'closetime', 'isopen',
    'pricerange', 'price_range', 'serving1', 'serving2', 'serving3',
    'serving_options'
]
df_model = df.drop(columns=drop_cols)

y = df_model['success_rate'] / 10.0
X = df_model.drop(columns=['success_rate'])

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

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

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
        This app predicts restaurant success rates using ML trained on real data from Dhaka.

        **Instructions:**
        - Click anywhere on the map
        - View the predicted success rate at that location
    """)

# Title
st.title("Restaurant Success Predictor")

# Output container shown first
output_container = st.container()

# Prediction section (initially blank)
lat, lon, raw, percent = None, None, None, None

# Map
m = folium.Map(location=[23.8103, 90.4125], zoom_start=12, control_scale=True)
heat_data = df[['latitude', 'longitude', 'success_rate']].dropna().values.tolist()
HeatMap(heat_data, radius=15, blur=20, min_opacity=0.5).add_to(m)
map_data = st_folium(m, width=900, height=550)

# Show prediction if clicked
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    with st.spinner("Calculating success rate..."):
        raw, percent = predict_success(lat, lon)

    with output_container:
        st.markdown("## 🔍 Prediction Result")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Coordinates**")
            st.write(f"📍 Latitude: `{lat:.5f}`")
            st.write(f"📍 Longitude: `{lon:.5f}`")
        with col2:
            st.markdown("**Success Rate**")
            st.metric(
                label="Predicted Success Rate",
                value=f"{percent:.2f}%",
                delta=f"Raw: {raw:.2f} / Max: {max_success_rate:.2f}"
            )

# Footer immediately after map
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding-top: 1rem;">
    Made with ❤️ by <a href="https://github.com/Tasrif-Ahmed-Mohsin" target="_blank">Tasrif_Ahmed</a>
</div>
""", unsafe_allow_html=True)
