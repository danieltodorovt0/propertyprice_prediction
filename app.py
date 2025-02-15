import streamlit as st
import urllib.request
from http.cookiejar import CookieJar
import re
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Configure page settings
st.set_page_config(
    page_title="Luxembourg Property Evaluator",
    page_icon="🏠",
    layout="wide"
)

# Add CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        with open('models/catboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/shap_explainer.pkl', 'rb') as f:
            explainer = pickle.load(f)
        return model, explainer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Function to scrape and process property data
def process_property_data(property_id):
    # Variables to extract (same as in your original script)
    variables = [
        "id", "lon", "lat", "floor", "group", "type", "isNewBuild", "createdAt",
        "updatedAt", "price", "energyClass", "thermalInsulationClass", "rooms",
        "bedrooms", "bathrooms", "showers", "basement", "garages", "indoorParking",
        "outdoorParking", "carparks", "surface", "groundSurface", "constructionYear",
        "renovationYear", "floors", "bathtubs", "toilets", "isRenovated",
        "hasEquippedKitchen", "hasSeparateKitchen", "hasOpenKitchen", "hasSolarPanels",
        "hasSeparateToilets", "hasCellar", "hasAttic", "hasLaundryRoom", "cityName",
        "hasGarden", "hasTerrace", "hasBalcony", "gardenSurface", "terraceSurface",
        "balconySurface", "livingRoomSurface", "hasLift", "hasParquet",
        "isSocialHousing", "isEmphyteuticLease", "hasOffice", "hasGasHeating",
        "hasOilHeating", "hasElectricHeating", "hasGeoThermalHeating",
        "hasPumpHeating", "hasPelletsHeating", "hasPhotovoltaicHeating"
    ]
    
    webpage_url = f"https://www.athome.lu/id-{property_id}.html"
    
    try:
        cookie_jar = CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
        opener.addheaders = [
            ('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
             '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3')
        ]
        
        with opener.open(webpage_url) as response:
            parsed_page = response.read().decode('utf-8')
            
        results = []
        for var in variables:
            pattern = r'"{}"\s*:\s*"?(.*?)"?(?=,|\s*}})'.format(re.escape(var))
            match = re.search(pattern, parsed_page)
            results.append(match.group(1) if match else None)
            
        df = pd.DataFrame([results], columns=variables)
        
        ###################################################################################
        ### DATA TRANSFORMATION AND CLEANUP
        ###################################################################################

        # Transform the column
        df['hasSeparateKitchen'].fillna(False, inplace=True)
        df['hasSeparateToilets'].fillna(False, inplace=True)
        df['hasCellar'].fillna(False, inplace=True)
        df['hasAttic'].fillna(False, inplace=True)
        df['hasGarden'].fillna(False, inplace=True)
        df['hasTerrace'].fillna(False, inplace=True)
        df['hasBalcony'].fillna(False, inplace=True)
        df['hasLift'].fillna(False, inplace=True)
        df['gardenSurface'].fillna(0, inplace=True)
        df['terraceSurface'].fillna(0, inplace=True)
        df['balconySurface'].fillna(0, inplace=True)
        df['livingRoomSurface'].fillna(0, inplace=True)
        df['constructionYear'].fillna(0, inplace=True)
        df['renovationYear'].fillna(0, inplace=True)

        # Replace if null the column
        df['hasSeparateKitchen'] = df['hasSeparateKitchen'].replace('null', False)
        df['hasSeparateToilets'] = df['hasSeparateToilets'].replace('null', False)
        df['hasCellar'] = df['hasCellar'].replace('null', False)
        df['hasAttic'] = df['hasAttic'].replace('null', False)
        df['hasGarden'] = df['hasGarden'].replace('null', False)
        df['hasTerrace'] = df['hasTerrace'].replace('null', False)
        df['hasBalcony'] = df['hasBalcony'].replace('null', False)
        df['hasLift'] = df['hasLift'].replace('null', False)
        df['gardenSurface'] = df['gardenSurface'].replace('null', 0)
        df['terraceSurface'] = df['terraceSurface'].replace('null', 0)
        df['balconySurface'] = df['balconySurface'].replace('null', 0)
        df['livingRoomSurface'] = df['livingRoomSurface'].replace('null', 0)
        df['constructionYear'] = df['constructionYear'].replace('null', 0)
        df['renovationYear']=  df['renovationYear'].replace('null', 0)

        # Define the replacements
        replacements = {'true': True, 'false': False}

        # Replace strings with boolean values
        df = df.replace(replacements)

        # Calculate distance from city center
        def haversine(lon1, lat1, lon2, lat2):
            """
            Calculate the great-circle distance between two points 
            on the Earth's surface using the haversine formula.
            """
            lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            r = 6371  # Radius of Earth in kilometers
            return c * r

        city_center_lon = 6.131935
        city_center_lat = 49.611622
        df['distance_from_city_center'] = haversine(df['lon'].astype('float64'), df['lat'].astype('float64'), city_center_lon, city_center_lat)

        # Parse `createdAt` and `updatedAt` with different formats
        try:
            # Parse `createdAt` with format 'YYYYMMDDTHHMMSSZ'
            df['createdAt'] = pd.to_datetime(df['createdAt'], format='%Y%m%dT%H%M%SZ', errors='coerce')
            # Parse `updatedAt` with ISO format 'YYYY-MM-DDTHH:MM:SS'
            df['updatedAt'] = pd.to_datetime(df['updatedAt'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
        except Exception as e:
            print(f"Error parsing dates: {e}")

        # Report any nulls after date conversion
        if df['createdAt'].isnull().any():
            print("Warning: Null values found in 'createdAt' after date conversion.")
        if df['updatedAt'].isnull().any():
            print("Warning: Null values found in 'updatedAt' after date conversion.")

        # Calculate difference in days from a specific date
        today = pd.to_datetime('2025-02-11')
        df['createdAt_days_diff'] = (today - df['createdAt']).dt.days
        df['updatedAt_days_diff'] = (today - df['updatedAt']).dt.days

        # Calculate price per sqm
        df['price_per_sqm'] = df['price'].astype('float64') / df['surface'].astype('float64')

        # Define data types for each feature
        feature_with_data_types = {
            #'lon': 'float64',
            #'lat': 'float64',
            ##'floor': 'int64',
            'group': 'category',
            ##'type': 'category',
            'isNewBuild': 'bool',
            'energyClass': 'category',
            'thermalInsulationClass': 'category',
            'rooms': 'int64',
            'bedrooms': 'int64',
            'bathrooms': 'int64',
            ##'showers': 'int64',
            'basement': 'int64',
            'garages': 'int64',
            ##'indoorParking': 'int64',
            ##'outdoorParking': 'int64',
            ##'carparks': 'int64',
            'surface': 'float64',
            ##'bathtubs': 'int64',
            'toilets': 'int64',
            'isRenovated': 'bool',
            'hasEquippedKitchen': 'bool',
            ##'hasSeparateKitchen': 'bool',
            'hasOpenKitchen': 'bool',
            ##'hasSolarPanels': 'bool',
            'hasSeparateToilets': 'bool',
            'hasCellar': 'bool',
            ##'hasAttic': 'bool',
            'hasLaundryRoom': 'bool',
            ##'hasGarden': 'bool',
            'hasTerrace': 'bool',
            'hasBalcony': 'bool',
            'gardenSurface': 'float64',
            'terraceSurface': 'float64',
            'balconySurface': 'float64',
            ##'livingRoomSurface': 'float64',
            'hasLift': 'bool',
            'hasParquet': 'bool',
            ##'isSocialHousing': 'bool',
            'isEmphyteuticLease': 'bool',
            ##'hasOffice': 'bool',
            'hasGasHeating': 'bool',
            ##'hasOilHeating': 'bool',
            ##'hasElectricHeating': 'bool',
            ##'hasGeoThermalHeating': 'bool',
            'hasPumpHeating': 'bool',
            ##'hasPelletsHeating': 'bool',
            ##'hasPhotovoltaicHeating': 'bool',
            'distance_from_city_center': 'float64',
            'createdAt_days_diff': 'int64',
            ##'updatedAt_days_diff': 'int64',
            'constructionYear': 'int64',
            'renovationYear': 'int64',
            'cityName': 'category'
            #'price_per_sqm': 'float64
            #'groundSurface': 'float64
            #'floors': 'int64
            #'price': 'float64
        }

        features = list(feature_with_data_types.keys())
        data_types = list(feature_with_data_types.values())

        # Split dataset into features and target variable
        X = df[features]
        y = df['price']

        # Change data types
        for col, dtype in feature_with_data_types.items():
            X[col] = X[col].astype(dtype)
        
        return df
        
    except Exception as e:
        raise Exception(f"Error processing property data: {str(e)}")

# Main app interface
st.title("Luxembourg Property Evaluator 🏠")
st.write("Enter a property ID from athome.lu to get an evaluation")

# Input section
with st.form("property_form"):
    property_id = st.text_input("Property ID:", "8471575")
    submitted = st.form_submit_button("Evaluate Property")

if submitted:
    try:
        with st.spinner("Loading models..."):
            model, explainer = load_models()
            
        if model is None or explainer is None:
            st.error("Failed to load models. Please try again.")
            st.stop()
            
        with st.spinner("Fetching and processing property data..."):
            df = process_property_data(property_id)
            
        # Create prediction
        with st.spinner("Generating prediction..."):
            prediction = model.predict(df)[0]
            shap_values = explainer(df)
            
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Predicted Price")
            st.metric(
                label="Property Value",
                value=f"€{prediction:,.2f}"
            )
            
        with col2:
            st.subheader("Property Details")
            st.write(f"Location: {df['cityName'].iloc[0]}")
            st.write(f"Surface: {df['surface'].iloc[0]} m²")
            st.write(f"Rooms: {df['rooms'].iloc[0]}")
            
        # SHAP visualization
        st.subheader("Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
        
        # Additional property insights
        st.subheader("Property Insights")
        price_per_sqm = prediction / float(df['surface'].iloc[0])
        st.write(f"Price per m²: €{price_per_sqm:,.2f}")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check the property ID and try again.")

# Footer
st.markdown("---")
st.caption("Data source: athome.lu | Prices are estimates only")
