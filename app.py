import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patheffects import withStroke
from matplotlib import patches
from matplotlib.path import Path
import numpy as np
import requests
import zipfile
import warnings
from colour import Color
import os
import tempfile
from shapely.geometry import Point, Polygon, LineString
import io
from scipy.interpolate import interp1d
import networkx as nx
import re
from matplotlib.ticker import AutoMinorLocator

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="World Frequency Map",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Comprehensive country name mapping
COUNTRY_NAME_MAPPING = {
    # English names variations
    'United States': 'United States',
    'USA': 'United States',
    'US': 'United States',
    'United Kingdom': 'United Kingdom',
    'UK': 'United Kingdom',
    'Great Britain': 'United Kingdom',
    'Russia': 'Russia',
    'Russian Federation': 'Russia',
    'RU': 'Russia',
    'Australia': 'Australia',
    'AU': 'Australia',
    'Poland': 'Poland',
    'PL': 'Poland',
    'Saudi Arabia': 'Saudi Arabia',
    'SA': 'Saudi Arabia',
    'KSA': 'Saudi Arabia',
    'Norway': 'Norway',
    'NO': 'Norway',
    'Finland': 'Finland',
    'FI': 'Finland',
    'Portugal': 'Portugal',
    'PT': 'Portugal',
    'Brunei': 'Brunei',
    'Brunei Darussalam': 'Brunei',
    'BN': 'Brunei',
    'China': 'China',
    'CN': 'China',
    'India': 'India',
    'IN': 'India',
    'Germany': 'Germany',
    'DE': 'Germany',
    'France': 'France',
    'FR': 'France',
    'Japan': 'Japan',
    'JP': 'Japan',
    'South Korea': 'South Korea',
    'Korea': 'South Korea',
    'KR': 'South Korea',
    'Canada': 'Canada',
    'CA': 'Canada',
    'Brazil': 'Brazil',
    'BR': 'Brazil',
    'Italy': 'Italy',
    'IT': 'Italy',
    'Spain': 'Spain',
    'ES': 'Spain',
    'Mexico': 'Mexico',
    'MX': 'Mexico',
    'Netherlands': 'Netherlands',
    'NL': 'Netherlands',
    'Sweden': 'Sweden',
    'SE': 'Sweden',
    'Switzerland': 'Switzerland',
    'CH': 'Switzerland',
    'Turkey': 'Turkey',
    'TR': 'Turkey',
    'Iran': 'Iran',
    'IR': 'Iran',
    'Egypt': 'Egypt',
    'EG': 'Egypt',
    'South Africa': 'South Africa',
    'ZA': 'South Africa',
    'Indonesia': 'Indonesia',
    'ID': 'Indonesia',
    'Pakistan': 'Pakistan',
    'PK': 'Pakistan',
    'Bangladesh': 'Bangladesh',
    'BD': 'Bangladesh',
    'Nigeria': 'Nigeria',
    'NG': 'Nigeria',
    'Vietnam': 'Vietnam',
    'VN': 'Vietnam',
    'Thailand': 'Thailand',
    'TH': 'Thailand',
    'Philippines': 'Philippines',
    'PH': 'Philippines',
    'Ukraine': 'Ukraine',
    'UA': 'Ukraine',
    'Israel': 'Israel',
    'IL': 'Israel',
    'United Arab Emirates': 'United Arab Emirates',
    'UAE': 'United Arab Emirates',
    'Singapore': 'Singapore',
    'SG': 'Singapore',
    'Malaysia': 'Malaysia',
    'MY': 'Malaysia',
    'Greece': 'Greece',
    'GR': 'Greece',
    'Czech Republic': 'Czech Republic',
    'Czechia': 'Czech Republic',
    'CZ': 'Czech Republic',
    'Austria': 'Austria',
    'AT': 'Austria',
    'Belgium': 'Belgium',
    'BE': 'Belgium',
    'Denmark': 'Denmark',
    'DK': 'Denmark',
    'Ireland': 'Ireland',
    'IE': 'Ireland',
    'New Zealand': 'New Zealand',
    'NZ': 'New Zealand',
    'Argentina': 'Argentina',
    'AR': 'Argentina',
    'Chile': 'Chile',
    'CL': 'Chile',
    'Colombia': 'Colombia',
    'CO': 'Colombia',
    'Peru': 'Peru',
    'PE': 'Peru',
    'Venezuela': 'Venezuela',
    'VE': 'Venezuela',
    'Kazakhstan': 'Kazakhstan',
    'KZ': 'Kazakhstan',
    'Belarus': 'Belarus',
    'BY': 'Belarus',
    'Romania': 'Romania',
    'RO': 'Romania',
    'Hungary': 'Hungary',
    'HU': 'Hungary',
    'Bulgaria': 'Bulgaria',
    'BG': 'Bulgaria',
    'Serbia': 'Serbia',
    'RS': 'Serbia',
    'Croatia': 'Croatia',
    'HR': 'Croatia',
    'Slovakia': 'Slovakia',
    'SK': 'Slovakia',
    'Slovenia': 'Slovenia',
    'SI': 'Slovenia',
    'Lithuania': 'Lithuania',
    'LT': 'Lithuania',
    'Latvia': 'Latvia',
    'LV': 'Latvia',
    'Estonia': 'Estonia',
    'EE': 'Estonia',
}

# Country code mapping for common codes
country_code_mapping = {
    'CN': 'CHN', 'IN': 'IND', 'RU': 'RUS', 'SA': 'SAU', 'US': 'USA',
    'EG': 'EGY', 'PK': 'PAK', 'BY': 'BLR', 'KR': 'KOR', 'GB': 'GBR',
    'KZ': 'KAZ', 'UA': 'UKR', 'FR': 'FRA', 'TR': 'TUR', 'JP': 'JPN',
    'DE': 'DEU', 'BR': 'BRA', 'IR': 'IRN', 'AU': 'AUS', 'ES': 'ESP',
    'MY': 'MYS', 'IT': 'ITA', 'CA': 'CAN', 'MX': 'MEX', 'NL': 'NLD',
    'SE': 'SWE', 'NO': 'NOR', 'FI': 'FIN', 'PL': 'POL', 'CZ': 'CZE',
    'PT': 'PRT', 'BN': 'BRN', 'DK': 'DNK', 'CH': 'CHE', 'AT': 'AUT',
    'GR': 'GRC', 'IL': 'ISR', 'SG': 'SGP', 'TH': 'THA', 'VN': 'VNM',
    'PH': 'PHL', 'ID': 'IDN', 'BD': 'BGD', 'NG': 'NGA', 'ZA': 'ZAF',
    'CL': 'CHL', 'AR': 'ARG', 'CO': 'COL', 'PE': 'PER', 'VE': 'VEN',
    'IE': 'IRL', 'NZ': 'NZL', 'BE': 'BEL', 'RO': 'ROU', 'HU': 'HUN',
    'BG': 'BGR', 'RS': 'SRB', 'HR': 'HRV', 'SK': 'SVK', 'SI': 'SVN',
    'LT': 'LTU', 'LV': 'LVA', 'EE': 'EST'
}

# Predefined color palettes
PREDEFINED_PALETTES = {
    'Blue Gradient': ['#E3F2FD', '#2196F3', '#0D47A1'],
    'Red Gradient': ['#FFEBEE', '#FF5252', '#D50000'],
    'Green Gradient': ['#E8F5E9', '#4CAF50', '#1B5E20'],
    'Purple Gradient': ['#F3E5F5', '#9C27B0', '#4A148C'],
    'Orange Gradient': ['#FFF3E0', '#FF9800', '#E65100'],
    'Viridis': ['#440154', '#21918C', '#FDE725'],
    'Plasma': ['#0D0887', '#CC4678', '#F0F921'],
    'Inferno': ['#000004', '#BC3754', '#FCA50A'],
    'Magma': ['#000004', '#B53679', '#FCFFA4'],
    'Cividis': ['#00204D', '#5D8FAA', '#FFEA46'],
}

# Enhanced fill styles with better visual effects
FILL_STYLES = {
    'Matte (Flat)': {'type': 'matte', 'gradient': False, 'edge_effect': 'none', 'texture': 'none'},
    'Glossy (3D Effect)': {'type': 'glossy', 'gradient': True, 'edge_effect': 'soft', 'texture': 'shine', 'height': 0.3},
    'Neon Glow': {'type': 'neon', 'gradient': True, 'edge_effect': 'glow', 'texture': 'glow', 'glow_strength': 0.8},
    'Topographic': {'type': 'topographic', 'gradient': True, 'edge_effect': 'mountain', 'texture': 'heightmap'},
    'Metallic Shine': {'type': 'metallic', 'gradient': True, 'edge_effect': 'sharp', 'texture': 'metal', 'reflection': 0.6},
    'Watercolor Wash': {'type': 'watercolor', 'gradient': True, 'edge_effect': 'blurry', 'texture': 'paper', 'blend': 0.7},
    'Heatmap Glow': {'type': 'heatmap', 'gradient': True, 'edge_effect': 'radiation', 'texture': 'gradient'},
}

# Line styles for chord connections
LINE_STYLES = {
    'Solid': '-',
    'Dashed': '--',
    'Dotted': ':',
    'Dash-dot': '-.',
}

# Load world map data function
@st.cache_data
def load_world_map_data():
    """Load world map data with fallback options"""
    try:
        # Try to download Natural Earth data
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        response = requests.get(url, stream=True, timeout=30)
        
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                tmp_file.flush()
                
                with zipfile.ZipFile(tmp_file.name, 'r') as zip_ref:
                    extract_dir = tempfile.mkdtemp()
                    zip_ref.extractall(extract_dir)
                    
                    # Find the shapefile
                    for file in os.listdir(extract_dir):
                        if file.endswith('.shp'):
                            world = gpd.read_file(os.path.join(extract_dir, file))
                            # Clean up
                            try:
                                os.unlink(tmp_file.name)
                            except:
                                pass
                            return world
            
            try:
                os.unlink(tmp_file.name)
            except:
                pass
        
        # Fallback to internal geopandas dataset
        st.info("Using backup data source (Natural Earth low resolution)")
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            return world
        except:
            # Alternative online source
            world_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
            world = gpd.read_file(world_url)
            return world
        
    except Exception as e:
        st.error(f"Error loading map data: {e}")
        # Create minimal world data as last resort
        st.info("Creating minimal world dataset...")
        return None

def normalize_country_name(country_input):
    """Normalize country name to standard format"""
    if not country_input or not isinstance(country_input, str):
        return country_input
    
    # Remove extra whitespace
    country_input = country_input.strip()
    
    # Check if it's already in our mapping
    if country_input in COUNTRY_NAME_MAPPING:
        return COUNTRY_NAME_MAPPING[country_input]
    
    # Try case-insensitive matching
    country_input_lower = country_input.lower()
    for key, value in COUNTRY_NAME_MAPPING.items():
        if country_input_lower == key.lower():
            return value
    
    # Try partial matching
    for key, value in COUNTRY_NAME_MAPPING.items():
        if (country_input_lower in key.lower() or 
            key.lower() in country_input_lower):
            return value
    
    # Return original if no match found
    return country_input

def parse_input_data(input_text):
    """Parse input data from text area with support for full country names"""
    country_data = {}
    connection_data = []
    
    lines = input_text.strip().split('\n')
    for line in lines:
        if line.strip():
            # Split by tab or multiple spaces
            parts = re.split(r'\t|\s{2,}', line.strip())
            parts = [p.strip() for p in parts if p.strip()]
            
            if len(parts) >= 2:
                country_part = parts[0]
                try:
                    frequency = float(parts[1])
                    
                    # Check if it's a connection (contains ;)
                    if ';' in country_part:
                        raw_countries = [c.strip() for c in country_part.split(';')]
                        # Normalize country names
                        countries = []
                        for c in raw_countries:
                            norm_name = normalize_country_name(c)
                            countries.append(norm_name)
                        
                        connection_data.append({
                            'countries': countries,
                            'frequency': frequency,
                            'raw_names': raw_countries
                        })
                    else:
                        # Single country frequency
                        country_name = normalize_country_name(country_part)
                        if country_name:
                            country_data[country_name] = {
                                'frequency': frequency,
                                'raw_name': country_part
                            }
                        
                except ValueError:
                    # Skip lines that can't be parsed
                    continue
                except Exception as e:
                    continue
    
    return country_data, connection_data

def create_custom_colormap(color1, color2, color3=None, n_points=256):
    """Create custom colormap from colors"""
    if color3 is None:
        colors = [color1, color2]
    else:
        colors = [color1, color3, color2]
    return LinearSegmentedColormap.from_list('custom', colors, N=n_points)

def create_enhanced_colormap(color1, color2, color3=None, style='default'):
    """Create enhanced colormap based on fill style"""
    if color3 is None:
        colors = [color1, color2]
    else:
        colors = [color1, color3, color2]
    
    # Adjust colors based on fill style for better visual effect
    color_objects = [Color(c) for c in colors]
    
    if style == 'Glossy (3D Effect)':
        # Make colors more vibrant for glossy effect
        enhanced_colors = []
        for col in color_objects:
            col.saturation = min(1.0, col.saturation * 1.3)
            col.luminance = min(0.9, col.luminance * 1.1)
            enhanced_colors.append(col.hex_l)
        colors = enhanced_colors
    
    elif style == 'Neon Glow':
        # Increase saturation and brightness for neon effect
        enhanced_colors = []
        for col in color_objects:
            col.saturation = min(1.0, col.saturation * 1.5)
            col.luminance = min(0.95, col.luminance * 1.3)
            enhanced_colors.append(col.hex_l)
        colors = enhanced_colors
    
    elif style == 'Metallic Shine':
        # Add metallic sheen - desaturate and increase luminance
        enhanced_colors = []
        for col in color_objects:
            col.saturation = max(0.2, col.saturation * 0.7)
            col.luminance = min(0.85, col.luminance + 0.2)
            enhanced_colors.append(col.hex_l)
        colors = enhanced_colors
    
    elif style == 'Heatmap Glow':
        # Bright heatmap colors
        enhanced_colors = []
        for col in color_objects:
            col.luminance = min(0.95, col.luminance * 1.4)
            enhanced_colors.append(col.hex_l)
        colors = enhanced_colors
    
    return LinearSegmentedColormap.from_list('enhanced', colors, N=512)

def apply_glossy_effect(ax, geometry, base_color):
    """Apply glossy 3D effect to country"""
    try:
        # Convert color
        base_color_obj = Color(base_color)
        
        # Main country fill with gradient
        gdf = gpd.GeoDataFrame([{'geometry': geometry}])
        gdf.plot(ax=ax, color=base_color, edgecolor='white', linewidth=0.8, alpha=0.95)
        
        # Create highlight (top part)
        highlight_color = Color(base_color)
        highlight_color.luminance = min(0.95, highlight_color.luminance + 0.4)
        
        try:
            # Create smaller geometry for highlight
            highlight_geom = geometry.buffer(-0.15)
            if not highlight_geom.is_empty and highlight_geom.geom_type in ['Polygon', 'MultiPolygon']:
                highlight_gdf = gpd.GeoDataFrame([{'geometry': highlight_geom}])
                highlight_gdf.plot(ax=ax, color=str(highlight_color), alpha=0.5, 
                                 edgecolor='none', linewidth=0)
        except:
            pass
        
        # Add subtle shadow
        try:
            shadow_geom = geometry.buffer(0.02)
            shadow_gdf = gpd.GeoDataFrame([{'geometry': shadow_geom}])
            shadow_gdf.plot(ax=ax, color='black', alpha=0.1, 
                          edgecolor='none', linewidth=0)
        except:
            pass
        
    except Exception as e:
        # Fallback
        gdf = gpd.GeoDataFrame([{'geometry': geometry}])
        gdf.plot(ax=ax, color=base_color, edgecolor='white', linewidth=0.8)

def apply_neon_glow_effect(ax, geometry, base_color):
    """Apply neon glow effect to country"""
    try:
        # Base color
        base_color_obj = Color(base_color)
        
        # Outer glow (larger buffer)
        glow_color = Color(base_color)
        glow_color.saturation = min(1.0, glow_color.saturation * 0.8)
        glow_color.luminance = min(1.0, glow_color.luminance + 0.3)
        
        # Multiple glow layers for intensity
        for i, alpha in enumerate([0.3, 0.2, 0.1]):
            try:
                glow_size = 0.05 + (i * 0.02)
                glow_geom = geometry.buffer(glow_size)
                if not glow_geom.is_empty:
                    glow_gdf = gpd.GeoDataFrame([{'geometry': glow_geom}])
                    glow_gdf.plot(ax=ax, color=str(glow_color), alpha=alpha,
                                edgecolor='none', linewidth=0)
            except:
                pass
        
        # Main country
        gdf = gpd.GeoDataFrame([{'geometry': geometry}])
        gdf.plot(ax=ax, color=base_color, edgecolor='white', 
                linewidth=1.5, alpha=0.9)
        
        # Inner glow
        inner_glow_color = Color(base_color)
        inner_glow_color.luminance = min(1.0, inner_glow_color.luminance + 0.5)
        try:
            inner_geom = geometry.buffer(-0.03)
            if not inner_geom.is_empty:
                inner_gdf = gpd.GeoDataFrame([{'geometry': inner_geom}])
                inner_gdf.plot(ax=ax, color=str(inner_glow_color), alpha=0.3,
                             edgecolor='none', linewidth=0)
        except:
            pass
        
    except Exception as e:
        gdf = gpd.GeoDataFrame([{'geometry': geometry}])
        gdf.plot(ax=ax, color=base_color, edgecolor='white', linewidth=1.5)

def apply_topographic_effect(ax, geometry, base_color):
    """Apply topographic/heightmap effect"""
    try:
        # Create height variation using multiple layers
        base_color_obj = Color(base_color)
        
        # Dark base layer
        dark_color = Color(base_color)
        dark_color.luminance = max(0.2, dark_color.luminance - 0.3)
        
        gdf = gpd.GeoDataFrame([{'geometry': geometry}])
        gdf.plot(ax=ax, color=str(dark_color), edgecolor='white', 
                linewidth=0.5, alpha=0.8)
        
        # Multiple elevation layers
        for i in range(3):
            try:
                layer_size = -0.08 * (i + 1)
                layer_geom = geometry.buffer(layer_size)
                if not layer_geom.is_empty:
                    # Calculate color for this layer
                    layer_color = Color(base_color)
                    elevation_factor = 0.2 * (i + 1)
                    layer_color.luminance = min(0.9, base_color_obj.luminance + elevation_factor)
                    
                    layer_gdf = gpd.GeoDataFrame([{'geometry': layer_geom}])
                    layer_gdf.plot(ax=ax, color=str(layer_color), alpha=0.6,
                                 edgecolor='none', linewidth=0)
            except:
                pass
        
        # Highlight ridges
        try:
            ridge_geom = geometry.buffer(-0.02)
            if not ridge_geom.is_empty:
                ridge_color = Color(base_color)
                ridge_color.luminance = min(1.0, ridge_color.luminance + 0.4)
                ridge_gdf = gpd.GeoDataFrame([{'geometry': ridge_geom}])
                ridge_gdf.plot(ax=ax, color=str(ridge_color), alpha=0.4,
                             edgecolor='none', linewidth=0)
        except:
            pass
        
    except Exception as e:
        gdf = gpd.GeoDataFrame([{'geometry': geometry}])
        gdf.plot(ax=ax, color=base_color, edgecolor='white', linewidth=0.5)

def apply_metallic_effect(ax, geometry, base_color):
    """Apply metallic shine effect"""
    try:
        # Metallic base
        metallic_base = Color(base_color)
        metallic_base.saturation = max(0.3, metallic_base.saturation * 0.6)
        
        gdf = gpd.GeoDataFrame([{'geometry': geometry}])
        gdf.plot(ax=ax, color=str(metallic_base), edgecolor='#AAAAAA', 
                linewidth=1.0, alpha=0.9)
        
        # Shine streaks (simulating light reflection)
        shine_color = Color(base_color)
        shine_color.luminance = min(1.0, shine_color.luminance + 0.5)
        shine_color.saturation = max(0.4, shine_color.saturation * 0.8)
        
        # Create diagonal shine patterns
        try:
            bounds = geometry.bounds
            if len(bounds) == 4:
                # Create gradient shine from top-left to bottom-right
                for i in range(2):
                    shine_geom = geometry.buffer(-0.1 - (i * 0.05))
                    if not shine_geom.is_empty:
                        shine_gdf = gpd.GeoDataFrame([{'geometry': shine_geom}])
                        shine_gdf.plot(ax=ax, color=str(shine_color), 
                                     alpha=0.25 - (i * 0.1), edgecolor='none', linewidth=0)
        except:
            pass
        
        # Edge highlights
        try:
            edge_geom = geometry.buffer(-0.01)
            if not edge_geom.is_empty:
                edge_color = Color(base_color)
                edge_color.luminance = min(1.0, edge_color.luminance + 0.6)
                edge_gdf = gpd.GeoDataFrame([{'geometry': edge_geom}])
                edge_gdf.plot(ax=ax, color=str(edge_color), alpha=0.3,
                            edgecolor='none', linewidth=0)
        except:
            pass
        
    except Exception as e:
        gdf = gpd.GeoDataFrame([{'geometry': geometry}])
        gdf.plot(ax=ax, color=base_color, edgecolor='#AAAAAA', linewidth=1.0)

def apply_watercolor_effect(ax, geometry, base_color):
    """Apply watercolor wash effect"""
    try:
        # Soft watercolor base
        watercolor_base = Color(base_color)
        watercolor_base.saturation = max(0.3, watercolor_base.saturation * 0.7)
        watercolor_base.luminance = min(0.95, watercolor_base.luminance + 0.2)
        
        # Main wash
        gdf = gpd.GeoDataFrame([{'geometry': geometry}])
        gdf.plot(ax=ax, color=str(watercolor_base), edgecolor='white', 
                linewidth=0.3, alpha=0.65)
        
        # Texture layers (simulating watercolor paper)
        for i in range(3):
            try:
                texture_size = -0.06 * (i + 1)
                texture_geom = geometry.buffer(texture_size)
                if not texture_geom.is_empty:
                    # Vary color slightly for each layer
                    texture_color = Color(base_color)
                    if i % 2 == 0:
                        texture_color.luminance = min(0.9, texture_color.luminance + 0.1)
                    else:
                        texture_color.luminance = max(0.3, texture_color.luminance - 0.1)
                    
                    texture_gdf = gpd.GeoDataFrame([{'geometry': texture_geom}])
                    alpha = 0.15 if i % 2 == 0 else 0.1
                    texture_gdf.plot(ax=ax, color=str(texture_color), alpha=alpha,
                                   edgecolor='none', linewidth=0)
            except:
                pass
        
        # Edge bleeding effect (characteristic of watercolor)
        try:
            bleed_geom = geometry.buffer(0.015)
            bleed_color = Color(base_color)
            bleed_color.luminance = min(0.98, bleed_color.luminance + 0.3)
            bleed_gdf = gpd.GeoDataFrame([{'geometry': bleed_geom}])
            bleed_gdf.plot(ax=ax, color=str(bleed_color), alpha=0.2,
                         edgecolor='none', linewidth=0)
        except:
            pass
        
    except Exception as e:
        gdf = gpd.GeoDataFrame([{'geometry': geometry}])
        gdf.plot(ax=ax, color=base_color, edgecolor='white', linewidth=0.3)

def apply_heatmap_effect(ax, geometry, base_color):
    """Apply heatmap glow effect"""
    try:
        # Bright center glow
        center_color = Color(base_color)
        center_color.luminance = min(0.98, center_color.luminance + 0.4)
        
        # Outer glow (cooler)
        outer_color = Color(base_color)
        outer_color.luminance = max(0.4, outer_color.luminance - 0.3)
        
        # Create radial gradient effect
        try:
            # Outer layer
            outer_geom = geometry.buffer(0.04)
            if not outer_geom.is_empty:
                outer_gdf = gpd.GeoDataFrame([{'geometry': outer_geom}])
                outer_gdf.plot(ax=ax, color=str(outer_color), alpha=0.3,
                             edgecolor='none', linewidth=0)
        except:
            pass
        
        # Middle layer
        try:
            middle_geom = geometry.buffer(0.02)
            if not middle_geom.is_empty:
                middle_color = Color(base_color)
                middle_gdf = gpd.GeoDataFrame([{'geometry': middle_geom}])
                middle_gdf.plot(ax=ax, color=base_color, alpha=0.5,
                              edgecolor='none', linewidth=0)
        except:
            pass
        
        # Center (hottest part)
        gdf = gpd.GeoDataFrame([{'geometry': geometry}])
        gdf.plot(ax=ax, color=str(center_color), edgecolor='white', 
                linewidth=0.8, alpha=0.9)
        
        # Pulsating effect dots
        try:
            dot_geom = geometry.buffer(-0.1)
            if not dot_geom.is_empty:
                dot_color = Color('white')
                dot_gdf = gpd.GeoDataFrame([{'geometry': dot_geom}])
                dot_gdf.plot(ax=ax, color=str(dot_color), alpha=0.4,
                           edgecolor='none', linewidth=0)
        except:
            pass
        
    except Exception as e:
        gdf = gpd.GeoDataFrame([{'geometry': geometry}])
        gdf.plot(ax=ax, color=base_color, edgecolor='white', linewidth=0.8)

def apply_fill_style_to_country(ax, geometry, base_color, style_name):
    """Apply enhanced fill style to a single country geometry"""
    try:
        if style_name == 'Matte (Flat)':
            gdf = gpd.GeoDataFrame([{'geometry': geometry}])
            gdf.plot(ax=ax, color=base_color, edgecolor='white', linewidth=0.8, alpha=0.95)
            
        elif style_name == 'Glossy (3D Effect)':
            apply_glossy_effect(ax, geometry, base_color)
            
        elif style_name == 'Neon Glow':
            apply_neon_glow_effect(ax, geometry, base_color)
            
        elif style_name == 'Topographic':
            apply_topographic_effect(ax, geometry, base_color)
            
        elif style_name == 'Metallic Shine':
            apply_metallic_effect(ax, geometry, base_color)
            
        elif style_name == 'Watercolor Wash':
            apply_watercolor_effect(ax, geometry, base_color)
            
        elif style_name == 'Heatmap Glow':
            apply_heatmap_effect(ax, geometry, base_color)
            
        else:
            # Default fallback
            gdf = gpd.GeoDataFrame([{'geometry': geometry}])
            gdf.plot(ax=ax, color=base_color, edgecolor='white', linewidth=0.8)
            
    except Exception as e:
        # Ultimate fallback
        try:
            gdf = gpd.GeoDataFrame([{'geometry': geometry}])
            gdf.plot(ax=ax, color=base_color, edgecolor='white', linewidth=0.8)
        except:
            pass

def get_country_centroid(geometry):
    """Get centroid of a country geometry with fallback"""
    try:
        centroid = geometry.centroid
        if centroid.is_empty:
            centroid = geometry.representative_point()
        return centroid
    except:
        try:
            return geometry.representative_point()
        except:
            # Return approximate center
            bounds = geometry.bounds
            return Point((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)

def find_country_in_world(country_name, world):
    """Find country in world data by name with fuzzy matching"""
    # Normalize the country name first
    normalized_name = normalize_country_name(country_name)
    
    # Try exact match in various name columns
    name_columns = ['NAME', 'NAME_LONG', 'ADMIN', 'SOVEREIGNT', 'FORMAL_EN', 'ISO_A3', 'ISO_A2']
    
    for col in name_columns:
        if col in world.columns:
            # Try exact match (case insensitive)
            mask = world[col].astype(str).str.contains(f'^{re.escape(normalized_name)}$', case=False, na=False, regex=True)
            matches = world[mask]
            if not matches.empty:
                return matches.iloc[0]
    
    # Try partial matching
    for col in name_columns:
        if col in world.columns:
            # Try contains match
            mask = world[col].astype(str).str.contains(re.escape(normalized_name), case=False, na=False, regex=True)
            matches = world[mask]
            if not matches.empty:
                return matches.iloc[0]
    
    # Try matching with normalized names from our mapping
    if normalized_name != country_name:
        # Try again with the normalized name
        for col in name_columns:
            if col in world.columns:
                mask = world[col].astype(str).str.contains(re.escape(normalized_name), case=False, na=False, regex=True)
                matches = world[mask]
                if not matches.empty:
                    return matches.iloc[0]
    
    return None

def map_country_names(country_data, connection_data, world):
    """Map country names to world data and create connection mappings"""
    mapped_data = {}
    unmapped_names = []
    name_to_iso3 = {}
    
    # Map single country frequencies
    for country_name, data in country_data.items():
        country_row = find_country_in_world(country_name, world)
        if country_row is not None and 'ISO_A3' in country_row:
            iso_code = str(country_row['ISO_A3'])
            mapped_data[iso_code] = {
                'frequency': data['frequency'],
                'name': country_name,
                'raw_name': data['raw_name']
            }
            name_to_iso3[country_name] = iso_code
        else:
            unmapped_names.append(country_name)
    
    # Map connections
    mapped_connections = []
    for conn in connection_data:
        countries = conn['countries']
        freq = conn['frequency']
        raw_names = conn['raw_names']
        
        mapped_countries = []
        iso_codes = []
        all_mapped = True
        
        for country, raw_name in zip(countries, raw_names):
            if country in name_to_iso3:
                iso_codes.append(name_to_iso3[country])
                mapped_countries.append(country)
            else:
                country_row = find_country_in_world(country, world)
                if country_row is not None and 'ISO_A3' in country_row:
                    iso_code = str(country_row['ISO_A3'])
                    iso_codes.append(iso_code)
                    mapped_countries.append(country)
                    name_to_iso3[country] = iso_code
                else:
                    all_mapped = False
                    if country not in unmapped_names:
                        unmapped_names.append(country)
        
        if all_mapped and len(iso_codes) >= 2:
            # Create pairwise connections for multiple countries
            if len(iso_codes) == 2:
                mapped_connections.append({
                    'iso_codes': iso_codes,
                    'countries': mapped_countries,
                    'raw_names': raw_names,
                    'frequency': freq
                })
            else:
                # For 3+ countries, create connections between each pair
                for i in range(len(iso_codes)):
                    for j in range(i+1, len(iso_codes)):
                        mapped_connections.append({
                            'iso_codes': [iso_codes[i], iso_codes[j]],
                            'countries': [mapped_countries[i], mapped_countries[j]],
                            'raw_names': [raw_names[i], raw_names[j]],
                            'frequency': freq
                        })
    
    return mapped_data, mapped_connections, unmapped_names, name_to_iso3

def calculate_bezier_curve(p1, p2, height_factor=0.3):
    """
    Calculate cubic Bezier curve points between two points
    p1, p2: Points as (x, y) tuples
    height_factor: Controls the curve height (0 = straight line)
    """
    # Calculate control points for a natural curve
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    # Distance between points
    dist = np.sqrt(dx**2 + dy**2)
    
    # Normalized direction
    if dist > 0:
        nx, ny = dx/dist, dy/dist
    else:
        nx, ny = 0, 0
    
    # Midpoint
    mid_x = (p1[0] + p2[0]) / 2
    mid_y = (p1[1] + p2[1]) / 2
    
    # Perpendicular direction for curve
    perp_x = -ny
    perp_y = nx
    
    # Control point at midpoint with perpendicular offset
    # Height increases with distance
    curve_height = dist * height_factor
    control_x = mid_x + perp_x * curve_height
    control_y = mid_y + perp_y * curve_height
    
    # Two control points for smoother curve
    control1_x = p1[0] + (control_x - p1[0]) * 0.5
    control1_y = p1[1] + (control_y - p1[1]) * 0.5
    
    control2_x = p2[0] + (control_x - p2[0]) * 0.5
    control2_y = p2[1] + (control_y - p2[1]) * 0.5
    
    # Generate curve points
    t = np.linspace(0, 1, 100)
    curve_x = (1-t)**3 * p1[0] + 3*(1-t)**2*t * control1_x + 3*(1-t)*t**2 * control2_x + t**3 * p2[0]
    curve_y = (1-t)**3 * p1[1] + 3*(1-t)**2*t * control1_y + 3*(1-t)*t**2 * control2_y + t**3 * p2[1]
    
    return curve_x, curve_y, (control_x, control_y)

def draw_connection_line(ax, p1, p2, frequency, max_frequency, min_frequency, 
                         color1, color2, line_style, show_arrows=True):
    """Draw a curved connection line between two points"""
    
    # Normalize line width
    min_line_width = st.session_state.min_line_width
    max_line_width = st.session_state.max_line_width
    
    if max_frequency > min_frequency:
        normalized_freq = (frequency - min_frequency) / (max_frequency - min_frequency)
        line_width = min_line_width + normalized_freq * (max_line_width - min_line_width)
    else:
        line_width = (min_line_width + max_line_width) / 2
    
    # Calculate curve
    curve_x, curve_y, control_point = calculate_bezier_curve(
        p1, p2, height_factor=st.session_state.curve_height
    )
    
    # Choose color based on settings
    if st.session_state.line_color_type == 'Single Color':
        line_color = st.session_state.line_color
    elif st.session_state.line_color_type == 'By Frequency':
        # Use colormap based on frequency
        norm_freq = (frequency - min_frequency) / max(1, (max_frequency - min_frequency))
        cmap = plt.cm.get_cmap(st.session_state.line_colormap)
        line_color = cmap(norm_freq)
    else:  # 'Gradient (Source to Destination)'
        line_color = color1  # For simplicity, use source color
    
    # Set alpha based on frequency
    min_alpha = 0.3
    max_alpha = 0.9
    if max_frequency > min_frequency:
        alpha = min_alpha + (frequency - min_frequency) / (max_frequency - min_frequency) * (max_alpha - min_alpha)
    else:
        alpha = (min_alpha + max_alpha) / 2
    
    # Draw the curve
    ax.plot(curve_x, curve_y, 
            linewidth=line_width,
            color=line_color,
            alpha=alpha,
            linestyle=line_style,
            solid_capstyle='round')
    
    # Add arrow if enabled
    if show_arrows and st.session_state.show_arrows:
        # Place arrow at 70% along the curve
        idx = int(len(curve_x) * 0.7)
        if idx < len(curve_x) - 1:
            dx = curve_x[idx+1] - curve_x[idx]
            dy = curve_y[idx+1] - curve_y[idx]
            
            arrow_length = line_width * 2
            ax.arrow(curve_x[idx], curve_y[idx], 
                    dx * 0.1, dy * 0.1,
                    head_width=arrow_length,
                    head_length=arrow_length,
                    fc=line_color, ec=line_color,
                    alpha=alpha*0.8)
    
    return line_width

def draw_chord_lines(ax, world, connections, mapped_data, country_colors):
    """Draw chord lines between countries"""
    if not connections:
        return
    
    # Get all frequencies for normalization
    connection_frequencies = [conn['frequency'] for conn in connections]
    if not connection_frequencies:
        return
    
    min_freq = min(connection_frequencies)
    max_freq = max(connection_frequencies)
    
    # Get country centroids
    country_centroids = {}
    for conn in connections:
        for iso_code in conn['iso_codes']:
            if iso_code not in country_centroids:
                country = world[world['ISO_A3'] == iso_code]
                if not country.empty:
                    centroid = get_country_centroid(country.geometry.iloc[0])
                    country_centroids[iso_code] = (centroid.x, centroid.y)
    
    # Sort connections by frequency (draw smaller ones first)
    sorted_connections = sorted(connections, key=lambda x: x['frequency'])
    
    # Draw each connection
    for conn in sorted_connections:
        iso_codes = conn['iso_codes']
        freq = conn['frequency']
        
        if len(iso_codes) >= 2:
            # Get coordinates for all countries
            coords = []
            valid = True
            for iso_code in iso_codes[:2]:  # Only use first 2 for line
                if iso_code in country_centroids:
                    coords.append(country_centroids[iso_code])
                else:
                    valid = False
                    break
            
            if valid and len(coords) == 2:
                # Get colors for these countries
                color1 = country_colors.get(iso_codes[0], '#888888')
                color2 = country_colors.get(iso_codes[1], '#888888')
                
                # Draw the line
                line_width = draw_connection_line(
                    ax, coords[0], coords[1], freq,
                    max_freq, min_freq,
                    color1, color2,
                    LINE_STYLES[st.session_state.line_style],
                    show_arrows=True
                )
                
                # Add label if enabled
                if st.session_state.show_line_labels and freq >= st.session_state.min_label_freq:
                    # Place label near the curve peak
                    curve_x, curve_y, _ = calculate_bezier_curve(
                        coords[0], coords[1], 
                        height_factor=st.session_state.curve_height
                    )
                    
                    # Find highest point of the curve
                    if len(curve_y) > 0:
                        max_idx = np.argmax(curve_y)
                        label_x, label_y = curve_x[max_idx], curve_y[max_idx]
                        
                        label_text = f"{freq:.0f}"
                        if st.session_state.line_label_type == 'Full':
                            # Use country names for the label
                            country_names = conn['countries'][:2] if len(conn['countries']) >= 2 else conn['iso_codes'][:2]
                            label_text = f"{country_names[0]}-{country_names[1]}: {freq:.0f}"
                        
                        ax.annotate(
                            label_text,
                            xy=(label_x, label_y),
                            xytext=(0, 0),
                            textcoords="offset points",
                            ha='center',
                            va='center',
                            fontsize=st.session_state.line_font_size,
                            color='black',
                            weight='bold',
                            fontfamily='sans-serif',
                            path_effects=[withStroke(linewidth=2, foreground='white')],
                            bbox=dict(
                                boxstyle="round,pad=0.3", 
                                facecolor='white', 
                                alpha=0.8, 
                                edgecolor='gray',
                                linewidth=0.5
                            )
                        )

def display_statistics(country_data, connection_data, mapped_data, mapped_connections, unmapped_names):
    """Display statistics panel"""
    st.subheader("📊 Data Statistics")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric("Countries in Input", len(country_data))
    
    with col_stat2:
        matched = len(mapped_data)
        st.metric("Countries Matched", f"{matched}/{len(country_data)}")
    
    with col_stat3:
        if connection_data:
            st.metric("Total Connections", len(connection_data))
        else:
            st.metric("Total Connections", "0")
    
    with col_stat4:
        if mapped_data:
            values = [data['frequency'] for data in mapped_data.values()]
            if values:
                avg = sum(values) / len(values)
                st.metric("Avg Country Freq", f"{avg:.2f}")
            else:
                st.metric("Avg Country Freq", "N/A")
        else:
            st.metric("Avg Country Freq", "N/A")
    
    # Connection statistics
    if mapped_connections:
        st.subheader("🔗 Connection Statistics")
        conn_col1, conn_col2, conn_col3 = st.columns(3)
        
        with conn_col1:
            conn_freqs = [c['frequency'] for c in mapped_connections]
            st.metric("Total Connection Freq", f"{sum(conn_freqs):.0f}")
        
        with conn_col2:
            st.metric("Avg Connection Freq", f"{np.mean(conn_freqs):.2f}")
        
        with conn_col3:
            st.metric("Max Connection Freq", f"{max(conn_freqs):.0f}")
        
        # Top connections table
        st.subheader("🏆 Top Connections")
        top_conns = sorted(mapped_connections, key=lambda x: x['frequency'], reverse=True)[:10]
        conn_df = pd.DataFrame([{
            'From': c['countries'][0] if len(c['countries']) > 0 else c['iso_codes'][0],
            'To': c['countries'][1] if len(c['countries']) > 1 else c['iso_codes'][1] if len(c['iso_codes']) > 1 else 'Multiple',
            'Frequency': c['frequency']
        } for c in top_conns])
        
        st.dataframe(conn_df, use_container_width=True)
    
    # Top countries table
    st.subheader("🏆 Top Countries")
    if country_data:
        # Create list of countries with their frequencies
        country_list = []
        for country_name, data in country_data.items():
            country_list.append({
                'Country': data['raw_name'],  # Use raw name for display
                'Normalized Name': country_name,  # Show normalized name
                'Frequency': data['frequency']
            })
        
        top_df = pd.DataFrame(country_list).sort_values('Frequency', ascending=False).head(10)
        
        st.dataframe(top_df, use_container_width=True, 
                    column_config={
                        "Country": st.column_config.TextColumn("Input Name"),
                        "Normalized Name": st.column_config.TextColumn("Matched Name"),
                        "Frequency": st.column_config.NumberColumn(
                            "Frequency",
                            format="%.0f"
                        )
                    })
    
    # Show unmatched names
    if unmapped_names:
        st.warning(f"⚠️ {len(unmapped_names)} country names could not be matched: {', '.join(sorted(set(unmapped_names))[:10])}")
        if len(unmapped_names) > 10:
            st.info(f"... and {len(unmapped_names) - 10} more unmatched names")
        
        # Show suggestions for unmatched names
        st.info("💡 **Suggestions**: Try using full country names or ISO codes (US, GB, RU, etc.)")
    
def generate_map():
    """Generate the world frequency map"""
    # Parse input data
    country_data, connection_data = parse_input_data(st.session_state.data_input)
    
    if not country_data and not connection_data:
        st.error("No valid data to display. Please enter country names and frequencies.")
        return None
    
    # Load world map data
    with st.spinner("Loading world map data..."):
        world = load_world_map_data()
        if world is None:
            st.error("Failed to load world map data.")
            return None
    
    # Map country names to world data
    mapped_data, mapped_connections, unmapped_names, name_to_iso3 = map_country_names(
        country_data, connection_data, world
    )
    
    # Create figure with high DPI
    fig, ax = plt.subplots(1, 1, figsize=(20, 10), dpi=150)
    
    # Set background
    ax.set_facecolor(st.session_state.background_color)
    
    # Plot all countries as background - WHITE with GRAY borders (default)
    world.plot(ax=ax, color='white', edgecolor=st.session_state.border_color,
              linewidth=0.5, alpha=1.0)
    
    # Get valid data
    if mapped_data:
        # Create a DataFrame from mapped data
        mapped_df_data = []
        for iso_code, data in mapped_data.items():
            mapped_df_data.append({
                'ISO_A3': iso_code,
                'frequency': data['frequency'],
                'name': data['name'],
                'raw_name': data['raw_name']
            })
        
        if mapped_df_data:
            mapped_df = pd.DataFrame(mapped_df_data)
            valid_data = world.merge(
                mapped_df,
                left_on='ISO_A3', right_on='ISO_A3', how='inner'
            )
        else:
            valid_data = pd.DataFrame()
    else:
        valid_data = pd.DataFrame()
    
    if not valid_data.empty:
        # Get color values
        color1 = st.session_state.color1
        color2 = st.session_state.color2
        color3 = st.session_state.color3 if st.session_state.color_points == '3 Colors (Diverging)' else None
        
        # Create enhanced colormap based on fill style
        cmap = create_enhanced_colormap(color1, color2, color3, st.session_state.fill_style)
        
        # Normalize data
        min_val = valid_data['frequency'].min()
        max_val = valid_data['frequency'].max()
        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
        
        # Store country colors for chord lines
        country_colors = {}
        
        # Plot countries with data using selected fill style
        for idx, row in valid_data.iterrows():
            color = cmap(norm(row['frequency']))
            country_colors[str(row['ISO_A3'])] = color
            apply_fill_style_to_country(ax, row.geometry, color, st.session_state.fill_style)
        
        # Draw chord lines if enabled
        if st.session_state.show_chords and mapped_connections:
            draw_chord_lines(ax, world, mapped_connections, mapped_data, country_colors)
        
        # Add labels for top countries if enabled
        if st.session_state.show_labels and not valid_data.empty:
            # Sort by frequency and take top N
            top_countries = valid_data.nlargest(st.session_state.top_n_labels, 'frequency')
            
            for idx, row in top_countries.iterrows():
                try:
                    # Get centroid for label placement
                    centroid = get_country_centroid(row.geometry)
                    
                    # Skip if centroid is invalid
                    if centroid.is_empty:
                        continue
                        
                    # Create label text
                    if st.session_state.label_type == 'Code Only':
                        label = str(row['ISO_A3']) if pd.notna(row['ISO_A3']) else ''
                    elif st.session_state.label_type == 'Value Only':
                        label = f"{row['frequency']:.0f}"
                    elif st.session_state.label_type == 'Name Only':
                        # Use the name from our mapped data
                        label = row['name'] if 'name' in row and pd.notna(row['name']) else str(row['ISO_A3'])
                    else:  # Name + Value
                        name = row['name'] if 'name' in row and pd.notna(row['name']) else str(row['ISO_A3'])
                        label = f"{name}\n{row['frequency']:.0f}"
                    
                    if label:
                        # Add text with outline for better visibility
                        ax.annotate(
                            label,
                            xy=(centroid.x, centroid.y),
                            xytext=(0, 0),
                            textcoords="offset points",
                            ha='center',
                            va='center',
                            fontsize=st.session_state.font_size,
                            color='black',
                            weight='bold' if st.session_state.label_type in ['Name + Value', 'Name Only'] else 'normal',
                            fontfamily='sans-serif',
                            path_effects=[withStroke(linewidth=3, foreground='white')],
                            bbox=dict(
                                boxstyle="round,pad=0.5", 
                                facecolor='white', 
                                alpha=0.85, 
                                edgecolor='gray',
                                linewidth=0.5
                            )
                        )
                except Exception as e:
                    continue
        
        # Add colorbar with style matching the map - 2/3 width with ticks at both ends
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # Create axes for colorbar - positioned at bottom center
        # Get the current axes position
        ax_pos = ax.get_position()
        
        # Calculate position for colorbar (2/3 width, centered)
        width_ratio = 2.0/3.0  # 2/3 of map width
        left = ax_pos.x0 + (ax_pos.width * (1 - width_ratio)) / 2
        bottom = ax_pos.y0 - 0.08  # Position below the map
        width = ax_pos.width * width_ratio
        height = 0.02  # Height of colorbar
        
        # Create axes for colorbar
        cbar_ax = fig.add_axes([left, bottom, width, height])
        
        # Create colorbar with ticks at min and max values
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(st.session_state.scale_title, fontsize=14, weight='bold', labelpad=10)
        
        # Set ticks - ensure we have ticks at both ends
        ticks = [min_val, max_val]
        
        # Add intermediate ticks if there's enough range
        if max_val - min_val > 10:
            # Add 1-2 intermediate ticks
            mid1 = min_val + (max_val - min_val) * 0.33
            mid2 = min_val + (max_val - min_val) * 0.67
            ticks = [min_val, mid1, mid2, max_val]
        elif max_val - min_val > 5:
            # Add one intermediate tick
            mid = (min_val + max_val) / 2
            ticks = [min_val, mid, max_val]
        
        # Format tick labels
        tick_labels = []
        for tick in ticks:
            if tick >= 1000:
                tick_labels.append(f"{tick/1000:.1f}k")
            elif tick >= 100:
                tick_labels.append(f"{tick:.0f}")
            elif tick >= 10:
                tick_labels.append(f"{tick:.1f}")
            else:
                tick_labels.append(f"{tick:.2f}")
        
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=11)
        
        # Add minor ticks for better scale visualization
        cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        
        # Style the colorbar
        cbar.outline.set_linewidth(1)
        cbar.outline.set_edgecolor('black')
        
    else:
        # If no country data but have connections, still draw connections
        if st.session_state.show_chords and mapped_connections:
            country_colors = {}
            for conn in mapped_connections:
                for iso_code in conn['iso_codes']:
                    if iso_code not in country_colors:
                        country_colors[iso_code] = '#888888'
            draw_chord_lines(ax, world, mapped_connections, {}, country_colors)
    
    # Set title with enhanced styling
    title_text = st.session_state.map_title
    if st.session_state.show_chords and mapped_connections:
        title_text += " with Connections"
    ax.set_title(title_text, fontsize=22, pad=30, 
                weight='bold', fontfamily='sans-serif',
                color='#2C3E50')
    
    # Add subtle grid for orientation
    ax.grid(True, alpha=0.1, linestyle='--', linewidth=0.5)
    
    # Remove axes but keep grid
    ax.set_axis_off()
    
    # Add subtle shadow to the whole map
    fig.patch.set_alpha(0.9)
    
    # Adjust layout to accommodate colorbar
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space at bottom for colorbar
    
    return fig, country_data, connection_data, mapped_data, mapped_connections, unmapped_names

def reset_settings():
    """Reset all settings to defaults"""
    st.session_state.data_input = """United States\t7860
United Kingdom\t2350
Russia\t1081
Australia\t927
Poland\t905
Saudi Arabia\t328
Norway\t297
Finland\t297
Portugal\t288
Brunei\t102
US;UK\t150
RU;CN\t120
AU;NZ\t80
DE;FR\t75
JP;KR\t60"""
    st.session_state.map_title = "World Frequency Map"
    st.session_state.scale_title = "Frequency"
    st.session_state.palette_selection = "Blue Gradient"
    st.session_state.color_points = "2 Colors (Linear)"
    st.session_state.fill_style = "Glossy (3D Effect)"
    st.session_state.show_labels = True
    st.session_state.label_type = "Name + Value"
    st.session_state.top_n_labels = 10
    st.session_state.font_size = 9
    st.session_state.background_color = "#F5F5F5"
    st.session_state.border_color = "#CCCCCC"
    st.session_state.show_chords = True
    
    # Chord line settings
    st.session_state.min_line_width = 0.5
    st.session_state.max_line_width = 5.0
    st.session_state.curve_height = 0.3
    st.session_state.line_style = 'Solid'
    st.session_state.line_color_type = 'Single Color'
    st.session_state.line_color = '#2196F3'
    st.session_state.line_colormap = 'viridis'
    st.session_state.show_arrows = True
    st.session_state.show_line_labels = True
    st.session_state.line_label_type = 'Value Only'
    st.session_state.min_label_freq = 50
    st.session_state.line_font_size = 9
    
    # Update color pickers based on palette
    selected_colors = PREDEFINED_PALETTES[st.session_state.palette_selection]
    st.session_state.color1 = selected_colors[0]
    st.session_state.color2 = selected_colors[-1]
    if len(selected_colors) > 1:
        st.session_state.color3 = selected_colors[1]

# Initialize session state
if 'data_input' not in st.session_state:
    reset_settings()

# Main app layout
st.title("World Frequency Map with Full Country Names")
st.markdown("""
Create stunning world frequency maps with chord lines showing connections between countries. 
Supports both full country names and abbreviations. Examples: "United States", "USA", "US", "Russia", "Russian Federation", "RU"
""")

# Create columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Data Input")
    
    data_input = st.text_area(
        "Enter country data and connections:",
        value=st.session_state.data_input,
        height=300,
        key="data_input_widget",
        help="""Format examples:
        Single country: United States 7860
        Single country (abbreviation): US 7860
        Connection: USA;UK 150
        Multiple countries: Germany;France;Italy 50 (will create pairwise connections)
        
        Separate country and frequency with TAB or multiple spaces."""
    )
    st.session_state.data_input = data_input
    
    st.info("""
    💡 **Data Format Tips**:
    - Use TAB between country names and frequency
    - For connections, separate countries with semicolon (;)
    - Supports: Full names (United States), abbreviations (US, USA), common names (Russia)
    - Case insensitive
    """)
    
    # Map settings
    st.subheader("🗺️ Map Settings")
    
    map_title = st.text_input(
        "Map Title:",
        value=st.session_state.map_title,
        key="map_title_widget"
    )
    st.session_state.map_title = map_title
    
    scale_title = st.text_input(
        "Scale Title:",
        value=st.session_state.scale_title,
        key="scale_title_widget"
    )
    st.session_state.scale_title = scale_title

with col2:
    st.subheader("🎨 Color Settings")
    
    # Palette selection
    palette_selection = st.selectbox(
        "Preset Palette:",
        options=list(PREDEFINED_PALETTES.keys()),
        index=list(PREDEFINED_PALETTES.keys()).index(st.session_state.palette_selection),
        key="palette_selection_widget"
    )
    st.session_state.palette_selection = palette_selection
    
    # Color points
    color_points = st.radio(
        "Color Points:",
        options=['2 Colors (Linear)', '3 Colors (Diverging)'],
        index=0 if st.session_state.color_points == '2 Colors (Linear)' else 1,
        key="color_points_widget",
        horizontal=True
    )
    st.session_state.color_points = color_points
    
    # Color pickers
    selected_colors = PREDEFINED_PALETTES[palette_selection]
    
    if color_points == '2 Colors (Linear)':
        col_a, col_b = st.columns(2)
        with col_a:
            color1 = st.color_picker(
                "Low Color:",
                value=st.session_state.color1,
                key="color1_widget"
            )
            st.session_state.color1 = color1
        with col_b:
            color2 = st.color_picker(
                "High Color:",
                value=st.session_state.color2,
                key="color2_widget"
            )
            st.session_state.color2 = color2
    else:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            color1 = st.color_picker(
                "Low Color:",
                value=st.session_state.color1,
                key="color1_div_widget"
            )
            st.session_state.color1 = color1
        with col_b:
            color3 = st.color_picker(
                "Middle Color:",
                value=st.session_state.color3,
                key="color3_widget"
            )
            st.session_state.color3 = color3
        with col_c:
            color2 = st.color_picker(
                "High Color:",
                value=st.session_state.color2,
                key="color2_div_widget"
            )
            st.session_state.color2 = color2
    
    # Fill style for maps
    st.subheader("🖌️ Fill Style")
    fill_style = st.selectbox(
        "Choose fill style:",
        options=list(FILL_STYLES.keys()),
        index=list(FILL_STYLES.keys()).index(st.session_state.fill_style),
        key="fill_style_widget"
    )
    st.session_state.fill_style = fill_style

# Chord connections settings
with st.expander("🔗 Chord Line Settings", expanded=False):
    col_ch1, col_ch2 = st.columns(2)
    
    with col_ch1:
        st.subheader("📏 Line Appearance")
        
        show_chords = st.checkbox(
            "Show chord lines",
            value=st.session_state.show_chords,
            key="show_chords_widget"
        )
        st.session_state.show_chords = show_chords
        
        if show_chords:
            min_line_width = st.slider(
                "Min line width:",
                min_value=0.1,
                max_value=3.0,
                value=st.session_state.min_line_width,
                step=0.1,
                key="min_line_width_widget"
            )
            st.session_state.min_line_width = min_line_width
            
            max_line_width = st.slider(
                "Max line width:",
                min_value=1.0,
                max_value=10.0,
                value=st.session_state.max_line_width,
                step=0.1,
                key="max_line_width_widget"
            )
            st.session_state.max_line_width = max_line_width
            
            curve_height = st.slider(
                "Curve height:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.curve_height,
                step=0.05,
                key="curve_height_widget"
            )
            st.session_state.curve_height = curve_height
            
            line_style = st.selectbox(
                "Line style:",
                options=list(LINE_STYLES.keys()),
                index=list(LINE_STYLES.keys()).index(st.session_state.line_style),
                key="line_style_widget"
            )
            st.session_state.line_style = line_style
    
    with col_ch2:
        if show_chords:
            st.subheader("🎨 Line Colors")
            
            line_color_type = st.radio(
                "Line coloring:",
                options=['Single Color', 'By Frequency'],
                index=0 if st.session_state.line_color_type == 'Single Color' else 1,
                key="line_color_type_widget",
                horizontal=True
            )
            st.session_state.line_color_type = line_color_type
            
            if line_color_type == 'Single Color':
                line_color = st.color_picker(
                    "Line color:",
                    value=st.session_state.line_color,
                    key="line_color_widget"
                )
                st.session_state.line_color = line_color
            else:
                line_colormap = st.selectbox(
                    "Colormap:",
                    options=['viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdYlBu'],
                    index=['viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdYlBu'].index(
                        st.session_state.line_colormap),
                    key="line_colormap_widget"
                )
                st.session_state.line_colormap = line_colormap
            
            show_arrows = st.checkbox(
                "Show direction arrows",
                value=st.session_state.show_arrows,
                key="show_arrows_widget"
            )
            st.session_state.show_arrows = show_arrows
            
            st.subheader("🏷️ Line Labels")
            show_line_labels = st.checkbox(
                "Show frequency labels on lines",
                value=st.session_state.show_line_labels,
                key="show_line_labels_widget"
            )
            st.session_state.show_line_labels = show_line_labels
            
            if show_line_labels:
                line_label_type = st.selectbox(
                    "Label type:",
                    options=['Value Only', 'Full'],
                    index=0 if st.session_state.line_label_type == 'Value Only' else 1,
                    key="line_label_type_widget"
                )
                st.session_state.line_label_type = line_label_type
                
                min_label_freq = st.slider(
                    "Min frequency for labels:",
                    min_value=0,
                    max_value=500,
                    value=st.session_state.min_label_freq,
                    key="min_label_freq_widget"
                )
                st.session_state.min_label_freq = min_label_freq
                
                line_font_size = st.slider(
                    "Label font size:",
                    min_value=6,
                    max_value=16,
                    value=st.session_state.line_font_size,
                    key="line_font_size_widget"
                )
                st.session_state.line_font_size = line_font_size

# Advanced settings expander
with st.expander("⚙️ Advanced Settings", expanded=False):
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🌐 Background & Borders")
        
        background_color = st.color_picker(
            "Background Color:",
            value=st.session_state.background_color,
            key="background_color_widget"
        )
        st.session_state.background_color = background_color
        
        border_color = st.color_picker(
            "Border Color for all countries:",
            value=st.session_state.border_color,
            key="border_color_widget",
            help="Color for country borders. Default is light gray (#CCCCCC)"
        )
        st.session_state.border_color = border_color
    
    with col4:
        st.subheader("🏷️ Country Label Settings")
        
        show_labels = st.checkbox(
            "Show country labels on map",
            value=st.session_state.show_labels,
            key="show_labels_widget"
        )
        st.session_state.show_labels = show_labels
        
        if show_labels:
            label_type = st.selectbox(
                "Label Type:",
                options=['Code Only', 'Value Only', 'Name Only', 'Name + Value'],
                index=['Code Only', 'Value Only', 'Name Only', 'Name + Value'].index(st.session_state.label_type),
                key="label_type_widget"
            )
            st.session_state.label_type = label_type
            
            top_n_labels = st.slider(
                "Top N labels to show:",
                min_value=1,
                max_value=50,
                value=st.session_state.top_n_labels,
                key="top_n_labels_widget"
            )
            st.session_state.top_n_labels = top_n_labels
            
            font_size = st.slider(
                "Font size:",
                min_value=6,
                max_value=20,
                value=st.session_state.font_size,
                key="font_size_widget"
            )
            st.session_state.font_size = font_size

# Buttons
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn1:
    generate_btn = st.button("🚀 Generate Map", type="primary", use_container_width=True)
with col_btn2:
    reset_btn = st.button("🔄 Reset Settings", use_container_width=True)
with col_btn3:
    if st.button("📥 Download Sample Data", use_container_width=True):
        sample_data = """United States\t7860
United Kingdom\t2350
Russia\t1081
Australia\t927
Poland\t905
Saudi Arabia\t328
Norway\t297
Finland\t297
Portugal\t288
Brunei\t102
US;UK\t150
RU;CN\t120
AU;NZ\t80
DE;FR\t75
JP;KR\t60
FR;DE;IT\t45"""
        st.download_button(
            label="📥 Click to download sample",
            data=sample_data,
            file_name="sample_country_data_full_names.txt",
            mime="text/plain",
            use_container_width=True
        )

# Handle reset button
if reset_btn:
    reset_settings()
    st.rerun()

# Generate map when button is clicked
if generate_btn:
    with st.spinner("🚀 Generating map with chord connections..."):
        result = generate_map()
        
        if result:
            fig, country_data, connection_data, mapped_data, mapped_connections, unmapped_names = result
            
            # Display the map
            st.subheader("🗺️ World Map with Chord Connections")
            st.pyplot(fig)
            
            # Display statistics
            display_statistics(country_data, connection_data, mapped_data, mapped_connections, unmapped_names)
            
            # Map download options
            st.subheader("💾 Export Options")
            
            # Convert figure to bytes for download
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', 
                      facecolor=st.session_state.background_color)
            buf.seek(0)
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                st.download_button(
                    label="📥 Download as High-Quality PNG",
                    data=buf,
                    file_name=f"world_map_with_chords.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col_dl2:
                # Save as SVG
                buf_svg = io.BytesIO()
                fig.savefig(buf_svg, format='svg', bbox_inches='tight',
                          facecolor=st.session_state.background_color)
                buf_svg.seek(0)
                
                st.download_button(
                    label="📥 Download as Scalable SVG",
                    data=buf_svg,
                    file_name=f"world_map_with_chords.svg",
                    mime="image/svg+xml",
                    use_container_width=True
                )

# Display initial instructions if no map generated yet
if not generate_btn:
    st.info("👈 Configure your settings above and click 'Generate Map' to create your visualization")
    
    # Display sample preview
    with st.expander("📋 Quick Start Guide", expanded=True):
        st.markdown("""
        ### 🚀 Getting Started:
        1. **Enter your data** in the format: `CountryName Frequency` or `Country1;Country2 Frequency`
        2. **Choose colors** from presets or create custom
        3. **Configure chord lines** (thickness, colors, labels)
        4. **Click Generate Map**!
        
        ### 🌍 Supported Country Name Formats:
        - **Full names**: `United States`, `Russia`, `United Kingdom`
        - **Abbreviations**: `US`, `USA`, `RU`, `UK`, `FR`
        - **Common names**: `Russia` (for Russian Federation), `South Korea` (for Korea)
        - **Case insensitive**: `united states`, `Us`, `rU` all work
        
        ### 📊 Data Formats:
        - **Single country**: `United States 7860` or `US 7860`
        - **Connection between 2 countries**: `USA;UK 150`
        - **Multiple countries**: `Germany;France;Italy 45` (creates pairwise connections)
        
        ### 🔗 Chord Line Features:
        - **Thickness** proportional to frequency
        - **Curved Bezier lines** for better visualization
        - **Color coding** options (single color or by frequency)
        - **Direction arrows** and labels
        - **Filtering** by minimum frequency
        
        ### 🎨 Default Visual Changes:
        - **Countries without data**: Now shown in **white** with **gray borders** (previously light gray)
        - **Better contrast** between colored and uncolored countries
        - **Cleaner, more professional look**
        
        ### 💡 Tips:
        - For best results, use consistent naming (all full names or all abbreviations)
        - The app will try to match variations automatically
        - Enable labels only for important connections to reduce clutter
        - Try different fill styles for unique visual effects
        """)

# Sidebar information
with st.sidebar:
    st.header("ℹ️ Features")
    
    st.markdown("""
    ### 🌍 Map Types:
    - **Heatmap**: Country colors by frequency
    - **Chord Diagram**: Lines show connections
    
    ### 🔗 Connection Lines:
    - Bezier curves between countries
    - Thickness = frequency
    - Multiple color schemes
    - Direction arrows
    - Frequency labels
    
    ### 🎨 Visual Effects:
    - 7 fill styles for countries
    - 10 color palettes
    - Custom 2/3-color gradients
    - **White background** for countries without data
    
    ### 📊 Data Support:
    - Full country names (United States)
    - Abbreviations (US, USA)
    - Common name variations
    - Pairwise connections
    - Multi-country interactions
    
    ### 💾 Export:
    - High-quality PNG (200 DPI)
    - Scalable SVG vector format
    """)
    
    st.divider()
    
    st.markdown("""
    **Supported Country Formats:**
    - United States, US, USA
    - Russia, Russian Federation, RU
    - United Kingdom, UK, GB
    - Saudi Arabia, SA, KSA
    - And 150+ other countries
    
    **Map Data Sources:**
    - Natural Earth geographic data
    - Built-in fallback datasets
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    developed by @daM, @CTA, https://chimicatechnoacta.ru
</div>
""", unsafe_allow_html=True)



