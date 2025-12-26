import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patheffects import withStroke
import numpy as np
import requests
import zipfile
import warnings
from colour import Color
import os
import tempfile
from shapely.geometry import Point
import io

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Interactive World Frequency Map",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Country code mapping for common codes
country_code_mapping = {
    'CN': 'CHN', 'IN': 'IND', 'RU': 'RUS', 'SA': 'SAU', 'US': 'USA',
    'EG': 'EGY', 'PK': 'PAK', 'BY': 'BLR', 'KR': 'KOR', 'GB': 'GBR',
    'KZ': 'KAZ', 'UA': 'UKR', 'FR': 'FRA', 'TR': 'TUR', 'JP': 'JPN',
    'DE': 'DEU', 'BR': 'BRA', 'IR': 'IRN', 'AU': 'AUS', 'ES': 'ESP',
    'MY': 'MYS', 'IT': 'ITA', 'CA': 'CAN', 'MX': 'MEX', 'NL': 'NLD',
    'SE': 'SWE', 'NO': 'NOR', 'FI': 'FIN', 'PL': 'POL', 'CZ': 'CZE'
}

# Predefined color palettes
PREDEFINED_PALETTES = {
    'Red Gradient': ['#FFEBEE', '#FF5252', '#D50000'],
    'Blue Gradient': ['#E3F2FD', '#2196F3', '#0D47A1'],
    'Green Gradient': ['#E8F5E9', '#4CAF50', '#1B5E20'],
    'Purple Gradient': ['#F3E5F5', '#9C27B0', '#4A148C'],
    'Orange Gradient': ['#FFF3E0', '#FF9800', '#E65100'],
    'Viridis': ['#440154', '#21918C', '#FDE725'],
    'Plasma': ['#0D0887', '#CC4678', '#F0F921'],
    'Inferno': ['#000004', '#BC3754', '#FCA50A'],
    'Magma': ['#000004', '#B53679', '#FCFFA4'],
    'Cividis': ['#00204D', '#5D8FAA', '#FFEA46'],
}

# Fill styles
FILL_STYLES = {
    'Matte (Flat)': {'type': 'matte', 'gradient': False, 'edge_effect': 'none', 'texture': 'none'},
    'Glossy': {'type': 'glossy', 'gradient': True, 'edge_effect': 'soft', 'texture': 'shine'},
    'Neon Glow': {'type': 'neon', 'gradient': True, 'edge_effect': 'glow', 'texture': 'glow'},
    'Smooth Gradient': {'type': 'smooth', 'gradient': True, 'edge_effect': 'smooth', 'texture': 'gradient'},
    'Metallic': {'type': 'metallic', 'gradient': True, 'edge_effect': 'sharp', 'texture': 'metal'},
    'Watercolor': {'type': 'watercolor', 'gradient': True, 'edge_effect': 'blurry', 'texture': 'paper'},
}

# Load world map data function
@st.cache_data
def load_world_map_data():
    """Load world map data with fallback options"""
    try:
        # Try to download Natural Earth data
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        response = requests.get(url, stream=True)
        
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_file.flush()
                
                with zipfile.ZipFile(tmp_file.name, 'r') as zip_ref:
                    extract_dir = tempfile.mkdtemp()
                    zip_ref.extractall(extract_dir)
                    
                    # Find the shapefile
                    for file in os.listdir(extract_dir):
                        if file.endswith('.shp'):
                            world = gpd.read_file(os.path.join(extract_dir, file))
                            os.unlink(tmp_file.name)
                            return world
            
            os.unlink(tmp_file.name)
        
        # Fallback to internal geopandas dataset
        st.warning("Using backup data source (Natural Earth low resolution)")
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        return world
        
    except Exception as e:
        st.error(f"Error loading map data: {e}")
        # Ultimate fallback
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            return world
        except:
            # Create minimal world data as last resort
            st.error("Could not load world map data. Using minimal dataset.")
            return None

def parse_input_data(input_text):
    """Parse input data from text area"""
    data = {}
    lines = input_text.strip().split('\n')
    for line in lines:
        if line.strip():
            parts = line.strip().split()
            if len(parts) >= 2:
                country_code = parts[0].upper()
                try:
                    frequency = float(parts[1])
                    data[country_code] = frequency
                except:
                    pass
    return data

def create_custom_colormap(color1, color2, color3=None, n_points=256):
    """Create custom colormap from colors"""
    if color3 is None:
        colors = [color1, color2]
    else:
        colors = [color1, color3, color2]
    return LinearSegmentedColormap.from_list('custom', colors, N=n_points)

def apply_fill_style_to_country(ax, geometry, base_color, style_name):
    """Apply fill style to a single country geometry"""
    try:
        style = FILL_STYLES[style_name]
        
        # Create GeoDataFrame for the country
        gdf = gpd.GeoDataFrame([{'geometry': geometry}])
        
        if style['type'] == 'matte':
            # Simple flat color
            gdf.plot(ax=ax, color=base_color, edgecolor='white', linewidth=0.5)
            
        elif style['type'] == 'glossy':
            # Glossy effect with gradient
            gdf.plot(ax=ax, color=base_color, edgecolor='white', linewidth=0.5, alpha=0.9)
            # Add highlight effect
            highlight_color = Color(base_color)
            highlight_color.luminance = min(1.0, highlight_color.luminance + 0.2)
            gdf.buffer(-0.05).plot(ax=ax, color=str(highlight_color), alpha=0.3, 
                                 edgecolor='none', linewidth=0)
            
        elif style['type'] == 'neon':
            # Neon glow effect
            gdf.plot(ax=ax, color=base_color, edgecolor='white', linewidth=1, alpha=0.8)
            # Add glow with lighter color
            glow_color = Color(base_color)
            glow_color.saturation = min(1.0, glow_color.saturation * 0.7)
            gdf.buffer(0.02).plot(ax=ax, color=str(glow_color), alpha=0.2, 
                                edgecolor='none', linewidth=0)
            
        elif style['type'] == 'smooth':
            # Smooth gradient effect
            gdf.plot(ax=ax, color=base_color, edgecolor='white', linewidth=0.5, alpha=0.7)
            # Add inner gradient
            inner_color = Color(base_color)
            inner_color.luminance = max(0.0, inner_color.luminance - 0.1)
            gdf.buffer(-0.1).plot(ax=ax, color=str(inner_color), alpha=0.9, 
                                edgecolor='none', linewidth=0)
            
        elif style['type'] == 'metallic':
            # Metallic effect
            gdf.plot(ax=ax, color=base_color, edgecolor='#CCCCCC', linewidth=0.8, alpha=0.85)
            # Add shine lines
            shine_color = Color(base_color)
            shine_color.luminance = min(1.0, shine_color.luminance + 0.3)
            # Create diagonal shine effect
            try:
                # Create a smaller geometry for shine
                shine_geom = geometry.buffer(-0.15)
                shine_gdf = gpd.GeoDataFrame([{'geometry': shine_geom}])
                shine_gdf.plot(ax=ax, color=str(shine_color), alpha=0.2, 
                             edgecolor='none', linewidth=0)
            except:
                pass
            
        elif style['type'] == 'watercolor':
            # Watercolor effect
            gdf.plot(ax=ax, color=base_color, edgecolor='white', linewidth=0.3, alpha=0.6)
            # Add texture with slightly different color
            texture_color = Color(base_color)
            texture_color.luminance = min(1.0, texture_color.luminance + 0.15)
            gdf.buffer(-0.08).plot(ax=ax, color=str(texture_color), alpha=0.4, 
                                 edgecolor='none', linewidth=0)
            
    except Exception as e:
        # Fallback to simple plotting
        gdf = gpd.GeoDataFrame([{'geometry': geometry}])
        gdf.plot(ax=ax, color=base_color, edgecolor='white', linewidth=0.5)

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

def generate_map():
    """Generate the world frequency map"""
    # Parse input data
    country_data = parse_input_data(st.session_state.data_input)
    
    if not country_data:
        st.error("No valid data to display. Please enter country codes and frequencies.")
        return None
    
    # Load world map data
    world = load_world_map_data()
    if world is None:
        st.error("Failed to load world map data.")
        return None
    
    # Map country codes to ISO_A3
    mapped_data = {}
    unmapped_codes = []
    for code, freq in country_data.items():
        matched = False
        if code in country_code_mapping:
            mapped_data[country_code_mapping[code]] = freq
            matched = True
        else:
            # Try to find in world data
            for iso_col in ['ISO_A3', 'ISO_A3_EH', 'ADM0_A3', 'ISO_A2']:
                if iso_col in world.columns:
                    matches = world[world[iso_col] == code]
                    if not matches.empty:
                        iso_code = matches.iloc[0]['ISO_A3'] if 'ISO_A3' in matches.columns else code
                        mapped_data[iso_code] = freq
                        matched = True
                        break
        
        if not matched:
            unmapped_codes.append(code)
    
    # Merge with world data
    if mapped_data:
        data_df = pd.DataFrame(list(mapped_data.items()), columns=['ISO_A3', 'frequency'])
        merged = world.merge(data_df, left_on='ISO_A3', right_on='ISO_A3', how='left')
    else:
        merged = world.copy()
        merged['frequency'] = None
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(18, 9), dpi=100)
    
    # Set background
    ax.set_facecolor(st.session_state.background_color)
    
    # Plot all countries as background
    world.plot(ax=ax, color='lightgrey', edgecolor=st.session_state.border_color, 
              linewidth=0.3, alpha=0.7)
    
    # Get valid data
    valid_data = merged[merged['frequency'].notna()].copy()
    
    if not valid_data.empty:
        # Get color values
        color1 = st.session_state.color1
        color2 = st.session_state.color2
        color3 = st.session_state.color3 if st.session_state.color_points == '3 Colors (Diverging)' else None
        
        # Create colormap
        cmap = create_custom_colormap(color1, color2, color3)
        
        # Normalize data
        min_val = valid_data['frequency'].min()
        max_val = valid_data['frequency'].max()
        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
        
        # Plot countries with data using selected fill style
        for idx, row in valid_data.iterrows():
            color = cmap(norm(row['frequency']))
            apply_fill_style_to_country(ax, row.geometry, color, st.session_state.fill_style)
        
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
                        label = row['ISO_A3'] if pd.notna(row['ISO_A3']) else ''
                    elif st.session_state.label_type == 'Value Only':
                        label = f"{row['frequency']:.0f}"
                    else:  # Code + Value
                        code = row['ISO_A3'] if pd.notna(row['ISO_A3']) else ''
                        label = f"{code}\n{row['frequency']:.0f}"
                    
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
                            weight='bold',
                            fontfamily='sans-serif',
                            path_effects=[withStroke(linewidth=2, foreground='white')],
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7, edgecolor='none')
                        )
                except Exception as e:
                    continue
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', 
                          pad=0.05, aspect=50, shrink=0.7)
        cbar.set_label(st.session_state.scale_title, fontsize=12)
        cbar.ax.tick_params(labelsize=10)
    
    # Set title
    ax.set_title(st.session_state.map_title, fontsize=18, pad=25, weight='bold', fontfamily='sans-serif')
    
    # Remove axes
    ax.set_axis_off()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, country_data, valid_data, unmapped_codes, min_val if not valid_data.empty else 0, max_val if not valid_data.empty else 0

def reset_settings():
    """Reset all settings to defaults"""
    st.session_state.data_input = """CN\t216
IN\t144
RU\t126
SA\t85
US\t83
EG\t51
PK\t46
BY\t41
KR\t38
GB\t35
KZ\t33
UA\t30
FR\t29
TR\t28
JP\t27
DE\t27
BR\t25
IR\t23
AU\t20
ES\t18
MY\t14"""
    st.session_state.map_title = "World Frequency Map"
    st.session_state.scale_title = "Frequency"
    st.session_state.palette_selection = "Red Gradient"
    st.session_state.color_points = "2 Colors (Linear)"
    st.session_state.fill_style = "Matte (Flat)"
    st.session_state.show_labels = False
    st.session_state.label_type = "Code + Value"
    st.session_state.top_n_labels = 10
    st.session_state.font_size = 8
    st.session_state.background_color = "#F5F5F5"
    st.session_state.border_color = "#FFFFFF"
    
    # Update color pickers based on palette
    selected_colors = PREDEFINED_PALETTES[st.session_state.palette_selection]
    st.session_state.color1 = selected_colors[0]
    st.session_state.color2 = selected_colors[-1]
    if len(selected_colors) > 1:
        st.session_state.color3 = selected_colors[1]

# Initialize session state
if 'data_input' not in st.session_state:
    st.session_state.data_input = """CN\t216
IN\t144
RU\t126
SA\t85
US\t83
EG\t51
PK\t46
BY\t41
KR\t38
GB\t35
KZ\t33
UA\t30
FR\t29
TR\t28
JP\t27
DE\t27
BR\t25
IR\t23
AU\t20
ES\t18
MY\t14"""

if 'map_title' not in st.session_state:
    st.session_state.map_title = "World Frequency Map"

if 'scale_title' not in st.session_state:
    st.session_state.scale_title = "Frequency"

if 'palette_selection' not in st.session_state:
    st.session_state.palette_selection = "Red Gradient"

if 'color_points' not in st.session_state:
    st.session_state.color_points = "2 Colors (Linear)"

if 'fill_style' not in st.session_state:
    st.session_state.fill_style = "Matte (Flat)"

if 'show_labels' not in st.session_state:
    st.session_state.show_labels = False

if 'label_type' not in st.session_state:
    st.session_state.label_type = "Code + Value"

if 'top_n_labels' not in st.session_state:
    st.session_state.top_n_labels = 10

if 'font_size' not in st.session_state:
    st.session_state.font_size = 8

if 'background_color' not in st.session_state:
    st.session_state.background_color = "#F5F5F5"

if 'border_color' not in st.session_state:
    st.session_state.border_color = "#FFFFFF"

# Initialize colors based on default palette
selected_colors = PREDEFINED_PALETTES[st.session_state.palette_selection]
if 'color1' not in st.session_state:
    st.session_state.color1 = selected_colors[0]
if 'color2' not in st.session_state:
    st.session_state.color2 = selected_colors[-1]
if 'color3' not in st.session_state:
    st.session_state.color3 = selected_colors[1] if len(selected_colors) > 1 else "#FFFFFF"

# Main app layout
st.markdown("""
## 📊 Interactive World Frequency Map Generator

This tool allows you to create customized world frequency maps with various styling options.
Enter country data below and customize the appearance settings.
""")

# Create columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Data Input")
    
    data_input = st.text_area(
        "Enter country codes and frequencies:",
        value=st.session_state.data_input,
        height=250,
        key="data_input_widget",
        help="Format: Country_Code<TAB>Frequency (Example: US\t100)"
    )
    
    st.session_state.data_input = data_input
    
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
    
    # Fill style
    st.subheader("🖌️ Fill Style")
    fill_style = st.selectbox(
        "Fill Style:",
        options=list(FILL_STYLES.keys()),
        index=list(FILL_STYLES.keys()).index(st.session_state.fill_style),
        key="fill_style_widget"
    )
    st.session_state.fill_style = fill_style

# Additional settings in expanders
with st.expander("🔧 Advanced Settings", expanded=False):
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
            "Border Color:",
            value=st.session_state.border_color,
            key="border_color_widget"
        )
        st.session_state.border_color = border_color
    
    with col4:
        st.subheader("🏷️ Label Settings")
        
        show_labels = st.checkbox(
            "Show country labels on map",
            value=st.session_state.show_labels,
            key="show_labels_widget"
        )
        st.session_state.show_labels = show_labels
        
        if show_labels:
            label_type = st.selectbox(
                "Label Type:",
                options=['Code Only', 'Value Only', 'Code + Value'],
                index=['Code Only', 'Value Only', 'Code + Value'].index(st.session_state.label_type),
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
    if st.button("💾 Download Sample Data", use_container_width=True):
        sample_data = """CN\t216
IN\t144
RU\t126
SA\t85
US\t83
EG\t51
PK\t46
BY\t41
KR\t38
GB\t35
KZ\t33
UA\t30
FR\t29
TR\t28
JP\t27
DE\t27
BR\t25
IR\t23
AU\t20
ES\t18
MY\t14"""
        st.download_button(
            label="📥 Click to download",
            data=sample_data,
            file_name="sample_country_data.txt",
            mime="text/plain"
        )

# Handle reset button
if reset_btn:
    reset_settings()
    st.rerun()

# Generate map when button is clicked
if generate_btn:
    with st.spinner("Generating map..."):
        result = generate_map()
        
        if result:
            fig, country_data, valid_data, unmapped_codes, min_val, max_val = result
            
            # Display the map
            st.subheader("🗺️ Generated Map")
            st.pyplot(fig)
            
            # Display statistics
            st.subheader("📊 Data Statistics")
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric("Countries in Input", len(country_data))
            
            with col_stat2:
                matched = valid_data['frequency'].notna().sum() if not valid_data.empty else 0
                st.metric("Countries Matched", f"{matched}/{len(country_data)}")
            
            with col_stat3:
                if not valid_data.empty:
                    st.metric("Frequency Range", f"{min_val:.1f} - {max_val:.1f}")
                else:
                    st.metric("Frequency Range", "N/A")
            
            with col_stat4:
                if not valid_data.empty:
                    st.metric("Average Frequency", f"{valid_data['frequency'].mean():.2f}")
                else:
                    st.metric("Average Frequency", "N/A")
            
            # Top countries table
            st.subheader("🏆 Top Countries")
            if country_data:
                top_df = pd.DataFrame({
                    'Country': list(country_data.keys()),
                    'Frequency': list(country_data.values())
                }).sort_values('Frequency', ascending=False).head(10)
                
                st.dataframe(top_df, use_container_width=True)
            
            # Show unmatched codes
            if unmapped_codes:
                st.warning(f"⚠️ {len(unmapped_codes)} country codes could not be matched: {', '.join(sorted(unmapped_codes)[:10])}")
                if len(unmapped_codes) > 10:
                    st.info(f"... and {len(unmapped_codes) - 10} more unmatched codes")
            
            # Map download options
            st.subheader("💾 Export Options")
            
            # Convert figure to bytes for download
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                st.download_button(
                    label="📥 Download Map as PNG",
                    data=buf,
                    file_name="world_frequency_map.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col_dl2:
                # Save as SVG
                buf_svg = io.BytesIO()
                fig.savefig(buf_svg, format='svg', bbox_inches='tight')
                buf_svg.seek(0)
                
                st.download_button(
                    label="📥 Download Map as SVG",
                    data=buf_svg,
                    file_name="world_frequency_map.svg",
                    mime="image/svg+xml",
                    use_container_width=True
                )

# Display initial instructions if no map generated yet
if not generate_btn:
    st.info("👈 Configure your settings in the sidebar and click 'Generate Map' to create your visualization")
    
    # Display sample preview
    with st.expander("📋 Sample Data Format", expanded=True):
        st.code("""CN\t216
IN\t144
RU\t126
US\t83
GB\t35
FR\t29
DE\t27
JP\t27
AU\t20
ES\t18""")
        
        st.markdown("""
        **Tips:**
        - Use ISO Alpha-2 (US) or Alpha-3 (USA) country codes
        - Separate country code and frequency with TAB or space
        - One country per line
        - Empty lines are ignored
        """)

# Sidebar information
with st.sidebar:
    st.header("ℹ️ Information")
    
    st.markdown("""
    ### How to Use:
    1. Enter country data (codes and frequencies)
    2. Customize colors using color pickers
    3. Select fill style for countries
    4. Configure labels if needed
    5. Click 'Generate Map'
    
    ### Supported Country Codes:
    - ISO Alpha-2 (US, GB, FR, etc.)
    - ISO Alpha-3 (USA, GBR, FRA, etc.)
    
    ### Features:
    - 10 predefined color palettes
    - 6 different fill styles
    - Custom background and borders
    - Country labeling options
    - Export to PNG/SVG
    
    ### Data Sources:
    - Natural Earth map data
    - Built-in fallback datasets
    """)
    
    st.divider()
    
    st.markdown("""
    **Note:** The application requires internet connection to download map data on first run.
    Subsequent runs will use cached data for faster loading.
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    Interactive World Frequency Map Generator | Created with Streamlit
</div>
""", unsafe_allow_html=True)