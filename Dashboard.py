#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import base64
import warnings
import logging
from io import BytesIO

import pandas as pd
import numpy as np
import geopandas as gpd
import circlify
import pycountry
from PIL import Image

import dash
from dash import Dash, dcc, html, Input, Output, State

import plotly.graph_objects as go
import plotly.express as px

app = Dash(__name__)
server = app.server

STARRY_BACKGROUND = "https://images.unsplash.com/photo-1534796636912-3b95b3ab5986?w=1000&q=80"

def load_satellite_data():
    satcat_data = pd.read_csv("satcat.tsv", sep="\t", low_memory=False)
    
    satcat_data['LAUNCH_DATETIME'] = pd.to_datetime(satcat_data['LDate'], errors='coerce')
    satcat_data['Launch Year'] = satcat_data['LAUNCH_DATETIME'].dt.year
    
    satcat_data = satcat_data.dropna(subset=['LAUNCH_DATETIME'])
    satcat_data['Type'] = satcat_data['Type'].fillna('Unknown')
    satcat_data['Status'] = satcat_data['Status'].fillna('Unknown')
    satcat_data.loc[satcat_data['State'] == 'SU', 'State'] = 'RU'
    
    status_category_map = {
        'O': 'Active', 'AO': 'Active', 'OX': 'Inactive', 'N': 'Inactive',
        'R': 'Reentered', 'D': 'Reentered', 'AR': 'Reentered',
        'L': 'Landed', 'AL': 'Landed', 'E': 'Destroyed',
        'C': 'Destroyed', 'F': 'Destroyed', 'DSO': 'Deep Space', 'DSA': 'Deep Space'
    }
    satcat_data['StatusCategory'] = satcat_data['Status'].map(status_category_map).fillna('Other')
    
    total_satellites = len(satcat_data)
    status_counts = satcat_data['StatusCategory'].value_counts()
    active_count = str(status_counts.get('Active', 0))
    inactive_count = str(status_counts.get('Inactive', 0))
    other_count = str(total_satellites - int(active_count) - int(inactive_count))
    
    coarse_type_map = {
        'P': 'Payload', 'C': 'Component', 'R': 'Rocket Stage',
        'D': 'Debris', 'S': 'Suborbital', 'Z': 'Spurious', 'X': 'Deleted'
    }
    satcat_data['TypeCategory'] = satcat_data['Type'].apply(
        lambda x: coarse_type_map.get(str(x).strip()[0], 'Other') if pd.notna(x) and str(x).strip() else 'Unknown'
    )
    
    orbit_mapping = {
        'LLEO/E': 'LEO', 'LLEO/I': 'LEO', 'LLEO/P': 'LEO', 'LLEO/S': 'LEO', 'LLEO/R': 'LEO',
        'LEO/E': 'LEO', 'LEO/I': 'LEO', 'LEO/P': 'LEO', 'LEO/S': 'LEO', 'LEO/R': 'LEO',
        'MEO': 'MEO', 'HEO': 'HEO', 'HEO/M': 'HEO', 'GEO/S': 'GEO', 'GEO/I': 'GEO',
        'GEO/T': 'GEO', 'GEO/D': 'GEO', 'GEO/SI': 'GEO', 'GEO/ID': 'GEO', 'GEO/NS': 'GEO',
        'GTO': 'GEO', 'DSO': 'Other', 'CLO': 'Other', 'EEO': 'Other', 'ATM': 'Other',
        'SO': 'Other', 'TA': 'Other', 'VHEO': 'Other', 'HCO': 'Other', 'PCO': 'Other'
    }
    satcat_data['OrbitCategory'] = satcat_data['OpOrbit'].apply(lambda x: orbit_mapping.get(str(x).strip(), 'Other'))
    
    country_map = {c.alpha_2: c.name for c in pycountry.countries}
    satcat_data['Country'] = satcat_data['State'].map(country_map)
    satcat_data_clean = satcat_data.dropna(subset=['Country', 'Launch Year'])
    
    return satcat_data_clean, total_satellites, active_count, inactive_count, other_count

satcat_data_clean, total_satellites, active_count, inactive_count, other_count = load_satellite_data()

available_categories = ['All'] + sorted(satcat_data_clean['TypeCategory'].unique())
available_statuses = ['All'] + sorted(satcat_data_clean['StatusCategory'].unique())
available_orbits = ['All'] + sorted(satcat_data_clean['OrbitCategory'].unique())



# In[ ]:


def kpi_card(title, value, color):
    return html.Div([
        html.Div([
            html.H4(title, style={'marginBottom': '5px', 'color': 'white'}),
            html.H2(value, style={'color': color, 'margin': 0})
        ], style={
            'backgroundColor': '#1f2c56',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 4px 12px rgba(0,0,0,0.15)',
            'textAlign': 'center',
            'width': '100%'
        })
    ], style={'flex': 1, 'padding': '10px'})


# In[ ]:


def build_global_launch_map(df, selected_type='All', selected_status='All', selected_orbits='All'):
    import pycountry

    manual_country_map = {
        'UK': 'United Kingdom',
        'D': 'Germany',
        'E': 'Spain',
        'F': 'France',
        'I': 'Italy',
        'J': 'Japan',
        'L': 'Luxembourg',
        'P': 'Portugal',
        'S': 'Sweden',
        'SU': 'Russia',
        'CSFR': 'Czech Republic',
        'CSSR': 'Czech Republic',
        'HKUK': 'United Kingdom',
        'MYM': 'Malaysia',
        'BGN': 'Bulgaria',
        'CYM': 'United Kingdom',
        '-': 'Unknown',
    }
    country_map = {c.alpha_2: c.name for c in pycountry.countries}
    country_map.update(manual_country_map)

    df = df.copy()
    df.loc[df['State'] == 'SU', 'State'] = 'RU'
    df['Country'] = df['State'].map(country_map)
    df['Launch Year'] = pd.to_datetime(df['LDate'], errors='coerce').dt.year
    df = df.dropna(subset=['Country', 'Launch Year'])

    if selected_type != 'All':
        df = df[df['TypeCategory'] == selected_type]
    if selected_status != 'All':
        df = df[df['StatusCategory'] == selected_status]
    if selected_orbits != 'All':
        df = df[df['OrbitCategory'] == selected_orbits]

    if df['Launch Year'].dropna().empty:
        years = [1957]  # fallback
    else:
        years = list(range(1957, int(df['Launch Year'].max()) + 1))    
    countries = df['Country'].unique()
    grid = pd.MultiIndex.from_product([years, countries], names=['Launch Year', 'Country'])
    launch_counts = df.groupby(['Launch Year', 'Country']).size()
    launches_by_year = launch_counts.reindex(grid, fill_value=0).reset_index(name='Launch_Count')
    launches_by_year['Cumulative_Launches'] = launches_by_year.groupby('Country')['Launch_Count'].cumsum()

    first_launch_years = launches_by_year[launches_by_year['Cumulative_Launches'] > 0].groupby('Country')['Launch Year'].min()
    launches_by_year['First_Launch_Year'] = launches_by_year['Country'].map(first_launch_years)
    filtered = launches_by_year[launches_by_year['Launch Year'] >= launches_by_year['First_Launch_Year']]
    launches_trimmed = filtered.copy()

    annotations_timeline = [
        (1957, "🛰 Sputnik 1: First satellite (USSR)", 1.13, 0.9, 1),
        (1961, "🚀 Cold War space race escalates", 0.5, 1.05, 3),
        (1970, "🇨🇳 First Chinese satellite: Dong Fang Hong 1", 1.16, 0.7, 3),
        (1980, "🇷🇺 USSR pulls ahead of USA in total launches", 1.15, 0.85, 3),
        (1998, "🇪🇬 Egypt launches first African satellite", 0.9, 0.68, 2),
        (2000, "🇨🇳 China surges in launches", 1.03, 0.65, 2),
        (2012, "🇪🇺 Mini-satellite boom: 4+ European nations join space", 0.6, 0.9, 4),
        (2015, "🛰 Commercial CubeSat boom begins", 0.5, 0.1, 3),
        (2020, "🚀 Starlink deployment begins (USA)", -0.05, 0.65, 3),
        (2022, "🌐 3,000+ Starlink satellites", -0.05, 0.75, 3),
        (2025, "📡 Starlink dominates LEO", 0.2, 0.6, 1)
    ]
    frame_annotations = {year: [] for year in years}
    for start_year, text, x, y, duration in annotations_timeline:
        for offset in range(duration):
            y_frame = start_year + offset
            if y_frame in frame_annotations:
                frame_annotations[y_frame].append(
                    dict(
                        text=text,
                        x=x, y=y,
                        xref="paper", yref="paper",
                        showarrow=False,
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=4,
                        font=dict(size=13, color="black")
                    )
                )

    frames = []
    for year in years:
        year_data = launches_trimmed[launches_trimmed['Launch Year'] == year]
        frames.append(go.Frame(
            data=[go.Choropleth(
                locations=year_data['Country'],
                locationmode='country names',
                z=year_data['Cumulative_Launches'],
                colorscale='YlOrRd',
                zmin=0,
                zmax=launches_trimmed['Cumulative_Launches'].max(),
                marker_line_color='rgba(255,255,255,0.5)',
                colorbar_title='Cumulative Launches',
                hoverinfo='location+z',
                hovertemplate='<b>%{location}</b><br>Cumulative Launches: %{z}<extra></extra>'
            )],
            name=str(year),
            layout=go.Layout(
                annotations=frame_annotations.get(year, []),
                paper_bgcolor="rgba(0,0,0,0)"
            )
        ))

    # Step 7: Initial map
    initial_data = launches_trimmed[launches_trimmed['Launch Year'] == min(years)]

    fig = go.Figure(
        data=go.Choropleth(
            locations=initial_data['Country'],
            locationmode='country names',
            z=initial_data['Cumulative_Launches'],
            colorscale='YlOrRd',
            zmin=0,
            zmax=launches_trimmed['Cumulative_Launches'].max(),
            marker_line_color='rgba(255,255,255,0.5)',
            colorbar_title='Cumulative Launches',
            hoverinfo='location+z',
            hovertemplate='<b>%{location}</b><br>Cumulative Launches: %{z}<extra></extra>'
        ),
        frames=frames
    )

    # Step 8: Layout + background
    fig.update_layout(
        title=f"<b>Global Satellite Launches by Country (1957–{years[-1]})</b>",
        title_x=0.5,
        geo=dict(
            showland=True,
            landcolor="rgb(169,169,169)",
            showocean=True,
            oceancolor="rgb(65,105,225)",
            showcoastlines=True,
            coastlinecolor="black",
            showcountries=True,
            countrycolor="black",
            showframe=False,
            bgcolor='rgba(0,0,0,0)',
            projection_type='natural earth'
        ),
        images=[dict(
            source=STARRY_BACKGROUND,
            xref="paper", yref="paper",
            x=0, y=1,
            sizex=1, sizey=1,
            xanchor="left", yanchor="top",
            sizing="stretch",
            opacity=1,
            layer="below"
        )],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12, color="white"),
        margin=dict(l=30, r=30, t=80, b=30),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.5, y=-0.01,
            xanchor="center", yanchor="top",
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, {
                    "frame": {"duration": 500, "redraw": True},
                    "fromcurrent": True
                }]),
                dict(label="⏸ Pause", method="animate", args=[[None], {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate"
                }])
            ],
            pad={"r": 10, "t": 10},
            showactive=False,
            bgcolor="#1f2c56",
            borderwidth=1,
            bordercolor="#555"
        )],
        sliders=[dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue={
                "prefix": "Launch Year = ",
                "font": {"size": 17, "color": "white"},
                "xanchor": "right"
            },
            pad={"t": 30},
            steps=[
                dict(method="animate", args=[[str(year)], {
                    "mode": "immediate",
                    "frame": {"duration": 500, "redraw": True},
                    "transition": {"duration": 0}
                }], label=str(year)) for year in years
            ],
            bgcolor="#1f2c56",
            bordercolor="#555",
            tickcolor="white"
        )],
        hoverlabel=dict(
            bgcolor="#2c3e50",
            font_size=14,
            font_family="Arial"
        )
    )

    return fig


# In[ ]:


def build_bar_race():
    df = satcat_data_clean.copy()
    df['LAUNCH_YEAR'] = df['Launch Year']

    launches = df.groupby(['LAUNCH_YEAR', 'State']).size().reset_index(name='Launches')

    top_3 = ['US', 'CN', 'RU']
    launches = launches[launches['State'].isin(top_3)]

    min_year = int(launches['LAUNCH_YEAR'].min()) if not launches.empty else 1957
    max_year = int(launches['LAUNCH_YEAR'].max()) if not launches.empty else 2024
    years = range(min_year, max_year + 1)
    countries = top_3  

    full_index = pd.MultiIndex.from_product([years, countries], names=['LAUNCH_YEAR', 'State'])
    complete_df = pd.DataFrame(index=full_index).reset_index()

    launches = pd.merge(complete_df, launches, on=['LAUNCH_YEAR', 'State'], how='left')
    launches['Launches'] = launches['Launches'].fillna(0)

    launches = launches.sort_values(by=['State', 'LAUNCH_YEAR'])
    launches['Cumulative'] = launches.groupby('State')['Launches'].cumsum()

    launches = launches.sort_values(['LAUNCH_YEAR', 'Cumulative'], ascending=[True, False])
    launches['Rank'] = launches.groupby('LAUNCH_YEAR')['Cumulative'].rank(method='first', ascending=False)

    country_names = {'US': 'United States', 'CN': 'China', 'RU': 'Russia'}
    flag_urls = {
        'US': 'https://flagcdn.com/w40/us.png',
        'CN': 'https://flagcdn.com/w40/cn.png',
        'RU': 'https://flagcdn.com/w40/ru.png'
    }

    launches['CountryName'] = launches['State'].map(country_names)
    launches['FlagURL'] = launches['State'].map(flag_urls)

    yearly_max = launches.groupby('LAUNCH_YEAR')['Cumulative'].max().reset_index()
    yearly_max['RangeMax'] = yearly_max['Cumulative'].apply(lambda x: 
        1000 if x <= 1000 else
        5000 if x <= 5000 else
        10000 if x <= 10000 else
        15000 if x <= 15000 else
        20000 if x <= 20000 else 25000
    )

    launches = pd.merge(launches, yearly_max[['LAUNCH_YEAR', 'RangeMax']], on='LAUNCH_YEAR')

    fig = px.bar(
        launches,
        x='Cumulative',
        y='CountryName',
        animation_frame='LAUNCH_YEAR',
        color='State',
        orientation='h',
        labels={'Cumulative': 'Total Launches', 'CountryName': 'Country'},
        height=600,
        width=500,
        title="<b>🚀 Global Space Race: Satellite Launches Over Time</b>",
        color_discrete_map={
            'US': '#3366CC',
            'CN': '#008000',
            'RU': '#FF9900'
        }
    )

    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                {
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 1000, 'redraw': True}, 'fromcurrent': True}]
                },
                {
                    'label': '⏸ Pause',
                    'method': 'animate',
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}]
                }
            ],
            'direction': 'left',
            'showactive': False,
            'x': 0.1,
            'xanchor': 'right',
            'y': 1.3,
            'yanchor': 'top',
            'bgcolor': '#1f2c56',
            'bordercolor': '#555',
            'font': {'color': 'white'}
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'prefix': 'Year: ',
                'font': {'size': 14, 'color': 'white'},
                'visible': True
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 40, 't': 100},
            'len': 0.9,
            'x': 0.1,
            'y': 1.5,
            'bgcolor': '#1f2c56',
            'bordercolor': '#555',
            'tickcolor': 'white',
            'ticklen': 5
        }],
        template='plotly_dark',
        font=dict(family="Arial, sans-serif", size=14, color="white"),
        title_font=dict(family="Arial, sans-serif", size=18, color="white"),
        title_x=0.5,
        transition=dict(duration=1000, easing='cubic-in-out'),
        showlegend=False,
        margin=dict(l=50, r=70, t=100, b=40),
        yaxis={'categoryorder': 'total ascending', 'title': ''},
        xaxis={'title': 'Total Satellite Launches', 'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.1)'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(bgcolor="#2c3e50", font_size=14, font_family="Arial, sans-serif")
    )

    for i, year in enumerate(launches['LAUNCH_YEAR'].unique()):
        year_range = launches[launches['LAUNCH_YEAR'] == year]['RangeMax'].iloc[0]
        if i < len(fig.frames):  # Prevent index error
            fig.frames[i].layout.update(xaxis_range=[0, year_range])

    if not launches.empty:
        initial_range = launches[launches['LAUNCH_YEAR'] == min_year]['RangeMax'].iloc[0]
        fig.update_layout(xaxis_range=[0, initial_range])

    fig.update_traces(
        textposition='outside',
        textfont=dict(color='white', size=14),
        marker=dict(line=dict(width=0)),
        hovertemplate='<b>%{y}</b><br>Launches: %{x}<extra></extra>'
    )

    for frame in fig.frames:
        frame_data = launches[launches['LAUNCH_YEAR'] == int(frame.name)]
        annotations = []
        for _, row in frame_data.iterrows():
            annotations.append(dict(
                x=row['Cumulative'],
                y=row['CountryName'],
                text=f" {int(row['Cumulative'])}",
                showarrow=False,
                font=dict(color="white", size=14),
                xanchor="left",
                yanchor="middle",
                xshift=5
            ))

        total_launches = int(frame_data['Cumulative'].sum())
        annotations.append(dict(
            xref='paper', yref='paper',
            x=0.95, y=0.2,
            xanchor='right', yanchor='top',
            text=f" <b>{frame.name}</b> ",
            showarrow=False,
            font=dict(color="white", size=32)
        ))
        annotations.append(dict(
            xref='paper', yref='paper',
            x=0.95, y=0.01,
            xanchor='right', yanchor='bottom',
            text=f"<b>Total: </b> {total_launches}",
            showarrow=False,
            font=dict(color="white", size=17)
        ))

        frame.layout.update(annotations=annotations)

        x_max = launches[launches['LAUNCH_YEAR'] == int(frame.name)]['RangeMax'].iloc[0]
        frame_images = []
        for _, row in frame_data.iterrows():
            if pd.isna(row['Cumulative']) or row['Cumulative'] == 0:
                continue

            x_val = row['Cumulative']
            y_val = row['CountryName']
            flag_url = row['FlagURL']
            x_paper = (x_val / x_max) * 0.8
            frame_images.append(dict(
                source=flag_url,
                xref="paper",
                yref="y",
                x=x_paper,
                y=y_val,
                sizex=0.08,
                sizey=0.6,
                xanchor="center",
                yanchor="middle",
                sizing="contain",
                opacity=1,
                layer="above"
            ))
        frame.layout.images = frame_images

    initial_data = launches[launches['LAUNCH_YEAR'] == min_year]
    initial_annotations = [
        dict(
            x=row['Cumulative'],
            y=row['CountryName'],
            text=f" {int(row['Cumulative'])}",
            showarrow=False,
            font=dict(color="white", size=14),
            xanchor="left",
            yanchor="middle",
            xshift=5
        )
        for _, row in initial_data.iterrows()
    ]
    fig.update_layout(annotations=initial_annotations)

    return fig


# In[ ]:


def launch_dashboard():
    return html.Div([
        html.Div([
            html.Div([
            html.Label("Satellite Type Category", style={'color': 'white'}),
            dcc.Dropdown(
                id='selected_type',
                options=[{'label': t, 'value': t} for t in available_categories],
                value='All',
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block', 'paddingRight': '2%', 'paddingLeft': '2%'}),
        html.Div([
            html.Label("Satellite Status Category", style={'color': 'white'}),
            dcc.Dropdown(
                id='selected_status',
                options=[{'label': s, 'value': s} for s in available_statuses],
                value='All',
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block', 'paddingRight': '2%'}),
        html.Div([
            html.Label("Satellite Orbit Category", style={'color': 'white'}),
            dcc.Dropdown(
                id='selected_orbits',
                options=[{'label': o, 'value': o} for o in available_orbits],
                value='All',
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(
                    id='global-launch-map',
                    figure=build_global_launch_map(satcat_data_clean),
                    style={'height': '700px'}
                )
            ], style={'width': '100%', 'padding': '10px'})
        ], style={'width': '100%'})
    ]) 


# In[ ]:


logging.getLogger('matplotlib.font_manager').disabled = True
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib.font_manager')

try:
    satcat = pd.read_csv("satcat.tsv", sep="\t", low_memory=False)
    pop_df = pd.read_csv("world_population_2023.csv")
    world = gpd.read_file("Natural Earth Data/ne_110m_admin_0_countries.shp")
except FileNotFoundError as e:
    print(f"Error loading data file: {e}. Make sure .tsv, .csv, and shapefiles are present.")
    exit()

# === Data Processing (Country Mapping, Corrections, etc.) ===
manual_country_map = {
    'UK': 'United Kingdom', 'D': 'Germany', 'E': 'Spain', 'F': 'France',
    'I': 'Italy', 'J': 'Japan', 'L': 'Luxembourg', 'P': 'Portugal',
    'S': 'Sweden', 'SU': 'Russian Federation', 'RU': 'Russian Federation' ,'CSFR': 'Czech Republic',
    'CSSR': 'Czech Republic', 'HKUK': 'United Kingdom', 'MYM': 'Malaysia',
    'BGN': 'Bulgaria', 'CYM': 'United Kingdom',
}
code_to_name = {}
for country in pycountry.countries:
    try: code_to_name[country.alpha_2] = country.name
    except AttributeError: pass

def resolve_country(code):
    if code in manual_country_map: return manual_country_map[code]
    if isinstance(code, str): return code_to_name.get(code)
    return None

satcat['Country'] = satcat['State'].apply(resolve_country)
satcat_clean = satcat.dropna(subset=['Country'])
launch_counts = satcat_clean['Country'].value_counts().to_dict()

pop_df['Country'] = pop_df['Country'].replace({'Russia': 'Russian Federation'})
population_counts = dict(zip(pop_df['Country'], pop_df['Population']))

world = world[['ADMIN', 'CONTINENT']]
world.columns = ['Country', 'Continent']
corrections = {
    "United States of America": "United States", "Russia": "Russian Federation",
    "South Korea": "Korea, Republic of", "North Korea": "Korea, Democratic People's Republic of",
    "Iran": "Iran, Islamic Republic of", "Syria": "Syrian Arab Republic",
    "Vietnam": "Viet Nam", "Laos": "Lao People's Democratic Republic",
    "Bolivia": "Bolivia, Plurinational State of", "Venezuela": "Venezuela, Bolivarian Republic of",
    "Tanzania": "Tanzania, United Republic of", "Brunei": "Brunei Darussalam",
    "Moldova": "Moldova, Republic of",
}
world['Country'] = world['Country'].replace(corrections)
world.loc[world['Country'] == 'Russian Federation', 'Continent'] = 'Europe'

country_to_continent = dict(zip(world['Country'], world['Continent']))
country_to_continent.update({
     "United States": "North America", 
     "Russia": "Europe" 
})


FLAG_DIR = "flags"
def normalize_country_name(name):
    name = str(name).lower()
    name = name.replace(",", "")
    replacements = {
        "republic of": "", "plurinational state of": "", "islamic republic of": "",
        "democratic people's republic of": "", "people's democratic republic": "",
        "united republic of": "", "bolivarian republic of": ""
    }
    for old, new in replacements.items(): name = name.replace(old, new)
    name = name.replace(" ", "_").strip("_")
    if name == 'viet_nam': return 'vietnam'
    return name

def load_flag_base64(country_name):
    manual_map = {
        "Korea, Republic of": "south_korea.png", "Korea, Democratic People's Republic of": "north_korea.png",
        "Iran, Islamic Republic of": "iran.png", "Russia": "russia.png",
        "Russian Federation": "russian_federation.png", "United States": "united_states.png",
        "United Kingdom": "united_kingdom.png", "Viet Nam": "vietnam.png",
    }
    flag_filename = manual_map.get(country_name) or f"{normalize_country_name(country_name)}.png"
    path = os.path.join(FLAG_DIR, flag_filename)
    if os.path.exists(path):
        try:
            with Image.open(path) as img:
                if img.format != 'PNG': img = img.convert('RGBA')
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                encoded_string = base64.b64encode(buffer.getvalue()).decode()
                return "data:image/png;base64," + encoded_string
        except Exception as e:
            print(f"Error processing flag {path}: {e}")
    return None

def create_circle_packing(data_mode='Satellite Launches'):
    global country_to_continent 

    if data_mode == 'Satellite Launches':
        value_map = launch_counts
        title = "Satellite Launches by Country and Continent"
    else:
        value_map = population_counts
        title = "Population by Country and Continent"

    df = world.copy()
    df['Value'] = df['Country'].map(value_map)
    df['ContinentLookup'] = df['Country'].map(country_to_continent)
    df = df.dropna(subset=['Value', 'ContinentLookup'])

    children = []
    for continent_name in df['ContinentLookup'].unique():
        if pd.isna(continent_name): continue
        d = df[df['ContinentLookup'] == continent_name]
        countries = [{'id': row['Country'], 'datum': row['Value']}
                     for _, row in d.iterrows() if pd.notna(row['Country']) and pd.notna(row['Value'])]
        if not countries: continue
        total = sum(c['datum'] for c in countries)
        children.append({'id': continent_name, 'datum': total, 'children': countries})

    if not children:
         fig = go.Figure()
         fig.update_layout(title=f"{title} (No data available)", xaxis={'visible':False}, yaxis={'visible':False}, paper_bgcolor='#1f2c56', plot_bgcolor='#1f2c56')
         return fig

    nested_data = {'id': 'World', 'children': children}
    try:
        circles = circlify.circlify(
            nested_data['children'], show_enclosure=False,
            target_enclosure=circlify.Circle(x=0, y=0, r=1),
        )
    except Exception as e:
         print(f"Error during circlify calculation: {e}")
         fig = go.Figure()
         fig.update_layout(title=f"{title} (Circlify Error)", xaxis={'visible':False}, yaxis={'visible':False}, paper_bgcolor='#1f2c56', plot_bgcolor='#1f2c56')
         return fig

    fig = go.Figure()
    shapes, annotations, images = [], [], []
    lim = max((abs(c.x) + c.r for c in circles), default=1) * 1.1

    continent_colors = {
        'Asia': '#A6CEE3', 'North America': '#B2DF8A', 'Europe': '#FB9A99',
        'South America': '#FDBF6F', 'Oceania': '#CAB2D6', 'Africa': '#FFFF99',
        'Antarctica': '#E0E0E0'
    }

    for c in circles:
        
        if c.ex is None: continue
        label = c.ex['id']
        level = c.level
        r = c.r
        x = c.x
    
        is_oceania = (
            (level == 1 and label == 'Oceania') or
            (level == 2 and country_to_continent.get(label) == 'Oceania')
        )
        
        is_south_america = (
            (level == 1 and label == 'South America') or
            (level == 2 and country_to_continent.get(label) == 'South America')
        )
        
        if is_oceania:
            y = c.y - 0.5
            x = c.x
        elif is_south_america:
            x = c.x + 0.27  
            y = c.y + 0.23   
        else:
            x = c.x
            y = c.y
        label = c.ex['id'] 
        level = c.level
        datum = c.ex.get('datum', 0)
        continent = None

        if level == 2: 
            country_name = label
            continent = country_to_continent.get(country_name)
            if continent is None:
                 
                 pass 
        elif level == 1: 
            continent = label

        facecolor = continent_colors.get(continent, '#DDDDDD') 

        shapes.append(go.layout.Shape(
            type="circle", x0=x - r, y0=y - r, x1=x + r, y1=y + r,
            fillcolor=facecolor, line_color="black", line_width=1,
            opacity=0.4, layer='below' if level == 1 else 'above'
        ))

        if level == 1: 
            font_size = 14 if r > 0.4 else (11 if r > 0.2 else 9)
        
            if data_mode == 'Satellite Launches':
                custom_label_positions = {
                    'South America': (0.35, 0.44),  
                    'Africa': (-0.017, 0.22),
                    'Oceania': (-0.01, -0.37)
                }
                label_x, label_y = custom_label_positions.get(label, (x, y))
            else:
                label_x, label_y = x, y
        
            annotations.append(go.layout.Annotation(
                x=label_x, y=label_y, text=f"<b>{label}</b>", showarrow=False,
                font=dict(size=font_size, color="black"),
                bgcolor="rgba(255,255,255,0.6)"
            ))

        elif level == 2: 
            flag_b64 = load_flag_base64(label) 
            if flag_b64 and r > 0.01: 
                images.append(go.layout.Image(
                    source=flag_b64, xref="x", yref="y", x=x, y=y,
                    sizex=r * 2, sizey=r * 2, xanchor="center", yanchor="middle",
                    sizing="contain", layer="above"
                ))
                shapes.append(go.layout.Shape( 
                    type="circle", x0=x - r, y0=y - r, x1=x + r, y1=y + r,
                    line_color="rgba(0,0,0,0.5)", line_width=1,
                    fillcolor='rgba(0,0,0,0)', layer='above'
                ))

            # Add text label
            text_threshold_radius = 0.03
            text_label_font_size = 7
            if isinstance(datum, (int, float)):
                 if datum >= 1_000_000: datum_str = f"{datum/1_000_000:.1f}M"
                 elif datum >= 1_000: datum_str = f"{datum/1_000:.1f}k"
                 else: datum_str = str(int(datum))
            else: datum_str = str(datum)

            if r > text_threshold_radius and (not flag_b64 or data_mode == 'Population'):
                 display_text = f"{label}<br>{datum_str}" if not flag_b64 else f"{datum_str}" 
                 annotations.append(go.layout.Annotation(
                     x=x, y=y, text=display_text, showarrow=False,
                     font=dict(size=text_label_font_size, color='black'),
                     bgcolor='rgba(255,255,255,0.6)', align='center'
                 ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='white'), x=0.5),
        shapes=shapes, annotations=annotations, images=images,
        xaxis=dict(range=[-lim, lim], visible=False, showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(range=[-lim, lim], visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, fixedrange=True),
        plot_bgcolor='#1f2c56', paper_bgcolor='#1f2c56',
        margin=dict(t=50, b=10, l=10, r=10), showlegend=False,
    )
    return fig


# In[ ]:


def infographic_section():
    return html.Div([
        html.Div([
            html.Div([
                dcc.Graph(
                    id='bar-race',
                    figure=build_bar_race(),
                    style={'height': '700px', 'width': '100%'},
                    config={'displayModeBar': False}
                )
            ], style={
                'width': '48%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '10px',
                'boxShadow': '0 4px 8px rgba(0,0,0,0.2)',
                'borderRadius': '10px',
                'backgroundColor': '#1f2c56',
                'marginLeft': '1%'
            }),
            
            html.Div([
                dcc.Dropdown(
                    id='circle-chart-mode',
                    options=[
                        {'label': 'Satellite Launches', 'value': 'Satellite Launches'},
                        {'label': 'Population', 'value': 'Population'}
                    ],
                    value='Satellite Launches',
                    clearable=False,
                    style={
                        'width': '80%',
                        'margin': '0 auto 15px',
                        'color': '#000000',
                        'backgroundColor': '#ffffff',
                        'borderColor': '#3d4f7a'
                    }
                ),
                dcc.Graph(
                    id='circle-chart',
                    figure=create_circle_packing(),
                    style={'height': '650px', 'width': '100%', 'marginTop': '20px'},  
                    config={'displayModeBar': False}
                )
            ], style={
                'width': '48%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '10px',
                'boxShadow': '0 4px 8px rgba(0,0,0,0.2)',
                'borderRadius': '10px',
                'backgroundColor': '#1f2c56',
                'overflow': 'hidden'  ,
                'marginLeft': '4%' 
            })
        ], style={
            'width': '100%',
            'marginTop': '20px',
            'display': 'flex',
            'justifyContent': 'space-between',
            'flexWrap': 'nowrap'
        })
    ])

def update_circle_chart(data_mode):
    return create_circle_packing(data_mode)

app.layout = html.Div(style={
    'backgroundColor': '#0e1a35',  
    'minHeight': '100vh', 
    'padding': '30px'
}, children=[
    html.H1("Satellite Dashboard", style={'textAlign': 'center', 'color': 'white', 'marginBottom': '40px'}),

    html.Div([
        kpi_card("Total Satellites", total_satellites, '#00ccff'),
        kpi_card("Active Satellites", active_count, '#00e396'),
        kpi_card("Inactive/Retired", inactive_count, '#ff4d4f'),
        kpi_card("Other", other_count, '#FFFF00'),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '20px'}),
    
    infographic_section(),
    
    launch_dashboard()
])

@app.callback(
    [Output('bar-race', 'figure'),
     Output('global-launch-map', 'figure'),
     Output('circle-chart', 'figure')],
    [Input('selected_type', 'value'),
     Input('selected_status', 'value'),
     Input('selected_orbits', 'value'),
     Input('circle-chart-mode', 'value')]
)
def update_charts(selected_type, selected_status, selected_orbits, data_mode):
    bar_race_fig = build_bar_race()
    launch_map_fig = build_global_launch_map(satcat_data_clean, selected_type, selected_status, selected_orbits)
    circle_packing_fig = create_circle_packing(data_mode)
    return bar_race_fig, launch_map_fig, circle_packing_fig


if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




