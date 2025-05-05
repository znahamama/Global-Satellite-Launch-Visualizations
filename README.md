# Global Satellite Launch Visualizations (1957–2025)

## Overview

This project visualizes global satellite launch activity from 1957 to 2023 using real-world data and compelling visuals. Through choropleth maps, sunburst charts, and infographics, it explores **which countries are launching satellites**, **why**, and **what this reveals about global inequality in space technology access**.

---

## Key Visuals

https://mybinder.org/v2/gh/znahamama/Global-Satellite-Launch-Visualizations/HEAD?urlpath=voila/render/Dashboard.ipynb

### Choropleth Map – Launches by Country
<img width="909" alt="Screenshot 2025-05-02 at 7 52 30 PM" src="https://github.com/user-attachments/assets/e3366040-02c4-4d42-a438-0fc7182a92c8" />


### Bubble Chart – Launch Volume by Country and Continent
<img width="785" alt="Screenshot 2025-05-02 at 7 53 43 PM" src="https://github.com/user-attachments/assets/8796651f-d7ad-402b-bf21-20678dd19008" />


### Infographic – Final Project Summary
![satellite_infographic](https://github.com/user-attachments/assets/69876ef8-6d6c-4744-a425-076f61402c52)

---

## 📁 Project Structure

```
├── Dashboard.ipynb                  # Jupyter notebook for the interactive dashboard (Voila-compatible)
├── Infographic.ipynb               # Jupyter notebook to generate the final infographic
├── satcat.tsv                      # Main raw satellite dataset (from GCAT)
├── psatcat.tsv                     # Preprocessed satellite dataset
├── world_population_2023.csv       # Country population reference data
├── satellite_infographic.png       # Final infographic (exported version)
├── Natural Earth Data/             # Contains shapefiles for choropleth maps
│   ├── ne_110m_admin_0_countries.shp, .shx, .dbf, etc.
├── flags/                          # Country flags used in bubble visualizations
│   ├── *.png
```
---

## Data Sources

- [CelesTrak / UCS Satellite Database](https://celestrak.org/satcat/)
- [Natural Earth](https://www.naturalearthdata.com/) – country shapefiles
- World Bank – 2023 population estimates

---

## Key Insights

- The **U.S. accounts for over 50%** of all launches from 2010 to 2023.
- **Communication** satellites dominate global payload purposes.
- **Africa and Global South** countries show little to no launch activity — highlighting digital infrastructure gaps.

---

## Technologies Used

- Python: `pandas`, `matplotlib`, `geopandas`, `seaborn`
- Jupyter Notebooks for data analysis and visualization
- Infographic layout using `matplotlib.gridspec`

---

## ▶️ How to Run the Project

1. **Clone this repository**
   ```bash
   git clone https://github.com/znahamama/Global-Satellite-Launch-Visualizations.git
   cd Global-Satellite-Launch-Visualizations
   '''
   
2. **Install dependencies**
   ```bash
   pip install pandas matplotlib geopandas
   '''

3. **Launch Jupyter**
   ```bash
   jupyter notebook
   '''

4. **Open:**
   - `Dashboard.ipynb` to explore the data interactively  
   - `Infographic.ipynb` to generate the final infographic




## 🪪 License

This project is licensed under the [MIT License](LICENSE).

---

## Credits

Developed by **Ziad Hamama**  
📍 Memorial University of Newfoundland  
📘 COMP 4304 — Final Project (Johnson Geo Centre Collaboration)
