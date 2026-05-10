# GNSS Guardian - GPS Spoofing Detection System

**Hackathon Kościuszkon 2026 - Honeywell Theme #2**

Real-time GPS spoofing detection using hybrid ML + physics rules approach. Tested on real-world datasets from UAV HackRF attacks and autonomous vehicle testbeds.

![Demo](demo/comparison_plot.png)

## Project Overview

GNSS Guardian detects GPS spoofing attacks across different domains (drones, vehicles) using:
- **Machine Learning**: RandomForest for pattern detection  
- **Physics Rules**: Speed/acceleration/position jump validation
- **Hybrid Scoring**: Combined risk assessment
- **Explainable AI**: Clear reasoning for each alert

### Key Innovation
Unlike traditional binary classification, our system explains **WHY** it detected an attack:
- "Position jump: 840m in 1 second"
- "Impossible speed: 3,024 km/h" 
- "ML confidence: 94%"

## Results Summary

| Dataset | Samples | Attack Rate | ML Accuracy | Rule Accuracy | Domain |
|---------|---------|-------------|-------------|---------------|---------|
| **UAV (HackRF)** | 10,067 | 19.4% | **100.0%** | 40.7% | Drone |
| **AV (Real-world)** | 62,042 | 25.4% | **97.8%** | 88.4% | Vehicle |

### Key Findings
- **Cross-domain validation**: Same approach works for drones + vehicles
- **Real attack detection**: Tested on genuine HackRF spoofing + testbed data
- **Explainable results**: System provides reasoning for each detection
- **High accuracy**: 97-100% detection rates across domains

## Quick Demo

### Installation
```bash
git clone https://github.com/your-username/gnss-guardian
cd gnss-guardian
pip install -r requirements.txt
```

### Launch Interactive Dashboard
```bash
streamlit run dashboard.py
```

### Run Analysis Notebooks
```bash
# Open Jupyter and run the analysis notebooks
jupyter notebook notebooks/01_UAV_Analysis.ipynb
jupyter notebook notebooks/02_AV_Analysis.ipynb
```

## Live Demo Features

1. **Dataset Comparison** - Side-by-side UAV vs Vehicle results
2. **Interactive Maps** - GPS tracks with attack visualization  
3. **Live Simulation** - Real-time spoofing detection demo
4. **Explainable Alerts** - Detailed reasoning for each detection

## Project Structure

```
gnss-guardian-final/
├── notebooks/
│   ├── 01_UAV_Analysis.ipynb    # Complete UAV analysis with results
│   └── 02_AV_Analysis.ipynb     # Complete AV analysis with results
├── dashboard.py                 # Interactive Streamlit demo
├── requirements.txt             # Python dependencies  
├── README.md                   # Project documentation
└── .gitignore                  # Git configuration
```

## Technical Approach

### Machine Learning Pipeline
- **Feature Engineering**: GPS coordinates → speed, acceleration, heading changes
- **Models**: RandomForest (fast) / XGBoost (accurate) 
- **Validation**: Train/test split with cross-domain testing

### Physics-Based Rules
- **Speed limits**: Max realistic velocity for domain (drones: 200 km/h, vehicles: 300 km/h)
- **Position jumps**: Impossible coordinate changes (>50-100m/second)
- **GPS quality**: HDOP/VDOP degradation during attacks

### Hybrid Risk Scoring
```python
risk_score = 0.7 * ML_probability + 0.3 * rule_score
```

## Datasets Used

### UAV Dataset (Live GPS Spoofing and Jamming)
- **Source**: IEEE DataPort (Open Access)
- **Content**: Real HackRF attacks on drones in controlled lab
- **Samples**: 10k+ with spoofing + jamming scenarios
- **Features**: 84 GPS + sensor measurements

### AV Dataset (Real-world Vehicle Spoofing)  
- **Source**: University of Arizona ACL-Rover testbed
- **Content**: Genuine GPS spoofing attacks on autonomous vehicle
- **Samples**: 62k+ real-world navigation data
- **Features**: 44 vehicle + GPS telemetry

## Competitive Advantages

1. **Real Data**: Tested on genuine attacks, not simulations
2. **Cross-Domain**: Works for multiple vehicle types  
3. **Explainable**: Shows reasoning, not just classification
4. **Hybrid Approach**: Combines ML accuracy with rule interpretability
5. **Live Demo**: Interactive visualization of detection process

## Use Cases

- **Aviation**: Drone/aircraft GPS spoofing detection
- **Automotive**: Autonomous vehicle navigation security  
- **Maritime**: Ship/vessel GPS attack monitoring
- **Critical Infrastructure**: GNSS-dependent systems protection

## Future Work

- Real-time deployment on embedded systems
- Integration with other GNSS constellations (Galileo, GLONASS)
- Advanced ML models (ensemble, deep learning)
- Jamming vs. Spoofing classification refinement
- Mobile/IoT device protection

## Team

**3-person hackathon team** - Kościuszkon 2026
- Data Science & ML implementation
- Physics rules & domain expertise  
- Dashboard & presentation

## Acknowledgments

- **Honeywell** for challenge theme and domain expertise
- **IEEE DataPort** for UAV attack dataset access
- **University of Arizona** for AV-GPS real-world data
- **Hackathon Kościuszkon 2026** for the opportunity

---

*"Protecting GNSS integrity through intelligent detection"*
