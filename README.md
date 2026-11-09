# ERA5 Wind ROM and Flight Routing

## Dataset
Download hourly wind data for January 1–31 of **2022, 2023, 2024, 2025, and 2026** from the [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/).

Select:
- Dataset: ERA5 hourly data on pressure levels
- Variables: U component of wind and V component of wind
- Pressure level: 250 hPa
- Time: 00:00–23:00
- Area: 18°N, 78°E, 0°N, 108°E
- Download format: NetCDF

## Folder Structure

```text
era5-wind-rom-flight-routing/
│
├── data/
│   └── Store downloaded ERA5 datasets here
│
├── src/
│   ├── datarep/
│   │   └── Data inspection and preprocessing scripts
│   │
│   └── main/
│       └── POD, DMD, OpInf and flight routing scripts
│
├── docs/
│   └── Project documentation and paper
│
└── results/
    └── Generated plots, figures and results
```

## Execution Order

1. Run the scripts in `src/datarep/` first.
2. Run the scripts in `src/main/` to perform ROM analysis and flight route optimization.
3. Generated outputs are stored in `results/`.