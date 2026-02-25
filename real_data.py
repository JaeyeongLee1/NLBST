import zipfile
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ═══════════════════════════════════════════════════════════════════
# EPA PM2.5 Dataset
# ═══════════════════════════════════════════════════════════════════

class EPAPM25Loader:
    
    BASE_URL = "https://aqs.epa.gov/aqsweb/airdata"
    
    def __init__(self, data_dir: str = "./data/pm25"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    
    def download_year(self, year: int) -> Path:
        """Download daily PM2.5 CSV for a given year."""
        filename = f"daily_88101_{year}.csv"
        zip_filename = f"daily_88101_{year}.zip"
        csv_path = self.data_dir / filename
        zip_path = self.data_dir / zip_filename
        
        if csv_path.exists():
            return csv_path
        
        if not HAS_REQUESTS:
            raise ImportError(
                "requests library required for download. "
                "Install with: pip install requests\n"
                f"Or manually download from {self.BASE_URL}/{zip_filename}"
            )
        
        url = f"{self.BASE_URL}/{zip_filename}"
        print(f"  Downloading {zip_filename}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(self.data_dir)

        zip_path.unlink()
        print(f"  ✓ {filename}")
        
        return csv_path
    
    def download_all(self, years: range) -> list:
        """Download data for all specified years."""
        print(f"Downloading EPA PM2.5 data ({years.start}-{years.stop-1})...")
        paths = []
        for year in years:
            paths.append(self.download_year(year))
        print("✓ All downloads complete.")
        return paths
    

    def load(self,
             years: range = range(2015, 2021),
             min_coverage: float = 0.8,
             max_stations: Optional[int] = 216,
             log_transform: bool = True,
             seed: int = 42,
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Load and preprocess EPA PM2.5 data.
        
        Args:
            years: Range of years to include
            min_coverage: Minimum fraction of weeks a station must have data
            max_stations: Cap on number of stations (select most complete)
            log_transform: Apply log(1 + x) transform (recommended for PM2.5)
            seed: Random seed for station selection tiebreaking
        
        Returns:
            observations: [T, S] weekly mean PM2.5 (with NaN for missing)
            times: [T] week indices (0, 1, 2, ...)
            sensor_coords: [S, 2] normalized (lon, lat) in [0, 1]^2
            metadata: dict with station info, raw coords, etc.
        """
        self.download_all(years)
        
        print("Processing PM2.5 data...")
        dfs = []
        for year in years:
            csv_path = self.data_dir / f"daily_88101_{year}.csv"
            df = pd.read_csv(csv_path, usecols=[
                'Date Local', 'Arithmetic Mean',
                'Latitude', 'Longitude',
                'State Code', 'County Code', 'Site Num',
                'State Name', 'County Name',
            ])
            dfs.append(df)
        
        df = pd.concat(dfs, ignore_index=True)
        df['Date Local'] = pd.to_datetime(df['Date Local'], format='mixed')
        
        df['station_id'] = (
            df['State Code'].astype(str).str.zfill(2) + '_' +
            df['County Code'].astype(str).str.zfill(3) + '_' +
            df['Site Num'].astype(str).str.zfill(4)
        )
        
        df['week'] = df['Date Local'].dt.isocalendar().week.astype(int)
        df['year'] = df['Date Local'].dt.isocalendar().year.astype(int)
        df['year_week'] = df['year'].astype(str) + '_' + df['week'].astype(str).str.zfill(2)
        
        weekly = df.groupby(['station_id', 'year_week']).agg({
            'Arithmetic Mean': 'mean',
            'Latitude': 'first',
            'Longitude': 'first',
            'State Name': 'first',
            'County Name': 'first',
            'Date Local': 'min',
        }).reset_index()
        
        week_order = sorted(weekly['year_week'].unique())
        week_to_idx = {w: i for i, w in enumerate(week_order)}
        weekly['week_idx'] = weekly['year_week'].map(week_to_idx)
        
        T_total = len(week_order)
        
        station_counts = weekly.groupby('station_id')['week_idx'].count()
        min_weeks = int(T_total * min_coverage)
        valid_stations = station_counts[station_counts >= min_weeks].index.tolist()
        
        print(f"  Total weeks: {T_total}")
        print(f"  Stations with ≥{min_coverage:.0%} coverage: {len(valid_stations)}")
        
        if len(valid_stations) == 0:
            raise ValueError(
                f"No stations meet {min_coverage:.0%} coverage threshold. "
                f"Try lowering min_coverage."
            )
        
        if max_stations and len(valid_stations) > max_stations:
            rng = np.random.default_rng(seed)
            coverage = station_counts.loc[valid_stations].sort_values(ascending=False)
            valid_stations = coverage.head(max_stations).index.tolist()
            print(f"  Selected top {max_stations} stations by coverage")
        
        S = len(valid_stations)
        station_to_idx = {s: i for i, s in enumerate(valid_stations)}
        

        observations = np.full((T_total, S), np.nan)
        
        filtered = weekly[weekly['station_id'].isin(valid_stations)]
        for _, row in filtered.iterrows():
            t = row['week_idx']
            s = station_to_idx[row['station_id']]
            observations[t, s] = row['Arithmetic Mean']
        

        station_info = filtered.groupby('station_id').agg({
            'Latitude': 'first',
            'Longitude': 'first',
            'State Name': 'first',
            'County Name': 'first',
        }).loc[valid_stations]
        
        raw_coords = np.column_stack([
            station_info['Longitude'].values,
            station_info['Latitude'].values,
        ])  
        
        coord_min = raw_coords.min(axis=0)
        coord_max = raw_coords.max(axis=0)
        coord_range = coord_max - coord_min
        coord_range[coord_range == 0] = 1.0
        sensor_coords = (raw_coords - coord_min) / coord_range
        
        observations = np.clip(observations, 0, None)
        

        if log_transform:
            valid_mask = ~np.isnan(observations)
            observations[valid_mask] = np.log1p(observations[valid_mask])
        
        natural_missing = np.isnan(observations)
        
        col_means = np.nanmean(observations, axis=0)
        for s in range(S):
            nan_mask = np.isnan(observations[:, s])
            observations[nan_mask, s] = col_means[s]
        
        times = np.arange(T_total, dtype=np.float64)
        
        coverage_rate = 1.0 - natural_missing.mean()
        
        print(f"  Final: T={T_total} weeks, S={S} stations")
        print(f"  Data coverage: {coverage_rate:.1%}")
        print(f"  PM2.5 range: [{np.nanmin(observations):.2f}, {np.nanmax(observations):.2f}]"
              + (" (log-transformed)" if log_transform else ""))
        
        metadata = {
            'dataset': 'EPA_PM25',
            'years': list(years),
            'T_total': T_total,
            'S_total': S,
            'log_transform': log_transform,
            'coverage': coverage_rate,
            'natural_missing_rate': natural_missing.mean(),
            'natural_missing_mask': natural_missing,
            'station_ids': valid_stations,
            'station_info': station_info,
            'raw_coords': raw_coords,
            'coord_min': coord_min,
            'coord_max': coord_max,
            'week_labels': week_order,
        }
        
        return observations, times, sensor_coords, metadata


def load_pm25_from_csvs(csv_dir: str,
                        years: range = range(2015, 2021),
                        **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load PM2.5 data from already-downloaded CSVs.
    
    Args:
        csv_dir: Directory containing daily_88101_{year}.csv files
        years: Year range
        **kwargs: Passed to EPAPM25Loader.load()
    
    Returns:
        Same as EPAPM25Loader.load()
    """
    loader = EPAPM25Loader(data_dir=csv_dir)
    return loader.load(years=years, **kwargs)