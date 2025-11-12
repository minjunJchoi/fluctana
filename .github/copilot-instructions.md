# Fluctana AI Coding Guidelines

Fluctana is a Python library for linear and nonlinear spectral analysis of tokamak plasma diagnostics data, specifically designed for analyzing fluctuations in KSTAR, DIII-D, and HL-3 fusion experiments.

## Core Architecture

### Central Analysis Workflow
The library follows a consistent pattern centered around the `FluctAna` class:

1. **Data Loading**: `A = FluctAna()` → `A.add_data()` with device-specific classes
2. **Preprocessing**: Optional filtering, normalization, channel selection
3. **Spectral Analysis**: `A.fftbins()` or `A.cwt()` for frequency decomposition
4. **Analysis Methods**: Coherence, cross-power, bicoherence, statistical measures
5. **Visualization**: `A.mplot()`, `A.cplot()`, `A.iplot()` for different plot types

### Device-Specific Data Classes
- `KstarEcei`, `KstarMds`, `KstarBes`, `KstarMir`, `KstarCss` - KSTAR tokamak diagnostics
- `DiiidBes`, `DiiidEcei`, `DiiidData` - DIII-D tokamak diagnostics  
- `Hl3Data` - HL-3 tokamak diagnostics
- Each class handles device-specific data paths, file formats, and channel naming conventions

## Key Patterns & Conventions

### Data Structure
- Data is stored in `FluctAna.Dlist[]` as list of `FluctData` objects
- Channel data: `D.data[channel_idx, time_idx]` (MxN array)
- Time series: `D.time` (1D array)
- Channel positions: `D.rpos`, `D.zpos` (radial/vertical positions in meters)
- Channel lists: `D.clist` (expandable using range notation like `'ECEI_GT1201-1208'`)

### Analysis Results Storage
- Spectral data: `D.spdata[channel, bin, frequency]` (complex)
- Analysis results: `D.val[channel, frequency/time]` 
- Analysis type: `D.vkind` (string identifier like 'coherence', 'cross_power')
- Reference channels: `D.rname[]` for cross-channel analysis

### Command-Line Analysis Scripts
Most analysis scripts in `/analysis/` follow this pattern:
```python
shot = int(sys.argv[1])        # Shot number
trange = eval(sys.argv[2])     # Time range [start, end]
ch1 = sys.argv[3]              # Channel specification
```

Example usage: `python check_coherence.py 18597 [1.78,1.81] ECEI_G0101-0408`

## Essential Methods & Parameters

### Data Loading
- `add_data(dev='KSTAR', shot=18597, clist=['ECEI_GT1201'], trange=[1.0,2.0], norm=1)`
- `norm`: 0=no normalization, 1=normalize by mean, 2=normalize by reference period
- Device names: 'KSTAR', 'DIII-D', 'HL-3'

### Spectral Analysis
- `fftbins(nfft=512, window='hann', overlap=0.5, detrend=0, full=0)`
- `full=0`: 0→fN (positive frequencies), `full=1`: -fN→fN (full spectrum)
- Common windows: 'hann', 'hamm', 'kaiser', 'HFT248D'

### Cross-Analysis Methods
- `coherence(done=0, dtwo=1)`: Coherence between datasets 0 and 1
- `cross_power(done=0, dtwo=1)`: Cross-power spectrum
- `cross_phase(done=0, dtwo=1)`: Phase relationships
- `bicoherence(done=0, dtwo=1)`: Nonlinear coupling analysis

### Visualization
- `mplot(dnum=0, type='time')`: Multi-channel time series
- `mplot(dnum=0, type='val')`: Analysis results (coherence, power, etc.)
- `cplot(dnum=0, frange=[0,100])`: 2D spatial plots with frequency integration
- `iplot()`: Interactive time-evolution imaging with keyboard controls

## Critical Implementation Details

### Channel Naming & Expansion
- ECEI channels use grid notation: `'ECEI_GT1201-1208'` expands to 8 channels
- Channel positions are automatically loaded for spatial analysis
- Use `expand_clist()` method for programmatic channel list generation

### Data Normalization Workflow
```python
# Typical analysis sequence
A.add_data(dev='KSTAR', shot=shot, clist=channels, trange=time_window, norm=1)
A.fftbins(nfft=1024, window='hann', overlap=0.5, detrend=0)
A.coherence(done=0, dtwo=1)  # Results stored in A.Dlist[1].val
A.mplot(dnum=1, type='val')   # Plot coherence results
```

### Filtering & Preprocessing
- SVD filtering: `svd_filt(cutoff=0.9)` for noise reduction
- Frequency filtering: `filt(name='FIR_pass', fL=1000, fH=50000)`
- Moving average: `ma_filt(twin=300e-6, window='hann')`

### Statistical Analysis
- `skplane()`: Skewness-kurtosis plane for turbulence characterization
- `chplane()`: Complexity-entropy plane analysis
- `hurst()`: Hurst exponent for self-similarity analysis

## Development Patterns

- Use `sys.path.insert(0, os.pardir)` to import from parent directory in analysis scripts
- Analysis scripts typically include commented parameter variations for experimentation
- Results are often saved using pickle for later analysis
- Interactive plotting uses matplotlib with custom colormaps (`CM = plt.cm.get_cmap('RdYlBu_r')`)
- Device-specific classes handle shot number ranges to determine correct data paths automatically
- Jupyter notebooks in `/examples/` follow VSCode cell format with markdown documentation

## Data Loading Architecture

### Device Data Path Management
- KSTAR ECEI: Automatic path selection based on shot number ranges (e.g., 2011-2022 campaigns)
- Data files follow pattern: `/eceidata*/exp_YEAR/ECEI.{shot:06d}.{device}.h5`
- Device classes handle both legacy (pre-19392) and current shot number formats
- All device classes inherit common channel expansion and time-base methods

### Channel Specification Patterns
- Range notation: `'ECEI_G0101-0408'` expands to full grid (4x8 = 32 channels)  
- Split lists: `sys.argv[3].split(',')` for multiple channel groups
- Position data automatically loaded: `D.rpos`, `D.zpos` arrays in meters

## File Organization

- `/analysis/`: Experiment-specific analysis scripts (often excluded from git)
- `/examples/`: Tutorial notebooks and standard analysis templates  
- `/docs/`: Documentation including tutorial PDFs
- Device-specific modules: `kstar*.py`, `diiid*.py`, `hl3*.py`
- Core analysis: `fluctana.py`, `specs.py`, `stats.py`, `filtdata.py`
- `/kstardata/`, `/chdata/`: Device-specific auxiliary data and calibration files

## Key Implementation Details

### Analysis Script Variations
- `/analysis/` vs `/examples/`: analysis scripts often handle multiple shots/time ranges
- Parameter handling: `eval(sys.argv[2])` for complex arguments like time ranges
- Example scripts use single shots, analysis scripts support shot arrays

### No Build System
- Pure Python library with direct imports, no setup.py or requirements.txt
- Dependencies managed manually (numpy, scipy, matplotlib, h5py, sklearn)
- Scripts run directly: `python check_coherence.py 18597 [1.78,1.81] ECEI_G0101-0408`

When creating new analysis scripts, follow the established command-line parameter pattern and use the standard analysis workflow for consistency with existing codebase.