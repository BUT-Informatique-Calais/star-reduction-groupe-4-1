[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zP0O23M7)

# Star Reduction - Astronomical Image Processing

FITS image processing application for star reduction in astronomical images. This project provides an intuitive graphical interface to detect, mask, and reduce stars in FITS images while preserving background structures (nebulae, galaxies, etc.).

## Features

- **Automatic star detection**: Uses DAOStarFinder to identify stars in images
- **Adaptive masking**: Creates masks adapted to each star's size
- **Controllable erosion**: Applies customizable erosion to adjust star reduction
- **Intelligent inpainting**: Reconstructs background by replacing stars with interpolated data
- **Modern GUI**: PyQt6 interface with interactive before/after visualization
- **Multi-format support**: Handles both monochrome and color (RGB) FITS images

## Installation

### Virtual Environment (recommended)

It is recommended to create a virtual environment before installing dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Dependencies

Install all dependencies with:

```bash
pip install -r requirements.txt
```

**Required libraries:**
- `astropy`: FITS file reading and manipulation
- `matplotlib`: Visualization and image saving
- `numpy`: Numerical computing and array manipulation
- `opencv-python`: Image processing (erosion, inpainting)
- `photutils`: Astronomical star detection
- `PyQt6`: Graphical interface

## Usage

### Graphical Interface (recommended)

Launch the application with GUI:

```bash
python interface.py
```

**Interface controls:**
1. Click **"Load FITS Image"** to open a FITS file
2. Adjust the **erosion kernel size** (3-31 pixels) with the slider
3. Modify the **detection threshold** (0.5-5.0) to control sensitivity
4. Click **"Process"** to start processing
5. View results in the tabs:
   - **Grid View**: Displays the 4 processing steps
   - **Comparison**: Compare before/after with a sliding bar
6. Save results with the **"Save Results"** button

### Command Line Script

For simple processing without interface:

```bash
python erosion.py
```

This script processes `examples/HorseHead.fits` by default and saves results in `results/`.

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for full dependency list

## Example Files

Test files are available in the `examples/` directory:

- **`HorseHead.fits`**: Monochrome FITS image (Horsehead Nebula)
- **`test_M31_linear.fits`**: RGB color FITS image (Andromeda Galaxy, linear)
- **`test_M31_raw.fits`**: Raw RGB color FITS image (Andromeda Galaxy, unprocessed)