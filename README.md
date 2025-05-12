# image-processing-filters-simulation

## Introduction

This is a Python desktop application that provides an interactive GUI for applying a variety of image filters to photos. It is built with the Tkinter GUI toolkit and the OpenCV library. The application allows users to load an image, then apply filters one by one to observe their effects. It supports multiple categories of filters (spatial/statistical, edge-detection, and frequency-domain filters) and is useful for learning how different filters affect an image.

## Features

- **Image Input/Output:** Load images from your local file system for processing.
- **Statistical Filters:** Includes Min, Max, Median, Mean, Geometric Mean, Harmonic Mean, Contraharmonic Mean, Midpoint, and Alpha-Trimmed Mean filters.
- **Edge Detection Filters:** Includes Laplacian, Sobel, and Mexican Hat filters.
- **Frequency-Domain Filters:** Implements Low-Pass, High-Pass, and Band-Reject filters, each available in Ideal, Butterworth, and Gaussian variants.
- **Sequential Filtering:** Apply multiple filters in sequence. Each filter’s result can be fed into the next filter step-by-step, allowing complex effect combinations.
- **Interactive GUI:** A user-friendly Tkinter interface lets users select and apply filters with adjustable parameters (such as kernel size or cutoff frequency) and view the filtered image in real time.

## Project Structure
```
The repository is organized as follows:
├── assets/                               # contain different images for background and testing
├── screens/
│ ├── comparison_screen.py                # screen that shows differences between original and filtered image
│ ├── filter_selection_screen.py          # selecting specific filter
│ ├── parameter_screen.py                 # select the specific parameters of the screen
│ ├── processing_screen.py                # line by line visual processing of the image
│ └── splash_screen.py                    # the first-to-appear screen
│ └── upload_screen.py                    # used to choose which image to filter
├── main.py                               # renders the screens one by one 
└── README.md
```
