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

The repository is organized as follows:

    main.py – The main Python script that implements the GUI and applies the image filters.

    assets/ – Directory containing static assets (such as icons or default images) used by the application.

    screens/ – Directory containing example screenshots referenced in this README.

    README.md – Project documentation (this file).
