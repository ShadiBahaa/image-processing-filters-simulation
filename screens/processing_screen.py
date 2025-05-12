import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
import threading

class ProcessingScreen:
    def __init__(self, root, image, selected_filter, filter_params, next_callback, set_filtered_image_callback):
        self.root = root
        self.image = image
        self.selected_filter = selected_filter
        self.filter_params = filter_params
        self.next_screen = next_callback
        self.set_filtered_image = set_filtered_image_callback
        
        # Processing state
        self.processing_complete = False
        self.current_row = 0
        self.filtered_image = None
        self.display_image = None
        
        self.setup_ui()
        self.start_processing()
        
    def setup_ui(self):
        # Main container
        self.main_frame = tk.Frame(self.root, bg='#1a1a1a')
        self.main_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        # Title
        self.title = tk.Label(
            self.main_frame,
            text=f"Applying {self.selected_filter}...",
            font=("Helvetica", 24, "bold"),
            bg='#1a1a1a',
            fg='white'
        )
        self.title.pack(pady=(50, 20))
        
        # Progress text
        self.progress_text = tk.Label(
            self.main_frame,
            text="Processing image...",
            font=("Helvetica", 12),
            bg='#1a1a1a',
            fg='#888888'
        )
        self.progress_text.pack(pady=10)
        
        # Image display frame
        self.image_frame = tk.Frame(
            self.main_frame,
            bg='#1a1a1a',
            width=800,
            height=600
        )
        self.image_frame.pack(pady=20)
        self.image_frame.pack_propagate(False)
        
        # Image label
        self.image_label = tk.Label(
            self.image_frame,
            bg='#1a1a1a'
        )
        self.image_label.place(relx=0.5, rely=0.5, anchor='center')
        
        # Progress bar
        self.progress_canvas = tk.Canvas(
            self.main_frame,
            width=400,
            height=4,
            bg='#333333',
            highlightthickness=0
        )
        self.progress_canvas.pack(pady=20)
        
        self.progress_bar = self.progress_canvas.create_rectangle(
            0, 0, 0, 4,
            fill='#00ff00'
        )
        
    def start_processing(self):
        # Create a copy of the image for processing
        self.filtered_image = self.image.copy()
        self.display_image = self.image.copy()
        
        # Get image dimensions
        height, width = self.image.shape[:2]
        self.total_rows = height
        
        # Start processing thread
        threading.Thread(target=self.process_image, daemon=True).start()
        
        # Start update loop
        self.update_display()
        
    def process_image(self):
        """Process the image using the selected filter"""
        # Get filter parameters
        kernel_size = self.filter_params.get('Kernel Size', 3)
        border_type = self.filter_params.get('Border Type', 'reflect')
        
        # Convert border type string to cv2 constant
        border_types = {
            'reflect': cv2.BORDER_REFLECT,
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
            'wrap': cv2.BORDER_WRAP
        }
        cv2_border = border_types.get(border_type, cv2.BORDER_REFLECT)
        
        # Process based on filter type
        if self.selected_filter in ["Min Filter", "Max Filter", "Median Filter", "Mean Filter"]:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            if self.selected_filter == "Min Filter":
                # BORDER_WRAP not supported for erode, use BORDER_REFLECT instead
                if cv2_border == cv2.BORDER_WRAP:
                    cv2_border = cv2.BORDER_REFLECT
                self.filtered_image = cv2.erode(self.image, kernel, borderType=cv2_border)
            elif self.selected_filter == "Max Filter":
                self.filtered_image = cv2.dilate(self.image, kernel, borderType=cv2_border)
            elif self.selected_filter == "Median Filter":
                self.filtered_image = cv2.medianBlur(self.image, kernel_size)
            elif self.selected_filter == "Mean Filter":
                self.filtered_image = cv2.blur(self.image, (kernel_size, kernel_size), borderType=cv2_border)
                
        elif self.selected_filter == "Laplacian":
            scale = self.filter_params.get('Scale', 1)
            delta = self.filter_params.get('Delta', 0)
            # Apply Laplacian filter
            laplacian = cv2.Laplacian(self.image, cv2.CV_64F, ksize=kernel_size, scale=scale, delta=delta)
            laplacian = np.uint8(np.absolute(laplacian))
            # Add Laplacian to original image to get final sharpened result
            self.filtered_image = cv2.addWeighted(self.image, 1, laplacian, 1, 0)
            
        elif self.selected_filter == "Sobel":
            direction = self.filter_params.get('Direction', 'Both')
            scale = self.filter_params.get('Scale', 1)
            delta = self.filter_params.get('Delta', 0)
            
            if direction == 'X':
                sobel = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=kernel_size, scale=scale, delta=delta)
            elif direction == 'Y':
                sobel = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=kernel_size, scale=scale, delta=delta)
            else:
                sobelx = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=kernel_size, scale=scale, delta=delta)
                sobely = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=kernel_size, scale=scale, delta=delta)
                sobel = np.sqrt(sobelx**2 + sobely**2)
            
            # Convert to uint8 and add to original image
            sobel = np.clip(np.absolute(sobel), 0, 255).astype(np.uint8)
            self.filtered_image = cv2.addWeighted(self.image, 1, sobel, 1, 0)
            
        elif self.selected_filter == "Mexican hat":
            sigma = self.filter_params.get('Sigma', 1)
            # Create Mexican hat kernel
            size = kernel_size // 2
            y, x = np.ogrid[-size:size+1, -size:size+1]
            mex_kernel = (1/(2*np.pi*sigma**4)) * (2 - (x**2 + y**2)/(sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))
            self.filtered_image = cv2.filter2D(self.image, -1, mex_kernel, borderType=cv2_border)
            
        elif self.selected_filter == "Alpha-Trimmed Mean":
            alpha = self.filter_params.get('Alpha', 20) / 100.0  # Convert percentage to decimal
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            # Implement alpha-trimmed mean filter
            pad_size = kernel_size // 2
            padded = cv2.copyMakeBorder(self.image, pad_size, pad_size, pad_size, pad_size, cv2_border)
            self.filtered_image = np.zeros_like(self.image)
            
            if len(self.image.shape) == 3:  # Color image
                for i in range(pad_size, padded.shape[0] - pad_size):
                    for j in range(pad_size, padded.shape[1] - pad_size):
                        for channel in range(3):
                            window = padded[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1, channel].flatten()
                            trim_size = int(len(window) * alpha / 2)
                            sorted_window = np.sort(window)[trim_size:-trim_size]
                            # Handle empty window case and ensure uint8 output
                            if len(sorted_window) > 0:
                                mean_val = np.mean(sorted_window)
                                self.filtered_image[i-pad_size, j-pad_size, channel] = np.clip(mean_val, 0, 255).astype(np.uint8)
                            else:
                                self.filtered_image[i-pad_size, j-pad_size, channel] = padded[i,j,channel]
            else:  # Grayscale image
                for i in range(pad_size, padded.shape[0] - pad_size):
                    for j in range(pad_size, padded.shape[1] - pad_size):
                        window = padded[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1].flatten()
                        trim_size = int(len(window) * alpha / 2)
                        sorted_window = np.sort(window)[trim_size:-trim_size]
                        if len(sorted_window) > 0:
                            self.filtered_image[i-pad_size, j-pad_size] = np.clip(np.mean(sorted_window), 0, 255).astype(np.uint8)
                        else:
                            self.filtered_image[i-pad_size, j-pad_size] = padded[i,j]
                    
        elif self.selected_filter == "Geometric Mean":
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            pad_size = kernel_size // 2
            padded = cv2.copyMakeBorder(self.image, pad_size, pad_size, pad_size, pad_size, cv2_border)
            self.filtered_image = np.zeros_like(self.image)
            
            if len(self.image.shape) == 3:  # Color image
                for i in range(pad_size, padded.shape[0] - pad_size):
                    for j in range(pad_size, padded.shape[1] - pad_size):
                        window = padded[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1].astype(float)
                        for channel in range(3):
                            channel_window = window[:,:,channel]
                            self.filtered_image[i-pad_size, j-pad_size, channel] = np.uint8(np.exp(np.mean(np.log(channel_window + 1e-8))))
            else:  # Grayscale image
                for i in range(pad_size, padded.shape[0] - pad_size):
                    for j in range(pad_size, padded.shape[1] - pad_size):
                        window = padded[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1].astype(float)
                        self.filtered_image[i-pad_size, j-pad_size] = np.uint8(np.exp(np.mean(np.log(window + 1e-8))))
                    
        elif self.selected_filter == "Harmonic Mean":
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            pad_size = kernel_size // 2
            padded = cv2.copyMakeBorder(self.image, pad_size, pad_size, pad_size, pad_size, cv2_border)
            self.filtered_image = np.zeros_like(self.image)
            
            if len(self.image.shape) == 3:  # Color image
                for i in range(pad_size, padded.shape[0] - pad_size):
                    for j in range(pad_size, padded.shape[1] - pad_size):
                        window = padded[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1].astype(float)
                        for channel in range(3):
                            channel_window = window[:,:,channel]
                            self.filtered_image[i-pad_size, j-pad_size, channel] = np.uint8(len(channel_window.flatten()) / np.sum(1.0/(channel_window + 1e-8)))
            else:  # Grayscale image
                for i in range(pad_size, padded.shape[0] - pad_size):
                    for j in range(pad_size, padded.shape[1] - pad_size):
                        window = padded[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1].astype(float)
                        self.filtered_image[i-pad_size, j-pad_size] = np.uint8(len(window.flatten()) / np.sum(1.0/(window + 1e-8)))
                    
        elif self.selected_filter == "Contraharmonic Mean":
            q = self.filter_params.get('Q Parameter', 0)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            pad_size = kernel_size // 2
            padded = cv2.copyMakeBorder(self.image, pad_size, pad_size, pad_size, pad_size, cv2_border)
            self.filtered_image = np.zeros_like(self.image)
            
            if len(self.image.shape) == 3:  # Color image
                for i in range(pad_size, padded.shape[0] - pad_size):
                    for j in range(pad_size, padded.shape[1] - pad_size):
                        window = padded[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1].astype(float)
                        for channel in range(3):
                            channel_window = window[:,:,channel]
                            numerator = np.sum(np.power(channel_window + 1e-8, q + 1))
                            denominator = np.sum(np.power(channel_window + 1e-8, q))
                            self.filtered_image[i-pad_size, j-pad_size, channel] = np.uint8(numerator / denominator)
            else:  # Grayscale image
                for i in range(pad_size, padded.shape[0] - pad_size):
                    for j in range(pad_size, padded.shape[1] - pad_size):
                        window = padded[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1].astype(float)
                        numerator = np.sum(np.power(window + 1e-8, q + 1))
                        denominator = np.sum(np.power(window + 1e-8, q))
                        self.filtered_image[i-pad_size, j-pad_size] = np.uint8(numerator / denominator)
                    
        elif self.selected_filter == "Midpoint":
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            pad_size = kernel_size // 2
            padded = cv2.copyMakeBorder(self.image, pad_size, pad_size, pad_size, pad_size, cv2_border)
            self.filtered_image = np.zeros_like(self.image)
            
            if len(self.image.shape) == 3:  # Color image
                for i in range(pad_size, padded.shape[0] - pad_size):
                    for j in range(pad_size, padded.shape[1] - pad_size):
                        window = padded[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
                        for channel in range(3):
                            channel_window = window[:,:,channel]
                            min_val = np.min(channel_window)
                            max_val = np.max(channel_window)
                            self.filtered_image[i-pad_size, j-pad_size, channel] = np.uint8(np.clip((min_val.astype(float) + max_val.astype(float)) / 2, 0, 255))
            else:  # Grayscale image
                for i in range(pad_size, padded.shape[0] - pad_size):
                    for j in range(pad_size, padded.shape[1] - pad_size):
                        window = padded[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
                        min_val = np.min(window)
                        max_val = np.max(window)
                        self.filtered_image[i-pad_size, j-pad_size] = np.uint8(np.clip((min_val.astype(float) + max_val.astype(float)) / 2, 0, 255))

        elif self.selected_filter == "Ideal Low Pass":
            cutoff = self.filter_params.get('Cutoff Frequency', 50)
            
            # Process each channel separately for RGB images
            if len(self.image.shape) == 3:
                self.filtered_image = np.zeros_like(self.image)
                for channel in range(3):
                    # Convert channel to float and apply FFT
                    f = np.fft.fft2(self.image[:,:,channel].astype(float))
                    fshift = np.fft.fftshift(f)
                    
                    # Create meshgrid for filter
                    rows, cols = self.image.shape[:2]
                    crow, ccol = rows//2, cols//2
                    u = np.linspace(-crow, crow-1, rows)
                    v = np.linspace(-ccol, ccol-1, cols)
                    u, v = np.meshgrid(u, v)
                    
                    # Create ideal low pass filter mask
                    D = np.sqrt(u*u + v*v)
                    mask = np.zeros((rows, cols))
                    mask[D.T <= cutoff] = 1
                    
                    # Apply filter and inverse FFT
                    fshift_filtered = fshift * mask
                    f_ishift = np.fft.ifftshift(fshift_filtered)
                    img_back = np.fft.ifft2(f_ishift)
                    self.filtered_image[:,:,channel] = np.clip(np.abs(img_back), 0, 255).astype(np.uint8)
            else:
                # Convert grayscale image to float and apply FFT
                f = np.fft.fft2(self.image.astype(float))
                fshift = np.fft.fftshift(f)
                
                # Create meshgrid for filter
                rows, cols = self.image.shape
                crow, ccol = rows//2, cols//2
                u = np.linspace(-crow, crow-1, rows)
                v = np.linspace(-ccol, ccol-1, cols)
                u, v = np.meshgrid(u, v)
                
                # Create ideal low pass filter mask
                D = np.sqrt(u*u + v*v)
                mask = np.zeros((rows, cols))
                mask[D <= cutoff] = 1
                
                # Apply filter and inverse FFT
                fshift_filtered = fshift * mask
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                self.filtered_image = np.clip(np.abs(img_back), 0, 255).astype(np.uint8)
            

        elif self.selected_filter == "Butterworth Low Pass":
            cutoff = self.filter_params.get('Cutoff Frequency', 50)
            order = self.filter_params.get('Order', 2)
            
            # Process each color channel separately for RGB images
            self.filtered_image = np.zeros_like(self.image)
            if len(self.image.shape) == 3:
                for channel in range(3):
                    # Convert channel to float and apply FFT
                    f = np.fft.fft2(self.image[:,:,channel].astype(float))
                    fshift = np.fft.fftshift(f)
                    
                    # Create meshgrid for filter
                    rows, cols = self.image.shape[:2]
                    crow, ccol = rows//2, cols//2
                    u = np.linspace(-crow, crow-1, rows)
                    v = np.linspace(-ccol, ccol-1, cols)
                    u, v = np.meshgrid(u, v)
                    
                    # Create Butterworth low pass filter mask
                    D = np.sqrt(u*u + v*v)
                    # Avoid division by zero
                    eps = np.finfo(float).eps
                    D = np.maximum(D, eps)
                    mask = 1 / (1 + (D/cutoff)**(2*order))
                    
                    # Apply filter and inverse FFT
                    fshift_filtered = fshift * mask.T
                    f_ishift = np.fft.ifftshift(fshift_filtered)
                    img_back = np.fft.ifft2(f_ishift)
                    self.filtered_image[:,:,channel] = np.clip(np.abs(img_back), 0, 255).astype(np.uint8)
            else:
                # Handle grayscale images
                f = np.fft.fft2(self.image.astype(float))
                fshift = np.fft.fftshift(f)
                
                # Create meshgrid for filter
                rows, cols = self.image.shape
                crow, ccol = rows//2, cols//2
                u = np.linspace(-crow, crow-1, rows)
                v = np.linspace(-ccol, ccol-1, cols)
                u, v = np.meshgrid(u, v)
                
                # Create Butterworth low pass filter mask
                D = np.sqrt(u*u + v*v)
                # Avoid division by zero
                eps = np.finfo(float).eps
                D = np.maximum(D, eps)
                mask = 1 / (1 + (D/cutoff)**(2*order))
                
                # Apply filter and inverse FFT
                fshift_filtered = fshift * mask
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                self.filtered_image = np.clip(np.abs(img_back), 0, 255).astype(np.uint8)

        elif self.selected_filter == "Gaussian Low Pass":
            sigma = self.filter_params.get('Sigma', 50)
            
            if len(self.image.shape) == 3:  # Color image
                self.filtered_image = np.zeros_like(self.image)
                for channel in range(3):
                    # Convert channel to float and apply FFT
                    f = np.fft.fft2(self.image[:,:,channel].astype(float))
                    fshift = np.fft.fftshift(f)
                    
                    # Create meshgrid for filter
                    rows, cols = self.image.shape[:2]
                    crow, ccol = rows//2, cols//2
                    u = np.linspace(-crow, crow-1, rows)
                    v = np.linspace(-ccol, ccol-1, cols)
                    u, v = np.meshgrid(u, v)
                    
                    # Create Gaussian low pass filter mask
                    D = np.sqrt(u*u + v*v)
                    mask = np.exp(-(D*D)/(2*sigma*sigma))
                    
                    # Apply filter and inverse FFT
                    fshift_filtered = fshift * mask.T
                    f_ishift = np.fft.ifftshift(fshift_filtered)
                    img_back = np.fft.ifft2(f_ishift)
                    self.filtered_image[:,:,channel] = np.clip(np.abs(img_back), 0, 255).astype(np.uint8)
            else:  # Grayscale image
                # Convert image to float and apply FFT
                f = np.fft.fft2(self.image.astype(float))
                fshift = np.fft.fftshift(f)
                
                # Create meshgrid for filter
                rows, cols = self.image.shape
                crow, ccol = rows//2, cols//2
                u = np.linspace(-crow, crow-1, rows)
                v = np.linspace(-ccol, ccol-1, cols)
                u, v = np.meshgrid(u, v)
                
                # Create Gaussian low pass filter mask
                D = np.sqrt(u*u + v*v)
                mask = np.exp(-(D*D)/(2*sigma*sigma))
                
                # Apply filter and inverse FFT
                fshift_filtered = fshift * mask
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                self.filtered_image = np.clip(np.abs(img_back), 0, 255).astype(np.uint8)

        elif self.selected_filter == "Ideal High Pass":
            cutoff = self.filter_params.get('Cutoff Frequency', 50)
            
            if len(self.image.shape) == 3:  # Color image
                self.filtered_image = np.zeros_like(self.image)
                for channel in range(3):
                    # Convert channel to float and apply FFT
                    f = np.fft.fft2(self.image[:,:,channel].astype(float))
                    fshift = np.fft.fftshift(f)
                    
                    # Create meshgrid for filter
                    rows, cols = self.image.shape[:2]
                    crow, ccol = rows//2, cols//2
                    u = np.linspace(-crow, crow-1, rows)
                    v = np.linspace(-ccol, ccol-1, cols)
                    u, v = np.meshgrid(u, v)
                    
                    # Create ideal high pass filter mask
                    D = np.sqrt(u*u + v*v)
                    mask = D > cutoff
                    
                    # Apply filter and inverse FFT
                    fshift_filtered = fshift * mask.T
                    f_ishift = np.fft.ifftshift(fshift_filtered)
                    img_back = np.fft.ifft2(f_ishift)
                    filtered = np.abs(img_back).astype(np.uint8)
                    # Add weighted original image
                    self.filtered_image[:,:,channel] = cv2.addWeighted(filtered, 0.5, self.image[:,:,channel], 0.5, 0)
            else:  # Grayscale image
                # Convert image to float and apply FFT
                f = np.fft.fft2(self.image.astype(float))
                fshift = np.fft.fftshift(f)
                
                # Create meshgrid for filter
                rows, cols = self.image.shape
                crow, ccol = rows//2, cols//2
                u = np.linspace(-crow, crow-1, rows)
                v = np.linspace(-ccol, ccol-1, cols)
                u, v = np.meshgrid(u, v)
                
                # Create ideal high pass filter mask
                D = np.sqrt(u*u + v*v)
                mask = D > cutoff
                
                # Apply filter and inverse FFT
                fshift_filtered = fshift * mask
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                filtered = np.abs(img_back).astype(np.uint8)
                # Add weighted original image
                self.filtered_image = cv2.addWeighted(filtered, 0.5, self.image, 0.5, 0)

        elif self.selected_filter == "Butterworth High Pass":
            cutoff = self.filter_params.get('Cutoff Frequency', 50)
            order = self.filter_params.get('Order', 2)
            
            if len(self.image.shape) == 3:  # Color image
                self.filtered_image = np.zeros_like(self.image)
                for channel in range(3):
                    # Convert channel to float and apply FFT
                    f = np.fft.fft2(self.image[:,:,channel].astype(float))
                    fshift = np.fft.fftshift(f)
                    
                    # Create meshgrid for filter
                    rows, cols = self.image.shape[:2]
                    crow, ccol = rows//2, cols//2
                    u = np.linspace(-crow, crow-1, rows)
                    v = np.linspace(-ccol, ccol-1, cols)
                    u, v = np.meshgrid(u, v)
                    
                    # Create Butterworth high pass filter mask
                    D = np.sqrt(u*u + v*v)
                    # Handle division by zero by adding small epsilon
                    D = np.maximum(D, np.finfo(float).eps)  # Avoid division by zero
                    mask = 1 / (1 + (cutoff/D)**(2*order))
                    
                    # Apply filter and inverse FFT
                    fshift_filtered = fshift * mask.T
                    f_ishift = np.fft.ifftshift(fshift_filtered)
                    img_back = np.fft.ifft2(f_ishift)
                    filtered = np.abs(img_back).astype(np.uint8)
                    # Add weighted original image
                    self.filtered_image[:,:,channel] = cv2.addWeighted(filtered, 0.5, self.image[:,:,channel], 0.5, 0)
            else:  # Grayscale image
                # Convert image to float and apply FFT
                f = np.fft.fft2(self.image.astype(float))
                fshift = np.fft.fftshift(f)
                
                # Create meshgrid for filter
                rows, cols = self.image.shape
                crow, ccol = rows//2, cols//2
                u = np.linspace(-crow, crow-1, rows)
                v = np.linspace(-ccol, ccol-1, cols)
                u, v = np.meshgrid(u, v)
                
                # Create Butterworth high pass filter mask
                D = np.sqrt(u*u + v*v)
                D = np.maximum(D, np.finfo(float).eps)  # Avoid division by zero
                mask = 1 / (1 + (cutoff/D)**(2*order))
                
                # Apply filter and inverse FFT
                fshift_filtered = fshift * mask
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                filtered = np.abs(img_back).astype(np.uint8)
                # Add weighted original image
                self.filtered_image = cv2.addWeighted(filtered, 0.5, self.image, 0.5, 0)

        elif self.selected_filter == "Gaussian High Pass":
            sigma = self.filter_params.get('Sigma', 50)
            
            if len(self.image.shape) == 3:  # Color image
                self.filtered_image = np.zeros_like(self.image)
                for channel in range(3):
                    # Convert channel to float and apply FFT
                    f = np.fft.fft2(self.image[:,:,channel].astype(float))
                    fshift = np.fft.fftshift(f)
                    
                    # Create meshgrid for filter
                    rows, cols = self.image.shape[:2]
                    crow, ccol = rows//2, cols//2
                    u = np.linspace(-crow, crow-1, rows)
                    v = np.linspace(-ccol, ccol-1, cols)
                    u, v = np.meshgrid(u, v)
                    
                    # Create Gaussian high pass filter mask
                    D = np.sqrt(u*u + v*v)
                    mask = 1 - np.exp(-(D*D)/(2*sigma*sigma))
                    
                    # Apply filter and inverse FFT
                    fshift_filtered = fshift * mask.T
                    f_ishift = np.fft.ifftshift(fshift_filtered)
                    img_back = np.fft.ifft2(f_ishift)
                    filtered = np.abs(img_back).astype(np.uint8)
                    # Add weighted original image
                    self.filtered_image[:,:,channel] = cv2.addWeighted(filtered, 0.5, self.image[:,:,channel], 0.5, 0)
            else:  # Grayscale image
                # Convert image to float and apply FFT
                f = np.fft.fft2(self.image.astype(float))
                fshift = np.fft.fftshift(f)
                
                # Create meshgrid for filter
                rows, cols = self.image.shape
                crow, ccol = rows//2, cols//2
                u = np.linspace(-crow, crow-1, rows)
                v = np.linspace(-ccol, ccol-1, cols)
                u, v = np.meshgrid(u, v)
                
                # Create Gaussian high pass filter mask
                D = np.sqrt(u*u + v*v)
                mask = 1 - np.exp(-(D*D)/(2*sigma*sigma))
                
                # Apply filter and inverse FFT
                fshift_filtered = fshift * mask
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                filtered = np.abs(img_back).astype(np.uint8)
                # Add weighted original image
                self.filtered_image = cv2.addWeighted(filtered, 0.5, self.image, 0.5, 0)

        elif self.selected_filter == "Ideal Band Reject":
            cutoff = self.filter_params.get('Cutoff Frequency', 50)
            bandwidth = self.filter_params.get('Bandwidth', 10)
            
            # Process each color channel separately
            self.filtered_image = np.zeros_like(self.image)
            for channel in range(3):
                # Convert channel to float and apply FFT
                f = np.fft.fft2(self.image[:,:,channel].astype(float))
                fshift = np.fft.fftshift(f)
                
                # Create meshgrid for filter
                rows, cols = self.image.shape[:2]
                crow, ccol = rows//2, cols//2
                u = np.linspace(-crow, crow-1, rows)
                v = np.linspace(-ccol, ccol-1, cols)
                u, v = np.meshgrid(u, v)
                
                # Create ideal band reject filter mask
                D = np.sqrt(u*u + v*v)
                mask = np.ones(D.shape)
                
                # Define band region
                D0 = cutoff
                W = bandwidth
                mask[(D >= D0 - W/2) & (D <= D0 + W/2)] = 0
                
                # Apply filter and inverse FFT
                fshift_filtered = fshift * mask.T
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                self.filtered_image[:,:,channel] = np.abs(img_back).astype(np.uint8)

        elif self.selected_filter == "Butterworth Band Reject":
            cutoff = self.filter_params.get('Cutoff Frequency', 50)
            order = self.filter_params.get('Order', 2)
            bandwidth = self.filter_params.get('Bandwidth', 10)
            
            # Process each color channel separately
            self.filtered_image = np.zeros_like(self.image)
            for channel in range(3):
                # Convert channel to float and apply FFT
                f = np.fft.fft2(self.image[:,:,channel].astype(float))
                fshift = np.fft.fftshift(f)
                
                # Create meshgrid for filter
                rows, cols = self.image.shape[:2]
                crow, ccol = rows//2, cols//2
                u = np.linspace(-crow, crow-1, rows)
                v = np.linspace(-ccol, ccol-1, cols)
                u, v = np.meshgrid(u, v)
                
                # Create Butterworth band reject filter mask
                D = np.sqrt(u*u + v*v)
                D0 = cutoff
                W = bandwidth
                # Avoid division by zero and numerical instability
                eps = np.finfo(float).eps
                D = np.maximum(D, eps)
                mask = 1 / (1 + ((D*W)/(D*D - D0*D0 + eps))**(2*order))
                
                # Apply filter and inverse FFT
                fshift_filtered = fshift * mask.T
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                self.filtered_image[:,:,channel] = np.clip(np.abs(img_back), 0, 255).astype(np.uint8)

        elif self.selected_filter == "Gaussian Band Reject":
            cutoff = self.filter_params.get('Cutoff Frequency', 50)
            bandwidth = self.filter_params.get('Bandwidth', 10)
            
            # Process each color channel separately
            self.filtered_image = np.zeros_like(self.image)
            for channel in range(3):
                # Convert channel to float and apply FFT
                f = np.fft.fft2(self.image[:,:,channel].astype(float))
                fshift = np.fft.fftshift(f)
                
                # Create meshgrid for filter
                rows, cols = self.image.shape[:2]
                crow, ccol = rows//2, cols//2
                u = np.linspace(-crow, crow-1, rows)
                v = np.linspace(-ccol, ccol-1, cols)
                u, v = np.meshgrid(u, v)
                
                # Create Gaussian band reject filter mask
                D = np.sqrt(u*u + v*v)
                D0 = cutoff
                W = bandwidth
                # Avoid division by zero
                eps = np.finfo(float).eps
                D = np.maximum(D, eps)
                mask = 1 - np.exp(-((D*D - D0*D0)/(D*W + eps))**2)
                
                # Apply filter and inverse FFT
                fshift_filtered = fshift * mask.T
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                self.filtered_image[:,:,channel] = np.clip(np.abs(img_back), 0, 255).astype(np.uint8)
            
        # Simulate row-by-row processing
        height, width = self.image.shape[:2]
        for row in range(height):
            self.display_image[row] = self.filtered_image[row]
            self.current_row = row
            #draw a thin black line showing the the current height
            #cv2.line(self.display_image, (0, row), (width, row), (0, 0, 0), 1)
            time.sleep(4/height)  # Distribute 4 seconds across all rows
            
            
        self.processing_complete = True
        
    def update_display(self):
        """Update the display with current processing progress"""
        if not self.processing_complete:
            # Update progress bar
            progress = self.current_row / self.total_rows
            self.progress_canvas.coords(
                self.progress_bar,
                0, 0,
                400 * progress, 4
            )
            
            # Update progress text
            percent = int(progress * 100)
            self.progress_text.configure(text=f"Processing... {percent}%")
            
            # Update image display
            if self.current_row > 0:
                display = self.display_image.copy()
                display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(display)
                
                # Resize to fit display while maintaining aspect ratio
                display_width = 800
                display_height = 600
                img.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(img)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
            
            # Schedule next update
            self.root.after(33, self.update_display)  # ~30 FPS
        else:
            # Processing complete, store result and move to next screen
            self.set_filtered_image(self.filtered_image)
            
            # Fade out animation
            def fade_out(alpha):
                if alpha > 0:
                    self.main_frame.configure(bg=f'#{int(alpha*255):02x}{int(alpha*255):02x}{int(alpha*255):02x}')
                    self.root.after(5, lambda: fade_out(alpha - 0.1))
                else:
                    self.next_screen()
                    
            fade_out(1.0) 