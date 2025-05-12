import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
from screens.splash_screen import SplashScreen
from screens.upload_screen import UploadScreen
from screens.filter_selection_screen import FilterSelectionScreen
from screens.parameter_screen import ParameterScreen
from screens.processing_screen import ProcessingScreen
from screens.comparison_screen import ComparisonScreen

class ImageProcessingApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Processing Filters Simulator")
        self.root.attributes('-zoomed', True)  # Start maximized
        
        # Initialize variables
        self.current_image = None
        self.filtered_image = None
        self.selected_filter = None
        self.filter_params = {}
        
        # Start with splash screen
        self.show_splash_screen()
        
    def show_splash_screen(self):
        # Clear current screen
        for widget in self.root.winfo_children():
            widget.destroy()
            
        splash = SplashScreen(self.root, self.show_upload_screen)
        
    def show_upload_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()
            
        upload = UploadScreen(self.root, self.show_filter_selection, self.set_current_image)
        
    def show_filter_selection(self):
        for widget in self.root.winfo_children():
            widget.destroy()
            
        filter_screen = FilterSelectionScreen(
            self.root, 
            self.show_parameter_screen,
            self.show_upload_screen,
            self.set_selected_filter
        )
        
    def show_parameter_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()
            
        param_screen = ParameterScreen(
            self.root,
            self.selected_filter,
            self.show_processing_screen,
            self.show_filter_selection,
            self.set_filter_params
        )
        
    def show_processing_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()
            
        processing = ProcessingScreen(
            self.root,
            self.current_image,
            self.selected_filter,
            self.filter_params,
            self.show_comparison_screen,
            self.set_filtered_image
        )

    def show_comparison_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()
            
        comparison = ComparisonScreen(
            self.root,
            self.current_image,
            self.filtered_image,
            self.selected_filter,
            self.continue_with_filtered,  # Changed to use filtered image
            self.show_upload_screen,  # For "New Image" button
            self.continue_with_generated
        )
    # Setter methods
    def set_current_image(self, image):
        self.current_image = image
        
    def set_filtered_image(self, image):
        self.filtered_image = image
        
    def set_selected_filter(self, filter_name):
        self.selected_filter = filter_name
        
    def set_filter_params(self, params):
        self.filter_params = params
        
    def continue_with_filtered(self):
        """Continue processing with filtered image"""
        # Update original image to filtered image
        # self.current_image = self.filtered_image.copy()
        self.show_filter_selection()

    def continue_with_generated(self):
        self.current_image = self.filtered_image.copy()
        self.show_filter_selection()
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ImageProcessingApp()
    app.run()
