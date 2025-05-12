import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os

class ComparisonScreen:
    def __init__(self, root, original_image, filtered_image, filter_name, 
                 filter_selection_callback, new_image_callback, continue_with_generated_callback):
        self.root = root
        self.original_image = original_image
        self.filtered_image = filtered_image
        self.filter_name = filter_name
        self.show_filter_selection = filter_selection_callback
        self.show_new_image = new_image_callback
        self.continue_with_generated = continue_with_generated_callback
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container with gradient background
        self.main_frame = tk.Frame(self.root, bg='#1a1a1a')
        self.main_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        # Title
        self.title = tk.Label(
            self.main_frame,
            text=f"{self.filter_name} - Results",
            font=("Helvetica", 24, "bold"),
            bg='#1a1a1a',
            fg='white'
        )
        self.title.pack(pady=(30, 20))
        
        # Images comparison frame
        comparison_frame = tk.Frame(self.main_frame, bg='#1a1a1a')
        comparison_frame.pack(fill='both', expand=True, padx=50)
        
        # Original image frame
        original_frame = tk.Frame(comparison_frame, bg='#2a2a2a')
        original_frame.pack(side='left', fill='both', expand=True, padx=10)
        
        tk.Label(
            original_frame,
            text="Original Image",
            font=("Helvetica", 14),
            bg='#2a2a2a',
            fg='white'
        ).pack(pady=10)
        
        self.original_label = tk.Label(original_frame, bg='#2a2a2a')
        self.original_label.pack(pady=10)
        
        # Filtered image frame
        filtered_frame = tk.Frame(comparison_frame, bg='#2a2a2a')
        filtered_frame.pack(side='right', fill='both', expand=True, padx=10)
        
        tk.Label(
            filtered_frame,
            text="Filtered Image",
            font=("Helvetica", 14),
            bg='#2a2a2a',
            fg='white'
        ).pack(pady=10)
        
        self.filtered_label = tk.Label(filtered_frame, bg='#2a2a2a')
        self.filtered_label.pack(pady=10)
        
        # Display images
        self.display_images()
        
        # Buttons frame
        buttons_frame = tk.Frame(self.main_frame, bg='#1a1a1a')
        buttons_frame.pack(fill='x', padx=50, pady=20)
        
        # New image button
        self.new_image_btn = tk.Button(
            buttons_frame,
            text="New Image",
            command=self.show_new_image,
            font=("Helvetica", 12),
            bg='#2196F3',
            fg='white',
            relief='flat',
            padx=20,
            pady=10
        )
        self.new_image_btn.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        
        # Original image button
        self.original_btn = tk.Button(
            buttons_frame,
            text="Original Image",
            command=self.show_filter_selection,
            font=("Helvetica", 12),
            bg='#4CAF50',
            fg='white',
            relief='flat',
            padx=20,
            pady=10
        )
        self.original_btn.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        
        # Generated image button
        self.generated_btn = tk.Button(
            buttons_frame,
            text="Generated Image",
            command=self.continue_with_generated,
            font=("Helvetica", 12),
            bg='#2196F3',
            fg='white',
            relief='flat',
            padx=20,
            pady=10
        )
        self.generated_btn.grid(row=0, column=2, padx=5, pady=5, sticky='ew')
        
        # Configure grid columns to expand equally
        buttons_frame.grid_columnconfigure(0, weight=1)
        buttons_frame.grid_columnconfigure(1, weight=1)
        buttons_frame.grid_columnconfigure(2, weight=1)
        
        # Bind hover effects
        self.bind_hover_effects()
        
    def display_images(self):
        """Display both original and filtered images"""
        # Convert and resize original image
        original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        original_pil = Image.fromarray(original_rgb)
        original_pil.thumbnail((500, 500), Image.Resampling.LANCZOS)
        self.original_photo = ImageTk.PhotoImage(original_pil)
        self.original_label.configure(image=self.original_photo)
        
        # Convert and resize filtered image
        filtered_rgb = cv2.cvtColor(self.filtered_image, cv2.COLOR_BGR2RGB)
        filtered_pil = Image.fromarray(filtered_rgb)
        filtered_pil.thumbnail((500, 500), Image.Resampling.LANCZOS)
        self.filtered_photo = ImageTk.PhotoImage(filtered_pil)
        self.filtered_label.configure(image=self.filtered_photo)
        
    def continue_with_filtered(self):
        """Continue processing with filtered image"""
        # Update original image to filtered image
        self.original_image = self.filtered_image.copy()
        self.show_filter_selection()
        
    def bind_hover_effects(self):
        """Add hover effects to buttons"""
        def on_enter(e, btn, color):
            btn['background'] = color
            
        def on_leave(e, btn, color):
            btn['background'] = color
            
        self.original_btn.bind("<Enter>", lambda e: on_enter(e, self.original_btn, '#45a049'))
        self.original_btn.bind("<Leave>", lambda e: on_leave(e, self.original_btn, '#4CAF50'))
        
        self.generated_btn.bind("<Enter>", lambda e: on_enter(e, self.generated_btn, '#45a049'))
        self.generated_btn.bind("<Leave>", lambda e: on_leave(e, self.generated_btn, '#4CAF50'))
        
        self.new_image_btn.bind("<Enter>", lambda e: on_enter(e, self.new_image_btn, '#1976D2'))
        self.new_image_btn.bind("<Leave>", lambda e: on_leave(e, self.new_image_btn, '#2196F3'))