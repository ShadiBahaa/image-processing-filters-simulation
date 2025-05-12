import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

class UploadScreen:
    def __init__(self, root, next_screen_callback, set_image_callback):
        self.root = root
        self.next_screen = next_screen_callback
        self.set_image = set_image_callback
        self.current_image = None
        
        # Create main frame with gradient background
        self.frame = tk.Frame(root)
        self.frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.create_gradient_bg()
        
        # Create content frame
        self.content_frame = tk.Frame(self.frame, bg='#ffffff')
        self.content_frame.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.8)
        
        # Add title
        self.title = tk.Label(
            self.content_frame,
            text="Upload Your Image",
            font=("Times New Roman", 24, "bold"),
            bg='#ffffff'
        )
        self.title.pack(pady=20)
        
        # Create image frame
        self.image_frame = tk.Frame(
            self.content_frame,
            bg='#f0f0f0',
            width=800,
            height=400
        )
        self.image_frame.pack(pady=(20,10), padx=20, expand=True)
        self.image_frame.pack_propagate(False)
        
        # Default message in image frame
        self.default_label = tk.Label(
            self.image_frame,
            text="No image selected\nClick 'Upload Image' to begin",
            font=("Times New Roman", 14),
            bg='#f0f0f0'
        )
        self.default_label.place(relx=0.5, rely=0.5, anchor='center')
        
        # Image label (hidden initially)
        self.image_label = tk.Label(self.image_frame, bg='#f0f0f0')
        
        # Create buttons frame
        self.buttons_frame = tk.Frame(self.content_frame, bg='#ffffff')
        self.buttons_frame.pack(fill='x', pady=(0,20), padx=20)
        
        # Upload button
        self.upload_btn = tk.Button(
            self.buttons_frame,
            text="Upload Image",
            command=self.upload_image,
            font=("Times New Roman", 14, "bold"),
            bg='#4CAF50',
            fg='white',
            relief='raised',
            padx=30,
            pady=15,
            width=18,
        )
        self.upload_btn.pack(side='left', padx=10)
        
        # Next button (disabled initially)
        self.next_btn = tk.Button(
            self.buttons_frame,
            text="Next →",
            command=self.on_next,
            font=("Times New Roman", 14, "bold"),
            bg='#2196F3',
            fg='white',
            relief='raised',
            padx=30,
            pady=15,
            width=18,
            state='disabled'
        )
        self.next_btn.pack(side='right', padx=10)
        
        # Bind hover effects
        self.bind_hover_effects()
        
    def create_gradient_bg(self):
        """Create a gradient background"""
        canvas = tk.Canvas(self.frame, highlightthickness=0)
        canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        # Create gradient
        colors = ['#e6f3ff', '#ffffff']
        height = self.root.winfo_screenheight()
        for i in range(len(colors)-1):
            canvas.create_rectangle(
                0, i*height/len(colors),
                self.root.winfo_screenwidth(), (i+1)*height/len(colors),
                fill=colors[i], outline=colors[i]
            )
            
    def bind_hover_effects(self):
        """Add hover effects to buttons"""
        def on_enter(e, btn, color):
            btn['background'] = color
            
        def on_leave(e, btn, color):
            btn['background'] = color
            
        self.upload_btn.bind("<Enter>", lambda e: on_enter(e, self.upload_btn, '#45a049'))
        self.upload_btn.bind("<Leave>", lambda e: on_leave(e, self.upload_btn, '#4CAF50'))
        
        # Next button hover effects
        def next_btn_enter(e):
            self.next_btn.configure(bg='#45a049', fg='black')
            
        def next_btn_leave(e):
            self.next_btn.configure(bg='#2196F3', fg='white')
            
        self.next_btn.bind("<Enter>", next_btn_enter)
        self.next_btn.bind("<Leave>", next_btn_leave)
        
    def upload_image(self):
        """Handle image upload"""
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff")
            ]
        )
        
        if file_path:
            # Load image using OpenCV
            self.current_image = cv2.imread(file_path)
            
            if self.current_image is not None:
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(rgb_image)
                
                # Resize image to fit frame while maintaining aspect ratio
                display_size = self.get_display_size(pil_image.size)
                pil_image = pil_image.resize(display_size, Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                self.photo = ImageTk.PhotoImage(pil_image)
                
                # Update display
                self.default_label.place_forget()
                self.image_label.configure(image=self.photo)
                self.image_label.place(relx=0.5, rely=0.5, anchor='center')
                
                # Enable next button
                self.next_btn.configure(state='normal')
                
                # Store image in parent
                self.set_image(self.current_image)
                
    def get_display_size(self, image_size):
        """Calculate display size maintaining aspect ratio"""
        # Get frame size
        frame_width = self.image_frame.winfo_width()
        frame_height = self.image_frame.winfo_height()
        
        # Calculate ratios
        width_ratio = frame_width / image_size[0]
        height_ratio = frame_height / image_size[1]
        
        # Use smaller ratio to fit image in frame
        ratio = min(width_ratio, height_ratio)
        
        return (int(image_size[0] * ratio), int(image_size[1] * ratio))
        
    def on_next(self):
        """Handle next button click with animation"""
        # Fade out animation
        def fade_out(alpha):
            if alpha > 0:
                self.content_frame.configure(bg=f'#{int(alpha*255):02x}{int(alpha*255):02x}{int(alpha*255):02x}')
                self.root.after(5, lambda: fade_out(alpha - 0.1))
            else:
                self.next_screen()
                
        fade_out(1.0)
