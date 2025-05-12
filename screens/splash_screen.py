import tkinter as tk
from PIL import Image, ImageTk
import time
import os
from itertools import count

class SplashScreen:
    def __init__(self, root, next_screen_callback):
        self.root = root
        self.next_screen = next_screen_callback
        
        # Create main frame
        self.frame = tk.Frame(root, bg='#121212')  # Darker background
        self.frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        # Create title label
        self.title = tk.Label(
            self.frame,
            text="Image Processing Filters Simulator",
            font=("Times New Roman", 24, "bold"),
            fg='#E0E0E0',  # Light gray text
            bg='#121212'
        )
        self.title.place(relx=0.5, rely=0.45, anchor='center')
        
        # Create subtitle
        self.subtitle = tk.Label(
            self.frame,
            text="Loading...",
            font=("Times New Roman", 14),
            fg='#757575',  # Medium gray text
            bg='#121212'
        )
        self.subtitle.place(relx=0.5, rely=0.55, anchor='center')
        
        # Create progress bar
        self.progress = tk.Canvas(
            self.frame, 
            width=400, 
            height=4, 
            bg='#1E1E1E',  # Slightly lighter than background
            highlightthickness=0
        )
        self.progress.place(relx=0.5, rely=0.65, anchor='center')
        
        # Progress bar fill
        self.progress_fill = self.progress.create_rectangle(0, 0, 0, 4, fill='#BB86FC')  # Purple accent color
        
        # Start animations
        self.animate_entrance()
        
    def animate_entrance(self):
        """Animate the entrance of elements"""
        # Start progress animation directly
        self.start_progress()
        
    def fade_in(self, widget, duration=1000, after_callback=None):
        """Fade in animation for widgets"""
        steps = 20
        step_time = duration / steps
        
        def update_alpha(step):
            alpha = step / steps
            # For Tkinter widgets, we need to use transparency differently
            # Convert alpha to hex format for transparency
            alpha_hex = int(alpha * 18)  # Adjusted for dark mode
            widget.configure(bg=f'#{alpha_hex:02x}{alpha_hex:02x}{alpha_hex:02x}')
            
            if step < steps:
                self.root.after(int(step_time), lambda: update_alpha(step + 1))
            elif after_callback:
                after_callback()
                
        update_alpha(0)
        
    def start_progress(self):
        """Animate the progress bar"""
        duration = 5000  # 5 seconds
        steps = 100
        step_time = duration / steps
        
        def update_progress(step):
            progress = step / steps
            width = 400 * progress
            self.progress.coords(self.progress_fill, 0, 0, width, 4)
            
            if step < steps:
                self.root.after(int(step_time), lambda: update_progress(step + 1))
            else:
                self.fade_out()
                
        update_progress(0)
        
    def fade_out(self):
        """Fade out animation before transitioning to next screen"""
        duration = 500  # 0.5 seconds
        steps = 20
        step_time = duration / steps
        
        def update_alpha(step):
            alpha = 1 - (step / steps)
            # Convert alpha to hex format for transparency
            alpha_hex = int(alpha * 18)  # Adjusted for dark mode
            self.frame.configure(bg=f'#{alpha_hex:02x}{alpha_hex:02x}{alpha_hex:02x}')
            
            if step < steps:
                self.root.after(int(step_time), lambda: update_alpha(step + 1))
            else:
                self.next_screen()
                
        update_alpha(0)
