import tkinter as tk
from tkinter import ttk
import json
import os

class FilterSelectionScreen:
    def __init__(self, root, next_callback, back_callback, set_filter_callback):
        self.root = root
        self.next_screen = next_callback
        self.back_screen = back_callback
        self.set_filter = set_filter_callback
        self.selected_filter = None
        self.selected_button = None
        self.hovered_button = None
        
        # Filter categories and their filters
        self.filter_categories = {
            "Statistical Filters": [
                "Min Filter", "Max Filter", "Median Filter", "Mean Filter",
                "Geometric Mean", "Harmonic Mean", "Contraharmonic Mean",
                "Midpoint", "Alpha-Trimmed Mean"
            ],
            "Edge Detection": [
                "Laplacian", "Sobel", "Mexican hat"
            ],
            "Frequency Domain - Low Pass": [
                "Ideal Low Pass", "Butterworth Low Pass", "Gaussian Low Pass"
            ],
            "Frequency Domain - High Pass": [
                "Ideal High Pass", "Butterworth High Pass", "Gaussian High Pass"
            ],
            "Frequency Domain - Band Reject": [
                "Ideal Band Reject", "Butterworth Band Reject", "Gaussian Band Reject"
            ]
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        self.main_frame = tk.Frame(self.root, bg='#f5f5f5')
        self.main_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        # Title
        title_frame = tk.Frame(self.main_frame, bg='#2196F3', height=80)
        title_frame.pack(fill='x', pady=(0, 20))
        
        title = tk.Label(
            title_frame,
            text="Select Filter",
            font=("Helvetica", 24, "bold"),
            bg='#2196F3',
            fg='white'
        )
        title.pack(pady=20)
        
        # Create content frame
        content_frame = tk.Frame(self.main_frame, bg='#f5f5f5')
        content_frame.pack(fill='both', expand=True, padx=50, pady=(0, 20))
        
        # Create category frames
        row = 0
        col = 0
        for category, filters in self.filter_categories.items():
            category_frame = self.create_category_frame(content_frame, category, filters)
            category_frame.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
            
            col += 1
            if col > 2:  # 3 columns layout
                col = 0
                row += 1
                
        # Configure grid
        for i in range(3):
            content_frame.grid_columnconfigure(i, weight=1)
            
        # Navigation buttons frame
        nav_frame = tk.Frame(self.main_frame, bg='#f5f5f5')
        nav_frame.pack(fill='x', padx=50, pady=20)
        
        # Back button
        self.back_btn = tk.Button(
            nav_frame,
            text="← Back",
            command=self.back_screen,
            font=("Helvetica", 12),
            bg='#e0e0e0',
            fg='#333333',
            relief='flat',
            padx=20,
            pady=10
        )
        self.back_btn.pack(side='left')
        
        # Next button
        self.next_btn = tk.Button(
            nav_frame,
            text="Next →",
            command=self.on_next,
            font=("Helvetica", 12),
            bg='#2196F3',
            fg='white',
            relief='flat',
            padx=20,
            pady=10,
            state='disabled'
        )
        self.next_btn.pack(side='right')
        
        # Bind hover effects
        self.bind_hover_effects()
        
    def create_category_frame(self, parent, category, filters):
        frame = tk.LabelFrame(
            parent,
            text=category,
            font=("Helvetica", 12, "bold"),
            bg='white',
            fg='#333333',
            relief='solid',
            bd=1
        )
        
        for i, filter_name in enumerate(filters):
            btn = tk.Button(
                frame,
                text=filter_name,
                command=lambda f=filter_name: self.select_filter(f),
                font=("Helvetica", 10),
                bg='#f8f9fa',
                fg='#333333',
                relief='flat',
                padx=10,
                pady=5,
                width=25,
                anchor='w'
            )
            btn.pack(pady=2, padx=5, fill='x')
            
            # Add visual indicator if this is the selected filter
            if filter_name == self.selected_filter:
                btn.configure(
                    bg='#4CAF50',  # Green background
                    fg='white',    # White text
                    relief='solid',
                    font=("Helvetica", 10, "bold"),  # Bold font
                    borderwidth=2  # Thicker border
                )
                self.selected_button = btn
                self.hovered_button = btn
            
            # Bind hover effect
            btn.bind("<Enter>", lambda e, b=btn: self.on_button_hover(b))
            btn.bind("<Leave>", lambda e, b=btn: self.on_button_leave(b))
            
        return frame
    
    def select_filter(self, filter_name):
        # Reset previous selection
        if self.selected_button:
            self.selected_button.configure(
                bg='#f8f9fa',
                fg='#333333',
                relief='flat',
                font=("Helvetica", 10),
                borderwidth=1
            )
            
        # Find and highlight selected button
        for widget in self.main_frame.winfo_children():
            if isinstance(widget, tk.Frame):  # This is the content_frame
                for category_frame in widget.winfo_children():
                    if isinstance(category_frame, tk.LabelFrame):
                        for btn in category_frame.winfo_children():
                            if isinstance(btn, tk.Button) and btn['text'] == filter_name:
                                btn.configure(
                                    bg='#4CAF50',  # Green background
                                    fg='white',    # White text
                                    relief='solid',
                                    font=("Helvetica", 10, "bold"),
                                    borderwidth=2
                                )
                                self.selected_button = btn
                                self.hovered_button = btn
                        
        self.selected_filter = filter_name
        self.set_filter(filter_name)
        self.next_btn.configure(state='normal')
        
    def bind_hover_effects(self):
        def on_enter(e, btn, bg_color, fg_color):
            btn['background'] = bg_color
            btn['foreground'] = fg_color
            
        def on_leave(e, btn, bg_color, fg_color):
            btn['background'] = bg_color
            btn['foreground'] = fg_color
            
        self.back_btn.bind("<Enter>", lambda e: on_enter(e, self.back_btn, '#d0d0d0', '#333333'))
        self.back_btn.bind("<Leave>", lambda e: on_leave(e, self.back_btn, '#e0e0e0', '#333333'))
        
        self.next_btn.bind("<Enter>", lambda e: on_enter(e, self.next_btn, '#1976D2', 'white'))
        self.next_btn.bind("<Leave>", lambda e: on_leave(e, self.next_btn, '#2196F3', 'white'))
        
    def on_button_hover(self, button):
        if button != self.selected_button:  # If not selected
            button.configure(bg='#e9ecef', relief='raised')  # Add raised effect on hover
        if self.hovered_button and self.hovered_button != button:
            self.on_button_leave(self.hovered_button)
        self.hovered_button = button
            
    def on_button_leave(self, button):
        if button != self.selected_button:  # If not selected
            button.configure(bg='#f8f9fa', relief='flat')
            if button == self.hovered_button:
                self.hovered_button = None
            
    def on_next(self):
        """Handle next button click with animation"""
        def fade_out(alpha):
            if alpha > 0:
                self.main_frame.configure(bg=f'#{int(alpha*255):02x}{int(alpha*255):02x}{int(alpha*255):02x}')
                self.root.after(5, lambda: fade_out(alpha - 0.1))
            else:
                self.next_screen()
                
        fade_out(1.0)
