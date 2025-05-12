import tkinter as tk
from tkinter import ttk
import json

class ParameterScreen:
    def __init__(self, root, selected_filter, next_callback, back_callback, set_params_callback):
        self.root = root
        self.selected_filter = selected_filter
        self.next_screen = next_callback
        self.back_screen = back_callback
        self.set_params = set_params_callback
        
        # Dictionary to store parameter values
        self.param_values = {}
        
        # Define parameters for each filter type
        self.filter_params = {
            "Min Filter": {
                "Kernel Size": {"type": "slider", "range": (3, 15, 2), "default": 3},
                "Border Type": {"type": "combobox", "values": ["reflect", "constant", "replicate", "wrap"]}
            },
            "Max Filter": {
                "Kernel Size": {"type": "slider", "range": (3, 15, 2), "default": 3}, 
                "Border Type": {"type": "combobox", "values": ["reflect", "constant", "replicate", "wrap"]}
            },
            "Median Filter": {
                "Kernel Size": {"type": "slider", "range": (3, 15, 2), "default": 3},
                "Border Type": {"type": "combobox", "values": ["reflect", "constant", "replicate", "wrap"]}
            },
            "Mean Filter": {
                "Kernel Size": {"type": "slider", "range": (3, 15, 2), "default": 3},
                "Border Type": {"type": "combobox", "values": ["reflect", "constant", "replicate", "wrap"]}
            },
            "Laplacian": {
                "Kernel Size": {"type": "slider", "range": (1, 31, 2), "default": 3},
                "Border Type": {"type": "combobox", "values": ["reflect", "constant", "replicate", "wrap"]},
                "Scale": {"type": "slider", "range": (1, 10, 1), "default": 1},
                "Delta": {"type": "slider", "range": (0, 255, 1), "default": 0}
            },
            "Sobel": {
                "Kernel Size": {"type": "slider", "range": (3, 31, 2), "default": 3},
                "Direction": {"type": "combobox", "values": ["X", "Y", "Both"]},
                "Border Type": {"type": "combobox", "values": ["reflect", "constant", "replicate", "wrap"]},
                "Scale": {"type": "slider", "range": (1, 10, 1), "default": 1},
                "Delta": {"type": "slider", "range": (0, 255, 1), "default": 0}
            },
            "Ideal Low Pass": {
                "Cutoff Frequency": {"type": "slider", "range": (0, 100, 1), "default": 50}
            },
            "Butterworth Low Pass": {
                "Cutoff Frequency": {"type": "slider", "range": (0, 100, 1), "default": 50},
                "Order": {"type": "slider", "range": (1, 10, 1), "default": 2}
            },
            "Gaussian Low Pass": {
                "Sigma": {"type": "slider", "range": (0, 100, 1), "default": 50}
            },
            "Mexican hat": {
                "Kernel Size": {"type": "slider", "range": (3, 31, 2), "default": 3},
                "Border Type": {"type": "combobox", "values": ["reflect", "constant", "replicate", "wrap"]},
                "Sigma": {"type": "slider", "range": (1, 10, 1), "default": 1}
            },
            "Alpha-Trimmed Mean": {
                "Kernel Size": {"type": "slider", "range": (3, 31, 2), "default": 3},
                "Border Type": {"type": "combobox", "values": ["reflect", "constant", "replicate", "wrap"]},
                "Alpha": {"type": "slider", "range": (0, 100, 1), "default": 20}
            },
            "Ideal Band Reject": {
                "Cutoff Frequency": {"type": "slider", "range": (0, 100, 1), "default": 50},
                "Bandwidth": {"type": "slider", "range": (1, 50, 1), "default": 10}
            },
            "Butterworth Band Reject": {
                "Cutoff Frequency": {"type": "slider", "range": (0, 100, 1), "default": 50},
                "Order": {"type": "slider", "range": (1, 10, 1), "default": 2},
                "Bandwidth": {"type": "slider", "range": (1, 50, 1), "default": 10}
            },
            "Gaussian Band Reject": {
                "Cutoff Frequency": {"type": "slider", "range": (0, 100, 1), "default": 50},
                "Bandwidth": {"type": "slider", "range": (1, 50, 1), "default": 10}
            },
            "Ideal High Pass": {
                "Cutoff Frequency": {"type": "slider", "range": (0, 100, 1), "default": 50}
            },
            "Butterworth High Pass": {
                "Cutoff Frequency": {"type": "slider", "range": (0, 100, 1), "default": 50},
                "Order": {"type": "slider", "range": (1, 10, 1), "default": 2}
            },
            "Gaussian High Pass": {
                "Sigma": {"type": "slider", "range": (0, 100, 1), "default": 50}
            },
            "Geometric Mean": {
                "Kernel Size": {"type": "slider", "range": (3, 31, 2), "default": 3},
                "Border Type": {"type": "combobox", "values": ["reflect", "constant", "replicate", "wrap"]}
            },
            "Harmonic Mean": {
                "Kernel Size": {"type": "slider", "range": (3, 31, 2), "default": 3},
                "Border Type": {"type": "combobox", "values": ["reflect", "constant", "replicate", "wrap"]}
            },
            "Contraharmonic Mean": {
                "Kernel Size": {"type": "slider", "range": (3, 31, 2), "default": 3},
                "Border Type": {"type": "combobox", "values": ["reflect", "constant", "replicate", "wrap"]},
                "Q Parameter": {"type": "slider", "range": (-5, 5, 1), "default": 0}
            },
            "Midpoint": {
                "Kernel Size": {"type": "slider", "range": (3, 31, 2), "default": 3},
                "Border Type": {"type": "combobox", "values": ["reflect", "constant", "replicate", "wrap"]}
            }
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        self.main_frame = tk.Frame(self.root, bg='#f5f5f5')
        self.main_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        # Title frame
        title_frame = tk.Frame(self.main_frame, bg='#2196F3', height=80)
        title_frame.pack(fill='x', pady=(0, 20))
        
        title = tk.Label(
            title_frame,
            text=f"Parameters - {self.selected_filter}",
            font=("Helvetica", 24, "bold"),
            bg='#2196F3',
            fg='white'
        )
        title.pack(pady=20)
        
        # Parameters frame
        params_frame = tk.Frame(self.main_frame, bg='white')
        params_frame.pack(fill='both', expand=True, padx=50, pady=(0, 20))
        
        # Create parameter controls
        if self.selected_filter in self.filter_params:
            row = 0
            for param_name, param_config in self.filter_params[self.selected_filter].items():
                self.create_parameter_control(params_frame, param_name, param_config, row)
                row += 1
                
        # Navigation frame
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
        
        # Apply button
        self.apply_btn = tk.Button(
            nav_frame,
            text="Apply Filter →",
            command=self.on_apply,
            font=("Helvetica", 12),
            bg='#2196F3',
            fg='white',
            relief='flat',
            padx=20,
            pady=10
        )
        self.apply_btn.pack(side='right')
        
        # Bind hover effects
        self.bind_hover_effects()
        
    def create_parameter_control(self, parent, param_name, param_config, row):
        # Parameter label
        label = tk.Label(
            parent,
            text=param_name,
            font=("Helvetica", 12),
            bg='white'
        )
        label.grid(row=row, column=0, padx=20, pady=10, sticky='w')
        
        # Create control based on parameter type
        if param_config["type"] == "slider":
            start, end, step = param_config["range"]
            value = tk.IntVar(value=param_config["default"])
            
            # For kernel size parameters, ensure integer values
            if param_name == "Kernel Size":
                slider = ttk.Scale(
                    parent,
                    from_=start,
                    to=end,
                    orient='horizontal',
                    variable=value,
                    command=lambda x: value.set(int(float(x)//2)*2 + 1),  # Force odd integers
                    length=300
                )
            # For Scale, Delta, Order and Alpha parameters, ensure integer values
            elif param_name in ["Scale", "Delta", "Order", "Alpha"]:
                slider = ttk.Scale(
                    parent,
                    from_=start,
                    to=end,
                    orient='horizontal',
                    variable=value,
                    command=lambda x: value.set(int(float(x))),  # Force integers
                    length=300
                )
            else:
                slider = ttk.Scale(
                    parent,
                    from_=start,
                    to=end,
                    orient='horizontal',
                    variable=value,
                    length=300
                )
            slider.grid(row=row, column=1, padx=20, pady=10, sticky='w')
            
            # Value label
            value_label = tk.Label(
                parent,
                textvariable=value,
                font=("Helvetica", 10),
                bg='white'
            )
            value_label.grid(row=row, column=2, padx=5, pady=10)
            
            self.param_values[param_name] = value
            
        elif param_config["type"] == "combobox":
            value = tk.StringVar(value=param_config["values"][0])
            
            combo = ttk.Combobox(
                parent,
                values=param_config["values"],
                textvariable=value,
                state='readonly',
                width=20
            )
            combo.grid(row=row, column=1, padx=20, pady=10, sticky='w')
            
            self.param_values[param_name] = value
            
    def bind_hover_effects(self):
        def on_enter(e, btn, bg_color, fg_color):
            btn['background'] = bg_color
            btn['foreground'] = fg_color
            
        def on_leave(e, btn, bg_color, fg_color):
            btn['background'] = bg_color
            btn['foreground'] = fg_color
            
        self.back_btn.bind("<Enter>", lambda e: on_enter(e, self.back_btn, '#d0d0d0', '#333333'))
        self.back_btn.bind("<Leave>", lambda e: on_leave(e, self.back_btn, '#e0e0e0', '#333333'))
        
        self.apply_btn.bind("<Enter>", lambda e: on_enter(e, self.apply_btn, '#1976D2', 'white'))
        self.apply_btn.bind("<Leave>", lambda e: on_leave(e, self.apply_btn, '#2196F3', 'white'))
        
    def get_parameters(self):
        """Collect all parameter values"""
        params = {}
        for param_name, var in self.param_values.items():
            params[param_name] = var.get()
        return params
        
    def on_apply(self):
        """Handle apply button click"""
        # Collect parameters
        params = self.get_parameters()
        self.set_params(params)
        
        # Fade out animation
        def fade_out(alpha):
            if alpha > 0:
                self.main_frame.configure(bg=f'#{int(alpha*255):02x}{int(alpha*255):02x}{int(alpha*255):02x}')
                self.root.after(5, lambda: fade_out(alpha - 0.1))
            else:
                self.next_screen()
                
        fade_out(1.0) 