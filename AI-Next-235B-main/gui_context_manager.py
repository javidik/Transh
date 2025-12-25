import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import torch
import torch.nn as nn
import numpy as np
from context_management_layer import ContextManager
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

class ContextManagementGUI:
    """
    Graphical User Interface for the Eisenhower Context Management System
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Eisenhower Context Management System")
        self.root.geometry("1200x800")
        
        # Initialize the context manager
        self.d_model = 512
        self.context_manager = ContextManager(d_model=self.d_model, n_heads=8)
        self.context_manager.eval()  # Set to evaluation mode
        
        # Store context history
        self.chat_history = []
        self.context_analysis = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frames
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        middle_frame = ttk.Frame(self.root, padding="10")
        middle_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)
        
        # Top frame: Input controls
        ttk.Label(top_frame, text="Enter Chat Context:", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.input_text = scrolledtext.ScrolledText(top_frame, height=5, width=100)
        self.input_text.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Buttons
        button_frame = ttk.Frame(top_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        
        analyze_btn = ttk.Button(button_frame, text="Analyze Context", command=self.analyze_context)
        analyze_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        clear_btn = ttk.Button(button_frame, text="Clear History", command=self.clear_history)
        clear_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        extract_btn = ttk.Button(button_frame, text="Extract Important", command=self.extract_important)
        extract_btn.pack(side=tk.LEFT)
        
        # Middle frame: Visualization area
        ttk.Label(middle_frame, text="Context Analysis Results:", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Create notebook for different visualizations
        self.notebook = ttk.Notebook(middle_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Importance vs Urgency plot tab
        self.plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text="Importance vs Urgency")
        
        # Quadrant classification tab
        self.quadrant_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.quadrant_frame, text="Quadrant Classification")
        
        # Attention visualization tab
        self.attention_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.attention_frame, text="Attention Patterns")
        
        # Configure middle frame grid weight
        middle_frame.columnconfigure(0, weight=1)
        middle_frame.rowconfigure(1, weight=1)
        
        # Bottom frame: Results display
        ttk.Label(bottom_frame, text="Analysis Details:", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.results_display = scrolledtext.ScrolledText(bottom_frame, height=8, width=100)
        self.results_display.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.S), pady=(0, 10))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(bottom_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
    def analyze_context(self):
        """Analyze the input context using the Eisenhower Context Layer"""
        input_text = self.input_text.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showwarning("Warning", "Please enter some context to analyze.")
            return
            
        self.status_var.set("Analyzing context...")
        self.root.update_idletasks()
        
        try:
            # Simulate tokenization and embedding
            # In a real system, we would use actual tokenization and embedding
            tokens = input_text.split()
            seq_len = len(tokens)
            
            # Create dummy embeddings (in real system, these would come from the embedding layer)
            batch_size = 1
            embeddings = torch.randn(batch_size, seq_len, self.d_model)
            mask = torch.ones(batch_size, seq_len)
            
            # Process through the context manager
            with torch.no_grad():
                output, analysis = self.context_manager(embeddings, mask)
                
            # Store analysis results
            self.context_analysis = analysis
            
            # Update UI in a thread-safe way
            self.display_results(analysis, tokens)
            self.create_visualizations(analysis, tokens)
            
            self.status_var.set(f"Analysis complete. Sequence length: {seq_len} tokens")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis: {str(e)}")
            self.status_var.set("Error occurred")
    
    def display_results(self, analysis, tokens):
        """Display analysis results in the text area"""
        self.results_display.delete(1.0, tk.END)
        
        # Extract values
        importance_scores = analysis['importance_scores'][0].cpu().numpy()
        urgency_scores = analysis['urgency_scores'][0].cpu().numpy()
        quadrant_probs = analysis['quadrant_probabilities'][0].cpu().numpy()
        combined_importance = analysis['combined_importance'][0].cpu().numpy()
        
        # Format results
        results = "Context Analysis Results:\n"
        results += "="*50 + "\n\n"
        
        results += f"Total Tokens Analyzed: {len(tokens)}\n\n"
        
        results += "Token-by-Token Analysis:\n"
        results += "-"*30 + "\n"
        
        for i, (token, imp_score, urg_score, comb_imp) in enumerate(zip(tokens[:20], 
                                                                      importance_scores[:20], 
                                                                      urgency_scores[:20],
                                                                      combined_importance[:20])):
            results += f"{i+1:2d}. '{token[:15]}{'...' if len(token) > 15 else '':<15}' | "
            results += f"Imp: {imp_score:.3f} | Urg: {urg_score:.3f} | Comb: {comb_imp:.3f}\n"
        
        if len(tokens) > 20:
            results += f"\n... and {len(tokens) - 20} more tokens\n"
        
        results += "\nQuadrant Distribution:\n"
        results += "-"*20 + "\n"
        quad_names = ["Do First (Imp+Urg)", "Schedule (Imp+NotUrg)", "Delegate (NotImp+Urg)", "Eliminate (NotImp+NotUrg)"]
        
        avg_quadrant_probs = np.mean(quadrant_probs, axis=0)
        for i, (name, prob) in enumerate(zip(quad_names, avg_quadrant_probs)):
            results += f"{name}: {prob:.3f}\n"
        
        self.results_display.insert(tk.END, results)
    
    def create_visualizations(self, analysis, tokens):
        """Create visualizations for the analysis results"""
        # Clear existing plots
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        for widget in self.quadrant_frame.winfo_children():
            widget.destroy()
        for widget in self.attention_frame.winfo_children():
            widget.destroy()
        
        # Convert tensors to numpy arrays
        importance_scores = analysis['importance_scores'][0].cpu().numpy()
        urgency_scores = analysis['urgency_scores'][0].cpu().numpy()
        quadrant_probs = analysis['quadrant_probabilities'][0].cpu().numpy()
        attention_weights = analysis['context_relevance_attention'][0].cpu().numpy()
        
        # Create Importance vs Urgency scatter plot
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        scatter = ax1.scatter(importance_scores, urgency_scores, c=range(len(importance_scores)), cmap='viridis', alpha=0.7)
        ax1.set_xlabel('Importance Score')
        ax1.set_ylabel('Urgency Score')
        ax1.set_title('Importance vs Urgency Scatter Plot')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Token Index')
        
        canvas1 = FigureCanvasTkAgg(fig1, self.plot_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create quadrant classification visualization
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        avg_quadrant_probs = np.mean(quadrant_probs, axis=0)
        quad_names = ['Do First', 'Schedule', 'Delegate', 'Eliminate']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        bars = ax2.bar(quad_names, avg_quadrant_probs, color=colors, alpha=0.7)
        ax2.set_ylabel('Average Probability')
        ax2.set_title('Average Quadrant Classification Probabilities')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_quadrant_probs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom')
        
        canvas2 = FigureCanvasTkAgg(fig2, self.quadrant_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create attention pattern heatmap
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        im = ax3.imshow(attention_weights, cmap='Blues', aspect='auto')
        ax3.set_xlabel('Key Positions')
        ax3.set_ylabel('Query Positions')
        ax3.set_title('Context Relevance Attention Pattern')
        
        # Add colorbar
        cbar3 = plt.colorbar(im, ax=ax3)
        cbar3.set_label('Attention Weight')
        
        canvas3 = FigureCanvasTkAgg(fig3, self.attention_frame)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def extract_important(self):
        """Extract important context from the current analysis"""
        if not self.context_analysis:
            messagebox.showwarning("Warning", "Please analyze context first.")
            return
            
        try:
            # Extract important tokens
            with torch.no_grad():
                # Create dummy input matching the original sequence
                # In a real system, we would use the actual processed input
                batch_size, seq_len, d_model = 1, 100, self.d_model
                dummy_input = torch.randn(batch_size, seq_len, d_model)
                
                important_indices, important_embeddings = self.context_manager.extract_important_context(
                    dummy_input, self.context_analysis, top_k=10
                )
            
            # Display results
            self.results_display.delete(1.0, tk.END)
            self.results_display.insert(tk.END, "Important Context Extraction Results:\n")
            self.results_display.insert(tk.END, "="*50 + "\n\n")
            self.results_display.insert(tk.END, f"Extracted {important_embeddings.shape[1]} most important tokens/segments\n")
            self.results_display.insert(tk.END, f"Embedding dimension: {important_embeddings.shape[2]}\n")
            self.results_display.insert(tk.END, f"Sample important token indices: {important_indices[0][:5]}...\n")
            
            self.status_var.set(f"Extracted {important_embeddings.shape[1]} important elements")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during extraction: {str(e)}")
    
    def clear_history(self):
        """Clear the chat history and reset the UI"""
        self.chat_history = []
        self.context_analysis = {}
        self.input_text.delete(1.0, tk.END)
        self.results_display.delete(1.0, tk.END)
        self.status_var.set("History cleared")


def main():
    root = tk.Tk()
    app = ContextManagementGUI(root)
    
    # Start the GUI event loop
    root.mainloop()


if __name__ == "__main__":
    main()