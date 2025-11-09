import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D

class FourierSeriesVisualizer:
    def __init__(self):
        # Initialize signal parameters for three signals
        # Signal 1: period, type ('sine' or 'cosine'), amplitude
        self.signal1 = {'period': 8, 'type': 'sine', 'amplitude': 1.0}
        # Signal 2
        self.signal2 = {'period': 16, 'type': 'cosine', 'amplitude': 0.5}
        # Signal 3
        self.signal3 = {'period': 32, 'type': 'sine', 'amplitude': 0.3}
        
        # Store composite signal and Fourier coefficients
        self.composite_signal = None
        self.N0 = None
        self.fourier_coeffs = None
        
        # Store figure references for cleanup
        self.plot_figures = []

        # Create main UI window
        self.create_ui()
    
    def create_ui(self):
        """Create user interface for signal definition"""
        # Create figure for UI widgets with constrained layout for adaptive sizing
        self.fig_ui = plt.figure(figsize=(14, 8),constrained_layout=False)
        self.fig_ui.suptitle('Fourier Series Visualization - Signal Definition', fontsize=14, fontweight='bold')
        
        # Store references to text objects for adaptive resizing
        self.text_objects = []
        self.widget_labels = []

        # Adjusted horizontal positions for better spacing
        col1_x = 0.150
        col2_x = 0.4650
        col3_x = 0.780
        col_width = 0.18

        # ===== Signal 1 Widgets =====
        # Period textbox for signal 1
        ax_period1 = plt.axes([col1_x, 0.85, col_width, 0.04])
        self.period1_box = TextBox(ax_period1, 'Signal 1 Period N1:', initial=str(self.signal1['period']))
        self.period1_box.label.set_fontsize(10)
        self.widget_labels.append(self.period1_box.label)
        self.period1_box.on_submit(self.update_period1)
        
        # Signal type radio buttons for signal 1
        ax_type1 = plt.axes([col1_x, 0.70, col_width, 0.10])
        self.type1_radio = RadioButtons(ax_type1, ('sine', 'cosine'), active=0)
        self.type1_radio.on_clicked(self.update_type1)
        txt1 = plt.text(col1_x, 0.82, 'Signal 1 Type:', transform=self.fig_ui.transFigure, fontsize=10)
        self.text_objects.append(txt1)

        # Amplitude textbox for signal 1
        ax_amp1 = plt.axes([col1_x, 0.60, col_width, 0.04])
        self.amp1_box = TextBox(ax_amp1, 'Signal 1 Amplitude A1:', initial=str(self.signal1['amplitude']))
        self.amp1_box.label.set_fontsize(10)
        self.widget_labels.append(self.amp1_box.label)
        self.amp1_box.on_submit(self.update_amp1)
        
        # ===== Signal 2 Widgets =====
        # Period textbox for signal 2
        ax_period2 = plt.axes([col2_x, 0.85, col_width, 0.04])
        self.period2_box = TextBox(ax_period2, 'Signal 2 Period N2:', initial=str(self.signal2['period']))
        self.period2_box.label.set_fontsize(10)
        self.widget_labels.append(self.period2_box.label)
        self.period2_box.on_submit(self.update_period2)
        
        # Signal type radio buttons for signal 2
        ax_type2 = plt.axes([col2_x, 0.70, col_width, 0.10])
        self.type2_radio = RadioButtons(ax_type2, ('sine', 'cosine'), active=1)
        self.type2_radio.on_clicked(self.update_type2)
        txt2 = plt.text(col2_x, 0.82, 'Signal 2 Type:', transform=self.fig_ui.transFigure, fontsize=10)
        self.text_objects.append(txt2)

        # Amplitude textbox for signal 2
        ax_amp2 = plt.axes([col2_x, 0.60, col_width, 0.04])
        self.amp2_box = TextBox(ax_amp2, 'Signal 2 Amplitude A2:', initial=str(self.signal2['amplitude']))
        self.amp2_box.label.set_fontsize(10)
        self.widget_labels.append(self.amp2_box.label)
        self.amp2_box.on_submit(self.update_amp2)
        
        # ===== Signal 3 Widgets =====
        # Period textbox for signal 3
        ax_period3 = plt.axes([col3_x, 0.85, col_width, 0.04])
        self.period3_box = TextBox(ax_period3, 'Signal 3 Period N3:', initial=str(self.signal3['period']))
        self.period3_box.label.set_fontsize(10)
        self.widget_labels.append(self.period3_box.label)
        self.period3_box.on_submit(self.update_period3)
        
        # Signal type radio buttons for signal 3
        ax_type3 = plt.axes([col3_x, 0.70, col_width, 0.10])
        self.type3_radio = RadioButtons(ax_type3, ('sine', 'cosine'), active=0)
        self.type3_radio.on_clicked(self.update_type3)
        txt3 = plt.text(col3_x, 0.82, 'Signal 3 Type:', transform=self.fig_ui.transFigure, fontsize=10)
        self.text_objects.append(txt3)

        # Amplitude textbox for signal 3
        ax_amp3 = plt.axes([col3_x, 0.60, col_width, 0.04])
        self.amp3_box = TextBox(ax_amp3, 'Signal 3 Amplitude A3:', initial=str(self.signal3['amplitude']))
        self.amp3_box.label.set_fontsize(10)
        self.widget_labels.append(self.amp3_box.label)
        self.amp3_box.on_submit(self.update_amp3)
        
        # ===== Compute Button =====
        ax_compute = plt.axes([0.4, 0.45, 0.2, 0.06])
        self.compute_btn = Button(ax_compute, 'Compute Fourier Series', color='lightgreen', hovercolor='green')
        self.compute_btn.on_clicked(self.compute_and_visualize)
        
        # Display message box area for signal info
        ax_msg = plt.axes([0.1, 0.05, 0.8, 0.35])
        ax_msg.axis('off')
        self.msg_text = ax_msg.text(0.5, 0.5, 'Click "Compute Fourier Series" to generate signal', 
                                      ha='center', va='center', fontsize=11, wrap=True)
        self.text_objects.append(self.msg_text)

        # Store title text object
        self.title_text = self.fig_ui._suptitle
        # Connect resize event for adaptive text sizing
        self.fig_ui.canvas.mpl_connect('resize_event', self.on_resize)
        
        plt.show()
    
    def on_resize(self, event):
        """Handle window resize events to adjust text sizes adaptively"""
        if event is None:
            return
        
        # Get current figure size in inches
        fig_width, fig_height = self.fig_ui.get_size_inches()
        
        # Calculate scaling factor based on figure dimensions
        # Base size is 14x8 inches, scale proportionally
        width_scale = fig_width / 14.0
        height_scale = fig_height / 8.0
        scale = min(width_scale, height_scale)
        
        # Adjust title font size
        if self.title_text:
            new_title_size = int(14 * scale)
            self.title_text.set_fontsize(new_title_size)
        
        # Adjust widget label font sizes
        for label in self.widget_labels:
            new_size = int(10 * scale)
            label.set_fontsize(new_size)

        # Adjust text objects (including message text and type labels)
        for txt in self.text_objects:
            if txt == self.msg_text:
                new_size = int(11 * scale)
            else:
                new_size = int(10 * scale)
            txt.set_fontsize(new_size)
        
        # Adjust button label
        if hasattr(self, 'compute_btn') and self.compute_btn.label:
            new_size = int(10 * scale)
            self.compute_btn.label.set_fontsize(new_size)
        
        # Redraw the canvas
        self.fig_ui.canvas.draw_idle()


    # ===== Callback functions for Signal 1 =====
    def update_period1(self, text):
        """Update period of signal 1"""
        try:
            # Parse input text to integer
            value = int(text)
            # Ensure period is at least 1
            if value < 1:
                value = 1
            self.signal1['period'] = value
        except ValueError:
            # If invalid input, keep previous value
            pass
    
    def update_type1(self, label):
        """Update type of signal 1"""
        self.signal1['type'] = label
    
    def update_amp1(self, text):
        """Update amplitude of signal 1"""
        try:
            # Parse input text to float
            self.signal1['amplitude'] = float(text)
        except ValueError:
            # If invalid input, keep previous value
            pass
    
    # ===== Callback functions for Signal 2 =====
    def update_period2(self, text):
        """Update period of signal 2"""
        try:
            value = int(text)
            if value < 1:
                value = 1
            self.signal2['period'] = value
        except ValueError:
            pass
    
    def update_type2(self, label):
        """Update type of signal 2"""
        self.signal2['type'] = label
    
    def update_amp2(self, text):
        """Update amplitude of signal 2"""
        try:
            self.signal2['amplitude'] = float(text)
        except ValueError:
            pass
    
    # ===== Callback functions for Signal 3 =====
    def update_period3(self, text):
        """Update period of signal 3"""
        try:
            value = int(text)
            if value < 1:
                value = 1
            self.signal3['period'] = value
        except ValueError:
            pass
    
    def update_type3(self, label):
        """Update type of signal 3"""
        self.signal3['type'] = label
    
    def update_amp3(self, text):
        """Update amplitude of signal 3"""
        try:
            self.signal3['amplitude'] = float(text)
        except ValueError:
            pass
    
    def gcd(self, a, b):
        """Compute greatest common divisor using Euclidean algorithm"""
        # Base case: if b is zero, return a
        if b == 0:
            return a
        # Recursive case: gcd(a,b) = gcd(b, a mod b)
        return self.gcd(b, a % b)
    
    def lcm(self, a, b):
        """Compute least common multiple"""
        # LCM formula: lcm(a,b) = (a*b) / gcd(a,b)
        return (a * b) // self.gcd(a, b)
    
    def generate_composite_signal(self):
        """Generate composite periodic signal x[n] from three sinusoids"""
        # Step 1: compute fundamental period N0 = LCM(N1, N2, N3)
        N1 = self.signal1['period']
        N2 = self.signal2['period']
        N3 = self.signal3['period']
        self.N0 = self.lcm(self.lcm(N1, N2), N3)
        
        # Step 2: generate time samples n = 0, 1, 2, ..., N0-1
        n = np.arange(self.N0)
        # Step 3: initialize composite signal
        x_n = np.zeros(self.N0)
        
        # Step 4: add signal 1 component
        omega1 = 2 * np.pi / N1  # fundamental frequency of signal 1
        if self.signal1['type'] == 'sine':
            x_n += self.signal1['amplitude'] * np.sin(omega1 * n)
        else:
            x_n += self.signal1['amplitude'] * np.cos(omega1 * n)
        
        # Step 5: add signal 2 component
        omega2 = 2 * np.pi / N2  # fundamental frequency of signal 2
        if self.signal2['type'] == 'sine':
            x_n += self.signal2['amplitude'] * np.sin(omega2 * n)
        else:
            x_n += self.signal2['amplitude'] * np.cos(omega2 * n)
        
        # Step 6: add signal 3 component
        omega3 = 2 * np.pi / N3  # fundamental frequency of signal 3
        if self.signal3['type'] == 'sine':
            x_n += self.signal3['amplitude'] * np.sin(omega3 * n)
        else:
            x_n += self.signal3['amplitude'] * np.cos(omega3 * n)
        
        # Store composite signal
        self.composite_signal = x_n
        return n, x_n
    
    def compute_fourier_coefficients(self):
        """Compute Fourier series coefficients X[k]"""
        # Step 1: number of frequency components to compute
        K = self.N0
        # Step 2: initialize arrays for coefficients
        k_values = np.arange(-K//2, K//2 + 1)  # frequency indices from -N0/2 to N0/2
        X_real = np.zeros(len(k_values))  # real part of X[k]
        X_imag = np.zeros(len(k_values))  # imaginary part of X[k]
        X_mag = np.zeros(len(k_values))   # magnitude |X[k]|
        
        # Step 3: compute fundamental frequency
        omega0 = 2 * np.pi / self.N0
        
        # Step 4: compute DFT for each frequency k
        for idx, k in enumerate(k_values):
            # Step 4a: initialize accumulator for X[k]
            X_k_real = 0
            X_k_imag = 0
            
            # Step 4b: sum over all time samples n
            for n in range(self.N0):
                # Step 4c: compute angle -k*omega0*n
                angle = -k * omega0 * n
                
                # Step 4d: Euler's formula: exp(j*angle) = cos(angle) + j*sin(angle)
                # X[k] = (1/N0) * sum(x[n] * exp(-j*k*omega0*n))
                X_k_real += self.composite_signal[n] * np.cos(angle)
                X_k_imag += self.composite_signal[n] * np.sin(angle)
            
            # Step 4e: normalize by N0
            X_k_real /= self.N0
            X_k_imag /= self.N0
            
            # Step 4f: store results
            X_real[idx] = X_k_real
            X_imag[idx] = X_k_imag
            X_mag[idx] = np.sqrt(X_k_real**2 + X_k_imag**2)
        
        # Store Fourier coefficients
        self.fourier_coeffs = {
            'k': k_values,
            'real': X_real,
            'imag': X_imag,
            'magnitude': X_mag
        }
        
        return k_values, X_real, X_imag, X_mag
    
    def decompose_sine_cosine(self):
        """Decompose Fourier series into sine and cosine components"""
        # Extract Fourier coefficients
        k_values = self.fourier_coeffs['k']
        X_real = self.fourier_coeffs['real']
        X_imag = self.fourier_coeffs['imag']
        
        # Initialize lists for sine and cosine components
        sine_components = []
        cosine_components = []
        
        # Step 1: find zero frequency index (DC component)
        zero_idx = np.where(k_values == 0)[0][0]
        
        # Step 2: DC component (k=0) - only add if non-zero
        if np.abs(X_real[zero_idx]) > 1e-10:
            cosine_components.append({
                'k': 0,
                'amplitude': X_real[zero_idx],
                'frequency': 0
            })
        
        # Step 3: process positive frequencies only (k > 0)
        for idx, k in enumerate(k_values):
            if k > 0:
                # Step 3a: compute fundamental frequency for this harmonic
                omega_k = 2 * np.pi * k / self.N0
                
                # Step 3b: cosine coefficient a_k = 2*Re{X[k]}
                a_k = 2 * X_real[idx]
                
                # Step 3c: sine coefficient b_k = -2*Im{X[k]}
                b_k = -2 * X_imag[idx]
                
                # Step 3d: store non-zero cosine component
                if np.abs(a_k) > 1e-10:
                    cosine_components.append({
                        'k': k,
                        'amplitude': a_k,
                        'frequency': omega_k
                    })
                
                # Step 3e: store non-zero sine component
                if np.abs(b_k) > 1e-10:
                    sine_components.append({
                        'k': k,
                        'amplitude': b_k,
                        'frequency': omega_k
                    })
        
        return sine_components, cosine_components
    
    def close_previous_plots(self):
        """Close all previous plot windows"""
        for fig in self.plot_figures:
            try:
                plt.close(fig)
            except:
                pass
        self.plot_figures = []

    def plot_3d_visualization(self):
        """Create 3D visualization plots"""
        # Generate time samples for plotting
        n = np.arange(self.N0)
        
        # Decompose into sine and cosine components
        sine_comps, cosine_comps = self.decompose_sine_cosine()
        
        # ===== Figure 1: Sine Components =====
        fig1 = plt.figure(figsize=(14, 10))
        self.plot_figures.append(fig1)
        ax1 = fig1.add_subplot(111, projection='3d')
        
        # Step 1: Plot magnitude spectrum on xOz plane (y=0)
        k_vals = self.fourier_coeffs['k']
        mag_vals = self.fourier_coeffs['magnitude']

        # Use stem plot for X[k] magnitude spectrum
        for i, (k, mag) in enumerate(zip(k_vals, mag_vals)):
            if mag > 1e-10:
                ax1.plot([k, k], [0, 0], [0, mag], 'b-', linewidth=2)
                ax1.scatter([k], [0], [mag], c='b', marker='o', s=50)
        
        # Step 2: Plot composite signal x[n] on yOz plane (x=0)
        ax1.plot(np.zeros_like(n), n, self.composite_signal, 'r-', linewidth=2, label='x[n]')
        
        # Step 3: Plot each sine component at its frequency axis
        for comp in sine_comps:
            # Generate sine wave for this component
            k = comp['k']
            amp = comp['amplitude']
            omega = comp['frequency']
            sine_wave = amp * np.sin(omega * n)
            # Plot sine component at x = k
            ax1.plot(k * np.ones_like(n), n, sine_wave, 'g-', linewidth=1.5, alpha=0.7)
        
        # Configure plot
        ax1.set_xlabel('Frequency Index k', fontsize=12)
        ax1.set_ylabel('Sample n', fontsize=12)
        ax1.set_zlabel('Amplitude', fontsize=12)
        ax1.set_title('Sine Components Visualization\nxOz: Magnitude Spectrum | yOz: Signal x[n] | Vertical: Sine Components', 
                      fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True)
        
        # ===== Figure 2: Cosine Components =====
        fig2 = plt.figure(figsize=(14, 10))
        self.plot_figures.append(fig2)
        ax2 = fig2.add_subplot(111, projection='3d')
        
        # Use stem plot for X[k] magnitude spectrum
        for i, (k, mag) in enumerate(zip(k_vals, mag_vals)):
            if mag > 1e-10:
                ax2.plot([k, k], [0, 0], [0, mag], 'b-', linewidth=2)
                ax2.scatter([k], [0], [mag], c='b', marker='o', s=50)
        
        # Step 2: Plot composite signal x[n] on yOz plane (x=0)
        ax2.plot(np.zeros_like(n), n, self.composite_signal, 'r-', linewidth=2, label='x[n]')
        
        # Step 3: Plot each cosine component at its frequency axis
        for comp in cosine_comps:
            # Generate cosine wave for this component
            k = comp['k']
            amp = comp['amplitude']
            omega = comp['frequency']
            
            # Handle DC component separately
            if k == 0:
                continue
            else:
                cosine_wave = amp * np.cos(omega * n)
                # Plot cosine component at x = k
                ax2.plot(k * np.ones_like(n), n, cosine_wave, 'm-', linewidth=1.5, alpha=0.7)
        
        # Configure plot
        ax2.set_xlabel('Frequency Index k', fontsize=12)
        ax2.set_ylabel('Sample n', fontsize=12)
        ax2.set_zlabel('Amplitude', fontsize=12)
        ax2.set_title('Cosine Components Visualization\nxOz: Magnitude Spectrum | yOz: Signal x[n] | Vertical: Cosine Components', 
                      fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True)
        
        plt.show()
    
    def compute_and_visualize(self, event):
        """Main computation and visualization routine"""
        #Step 1: close previous plot windows
        self.close_previous_plots()

        # Step 2: generate composite signal
        n, x_n = self.generate_composite_signal()
        
        # Step 3: compute Fourier coefficients
        k, X_real, X_imag, X_mag = self.compute_fourier_coefficients()
        
        # Step 4: update message box with signal information
        msg = f"Composite Signal Generated!\n\n"
        msg += f"Signal 1: {self.signal1['type']}, N1={self.signal1['period']}, A1={self.signal1['amplitude']}\n"
        msg += f"Signal 2: {self.signal2['type']}, N2={self.signal2['period']}, A2={self.signal2['amplitude']}\n"
        msg += f"Signal 3: {self.signal3['type']}, N3={self.signal3['period']}, A3={self.signal3['amplitude']}\n\n"
        msg += f"Fundamental Period: N0 = LCM(N1,N2,N3) = {self.N0}\n"
        msg += f"Fundamental Frequency: ω0 = 2π/{self.N0} = {2*np.pi/self.N0:.4f} rad/sample\n\n"
        msg += f"Number of Fourier Coefficients: {len(k)}\n"
        msg += f"Check the 3D plots for visualization!"
        
        self.msg_text.set_text(msg)
        self.fig_ui.canvas.draw()
        
        # Step 4: create 3D visualization plots
        self.plot_3d_visualization()


# ===== Main Program Entry Point =====
if __name__ == "__main__":
    # Create and run the Fourier Series Visualizer
    visualizer = FourierSeriesVisualizer()