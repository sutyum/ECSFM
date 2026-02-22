import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys

# Import the simulator from our newly refactored core module
from ecsfm.sim.cv import simulate_cv

def build_dashboard(args):
    """
    Runs the CV Simulation and builds the interactive Matplotlib Dashboard.
    """
    print(f"Running simulation with {args.scan_rate} V/s scan rate...")
    x, C_ox, C_red, E_full, I_full, _ = simulate_cv(
        D_ox=args.d_ox,
        D_red=args.d_red,
        C_bulk_ox=args.c_ox,
        C_bulk_red=args.c_red,
        E0=args.e0,
        k0=args.k0,
        alpha=args.alpha,
        scan_rate=args.scan_rate,
        E_start=args.e_start,
        E_vertex=args.e_vertex,
        L=args.length,
        nx=args.nx
    )
    
    # Beautiful rich colors
    bg_color = '#121212'
    text_color = '#e0e0e0'
    line_ox = '#ff4b4b'
    line_red = '#4b9fff'
    line_cv = '#ffd166'
    
    fig = plt.figure(figsize=(12, 6), facecolor=bg_color)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    
    # Left plot: Concentration profiles
    ax1 = fig.add_subplot(gs[0], facecolor=bg_color)
    ax1.set_title("Concentration Profile", color=text_color, fontsize=14, pad=15)
    ax1.set_xlabel("Distance from Electrode [cm]", color=text_color)
    ax1.set_ylabel("Concentration [mM]", color=text_color)
    ax1.tick_params(colors=text_color)
    for spine in ax1.spines.values():
        spine.set_color('#333333')
        
    ax1.set_xlim(0, x.max() / 5) # zoom in near electrode
    ax1.set_ylim(-0.1, max(args.c_ox, args.c_red) * 1.2)
    
    line_ox_plot, = ax1.plot(x, C_ox[0], color=line_ox, lw=3, label="Oxidized ($C_{ox}$)")
    line_red_plot, = ax1.plot(x, C_red[0], color=line_red, lw=3, label="Reduced ($C_{red}$)")
    ax1.legend(loc="upper right", facecolor=bg_color, edgecolor='#333333', labelcolor=text_color)
    
    # Right plot: CV Curve
    ax2 = fig.add_subplot(gs[1], facecolor=bg_color)
    ax2.set_title("Cyclic Voltammogram", color=text_color, fontsize=14, pad=15)
    ax2.set_xlabel("Potential / V", color=text_color)
    ax2.set_ylabel(r"Current Density / $mA \cdot cm^{-2}$", color=text_color)
    ax2.tick_params(colors=text_color)
    for spine in ax2.spines.values():
        spine.set_color('#333333')
        
    ax2.plot(E_full, I_full, color='#444444', lw=2, alpha=0.5) # The full track
    line_cv_plot, = ax2.plot(E_full[:1], I_full[:1], color=line_cv, lw=3) # The history track up to current time
    current_point, = ax2.plot(E_full[0], I_full[0], 'o', color=text_color, markersize=8)
    
    plt.subplots_adjust(bottom=0.25, wspace=0.3)
    
    # Add a slider
    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor='#333333')
    slider = Slider(
        ax=ax_slider,
        label='Time Stepper',
        valmin=0,
        valmax=len(C_ox) - 1,
        valinit=0,
        valstep=1,
        color=line_cv
    )
    
    slider.label.set_color(text_color)
    slider.valtext.set_color(text_color)
    
    # Scale index mapping from thin history to full history
    if len(C_ox) > 1:
        scale_factor = (len(E_full) - 1) / (len(C_ox) - 1)
    else:
        scale_factor = 1.0

    def update(val):
        idx = int(slider.val)
        full_idx = min(int(round(idx * scale_factor)), len(E_full) - 1)
        
        line_ox_plot.set_ydata(C_ox[idx])
        line_red_plot.set_ydata(C_red[idx])
        
        line_cv_plot.set_data(E_full[:full_idx+1], I_full[:full_idx+1])
        current_point.set_data([E_full[full_idx]], [I_full[full_idx]])
        
        fig.canvas.draw_idle()
        
    slider.on_changed(update)
    
    print("Simulation complete! Building interactive plot...")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cyclic Voltammetry Interactive Dashboard")
    parser.add_argument("--d-ox", type=float, default=1e-5, help="Diffusion coefficient of Ox (cm^2/s)")
    parser.add_argument("--d-red", type=float, default=1e-5, help="Diffusion coefficient of Red (cm^2/s)")
    parser.add_argument("--c-ox", type=float, default=1.0, help="Bulk concentration of Ox (mM)")
    parser.add_argument("--c-red", type=float, default=0.0, help="Bulk concentration of Red (mM)")
    parser.add_argument("--e0", type=float, default=0.0, help="Formal potential (V)")
    parser.add_argument("--k0", type=float, default=0.01, help="Standard rate constant (cm/s)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Charge transfer coefficient")
    parser.add_argument("--scan-rate", type=float, default=0.1, help="Scan rate (V/s)")
    parser.add_argument("--e-start", type=float, default=0.5, help="Start potential (V)")
    parser.add_argument("--e-vertex", type=float, default=-0.5, help="Vertex potential (V)")
    parser.add_argument("--length", type=float, default=0.05, help="Simulation domain length (cm)")
    parser.add_argument("--nx", type=int, default=200, help="Number of spatial grid points")

    args = parser.parse_args()
    
    try:
        build_dashboard(args)
    except KeyboardInterrupt:
        sys.exit(0)
