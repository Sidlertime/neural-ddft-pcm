"""
plot_training.py
Utility script to visualize the training losses and curriculum phases from a training_log.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_training_log(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    # Load data
    df = pd.read_csv(csv_path)

    # Setup the figure
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    epochs = df['epoch']

    # Identify phase transitions
    phase_changes = df[df['phase'].diff() != 0]['epoch'].tolist()
    
    # 1. Total Loss Plot (Log Scale)
    axs[0].plot(epochs, df['loss_total'], label='Total Loss', color='black', linewidth=2)
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Loss (Log Scale)')
    axs[0].set_title('Training Loss Progression')
    axs[0].grid(True, which="both", ls="-", alpha=0.2)

    # 2. Component Losses
    axs[1].plot(epochs, df['loss_rho'], label=r'Density Loss ($\rho$)', color='blue')
    axs[1].plot(epochs, df['loss_v'], label='Velocity Loss ($v$)', color='red')
    axs[1].set_yscale('log')
    axs[1].set_ylabel('Component Loss')
    axs[1].legend()
    axs[1].grid(True, which="both", ls="-", alpha=0.2)

    # 3. Force Ratio (if hybrid)
    if df['force_ratio'].sum() > 0:
        axs[2].plot(epochs, df['force_ratio'], label=r'$||F_{sup}|| / ||F_{eq}||$', color='green')
        axs[2].set_ylabel('Force Ratio')
    else:
        axs[2].text(0.5, 0.5, 'Force Ratio Not Tracked', ha='center', va='center', transform=axs[2].transAxes)
    axs[2].set_xlabel('Global Epoch')
    axs[2].grid(True, alpha=0.2)

    # Draw vertical lines for curriculum phases across all subplots
    for ax in axs:
        for change_epoch in phase_changes[1:]: # Skip the first epoch
            ax.axvline(x=change_epoch, color='gray', linestyle='--', alpha=0.7)
            
        # Annotate phase parameters on the top plot
        for i, change_epoch in enumerate(phase_changes):
            phase_data = df[df['epoch'] == change_epoch].iloc[0]
            ws = int(phase_data['window_size'])
            lr = phase_data['lr']
            label = f"Phase {i+1}\nWS: {ws}\nLR: {lr:.0e}"
            axs[0].text(change_epoch + 0.5, axs[0].get_ylim()[1]*0.8, label, fontsize=9, 
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    
    # Save the plot in the same directory as the CSV
    save_path = os.path.dirname(csv_path)
    img_path = os.path.join(save_path, "training_progression.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to: {img_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Neural DDFT training metrics")
    parser.add_argument("csv_path", type=str, help="Path to the training_log.csv file")
    args = parser.parse_args()
    
    plot_training_log(args.csv_path)
