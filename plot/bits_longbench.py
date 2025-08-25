import matplotlib.pyplot as plt
import numpy as np

# Set style for better aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'

# Data
# MSE data
mse_bit = [7.55, 7.39, 7.11, 6.89, 6.54, 6.2, 5.88, 5.45, 5.06, 4.98, 4.72, 4.46]
mse_longbench = [30.46, 30.41, 30.4, 30.22, 29.99, 29.68, 28.24, 25.44, 20.18, 15.42, 6.54, 5.99]

# IPW data
ipw_bit = [7.48, 7.29, 6.85, 6.47, 6.09, 5.91, 5.64, 5.11]
ipw_longbench = [30.39, 30.37, 30.41, 30.25, 30.21, 29.99, 27.72, 23.48]

# Create figure with high DPI for better quality
fig, ax = plt.subplots(figsize=(14, 9), dpi=100)

# Plot lines with enhanced styling and different patterns for better distinction
line1 = ax.plot(mse_bit, mse_longbench, 
                marker='o', linewidth=4, markersize=10, 
                label='MSE', color='#2980b9', markerfacecolor='#85c1e9',
                markeredgecolor='#1f4e79', markeredgewidth=2,
                linestyle='-', alpha=0.9, zorder=3)

line2 = ax.plot(ipw_bit, ipw_longbench, 
                marker='^', linewidth=4, markersize=10, 
                label='IPW', color='#c0392b', markerfacecolor='#f1948a',
                markeredgecolor='#7d2818', markeredgewidth=2,
                linestyle='--', alpha=0.9, zorder=3)  # Different line style

# Enhanced title and labels
ax.set_title('Performance Comparison: MSE vs IPW on LongBench', 
             fontsize=20, fontweight='bold', pad=25, color='#2c3e50')
ax.set_xlabel('Bit Precision', fontsize=16, fontweight='bold', color='#34495e')
ax.set_ylabel('LongBench Score', fontsize=16, fontweight='bold', color='#34495e')

# Enhanced legend (will be updated after adding reference lines)
legend = ax.legend(fontsize=12, loc='center right', frameon=True, 
                   fancybox=True, shadow=True, framealpha=0.95,
                   edgecolor='#bdc3c7', facecolor='white')
legend.get_frame().set_linewidth(1.5)

# Enhanced grid
ax.grid(True, linestyle='--', alpha=0.4, color='#95a5a6', linewidth=1)
ax.set_axisbelow(True)

# Set axis limits and ticks - zoom in on the high performance region
ax.set_xlim(4.2, 7.8)
ax.set_ylim(20, 32)  # Focus on the high performance range

# Custom tick styling
ax.tick_params(axis='both', which='major', labelsize=12, colors='#2c3e50')
ax.tick_params(axis='both', which='major', length=6, width=1.2)

# Set custom ticks with finer granularity
x_ticks = np.arange(4.5, 8, 0.25)  # More frequent x-axis ticks
y_ticks = np.arange(20, 32.5, 1)   # Finer y-axis granularity
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)

# Add subtle background color zones - adjusted for new y-range
ax.axhspan(30.5, 32, alpha=0.08, color='#27ae60', label='Excellent Performance')
ax.axhspan(29, 30.5, alpha=0.08, color='#f39c12', label='Good Performance')
ax.axhspan(20, 29, alpha=0.08, color='#e67e22', label='Moderate Performance')

# Add reference lines for baseline models
fp_full_score = 30.38
sageattn_8bit_score = 29.93

# Add horizontal reference lines with enhanced styling
ax.axhline(y=fp_full_score, color='#27ae60', linestyle='-.', linewidth=3, alpha=0.8, 
           label='FP Full Attention', zorder=1)
ax.axhline(y=sageattn_8bit_score, color='#8e44ad', linestyle=':', linewidth=3, alpha=0.8, 
           label='SageAttn 8-bit', zorder=1)

# Add text annotations for reference lines with better positioning
ax.text(4.3, fp_full_score + 0.15, f'FP Full Attention: {fp_full_score}', 
        fontsize=11, fontweight='bold', color='#27ae60', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor='#d5f4e6', alpha=0.9, edgecolor='#27ae60'),
        ha='left', va='bottom')

ax.text(4.3, sageattn_8bit_score - 0.25, f'SageAttn 8-bit: {sageattn_8bit_score}', 
        fontsize=11, fontweight='bold', color='#8e44ad',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='#f4ecf7', alpha=0.9, edgecolor='#8e44ad'),
        ha='left', va='top')

# Add value labels at high bit precision points for better distinction
for i, (x, y) in enumerate(zip(mse_bit[-4:], mse_longbench[-4:])):
    if x >= 6.5:  # Only for high bit precision
        ax.annotate(f'{y:.2f}', 
                    xy=(x, y), xytext=(5, 8),
                    textcoords='offset points', ha='center',
                    fontsize=9, color='#1f4e79', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='#ebf3fd', alpha=0.8))

for i, (x, y) in enumerate(zip(ipw_bit[-4:], ipw_longbench[-4:])):
    if x >= 6.5:  # Only for high bit precision  
        ax.annotate(f'{y:.2f}', 
                    xy=(x, y), xytext=(5, -12),
                    textcoords='offset points', ha='center',
                    fontsize=9, color='#7d2818', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='#fdf2f2', alpha=0.8))

# MSE peak annotation - adjusted for new scale
max_mse_idx = mse_longbench.index(max(mse_longbench))
ax.annotate(f'MSE Peak\n({mse_bit[max_mse_idx]:.2f}, {mse_longbench[max_mse_idx]:.2f})', 
            xy=(mse_bit[max_mse_idx], mse_longbench[max_mse_idx]),
            xytext=(mse_bit[max_mse_idx]-0.8, mse_longbench[max_mse_idx]+0.8),
            arrowprops=dict(arrowstyle='->', color='#1f4e79', alpha=0.8, lw=2),
            fontsize=10, ha='center', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#ebf3fd', alpha=0.9, edgecolor='#1f4e79'))

# IPW peak annotation - adjusted for new scale
max_ipw_idx = ipw_longbench.index(max(ipw_longbench))
ax.annotate(f'IPW Peak\n({ipw_bit[max_ipw_idx]:.2f}, {ipw_longbench[max_ipw_idx]:.2f})', 
            xy=(ipw_bit[max_ipw_idx], ipw_longbench[max_ipw_idx]),
            xytext=(ipw_bit[max_ipw_idx]+0.5, ipw_longbench[max_ipw_idx]+0.8),
            arrowprops=dict(arrowstyle='->', color='#7d2818', alpha=0.8, lw=2),
            fontsize=10, ha='center', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#fdf2f2', alpha=0.9, edgecolor='#7d2818'))

# Style the spines (borders)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('#bdc3c7')

# Remove top and right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust layout
plt.tight_layout()

# Add subtle watermark/signature
fig.text(0.99, 0.01, 'Generated with Python & Matplotlib', 
         fontsize=8, color='#95a5a6', ha='right', style='italic', alpha=0.7)

# Display the plot
plt.show()

# Print statistics
print("-" * 40)
print(f"FP Full Attention Score: {fp_full_score}")
print(f"SageAttn 8-bit Score: {sageattn_8bit_score}")
print(f"Best MSE Score: {max(mse_longbench):.2f} (at {mse_bit[mse_longbench.index(max(mse_longbench))]:.2f} bits)")
print(f"Best IPW Score: {max(ipw_longbench):.2f} (at {ipw_bit[ipw_longbench.index(max(ipw_longbench))]:.2f} bits)")
print("=" * 60)
print("PERFORMANCE ANALYSIS SUMMARY")
print("=" * 60)
print(f"MSE - Bit Range: {min(mse_bit):.2f} to {max(mse_bit):.2f}")
print(f"MSE - Score Range: {min(mse_longbench):.2f} to {max(mse_longbench):.2f}")
print(f"MSE - Average Score: {np.mean(mse_longbench):.2f}")
print("-" * 40)
print(f"IPW - Bit Range: {min(ipw_bit):.2f} to {max(ipw_bit):.2f}")
print(f"IPW - Score Range: {min(ipw_longbench):.2f} to {max(ipw_longbench):.2f}")
print(f"IPW - Average Score: {np.mean(ipw_longbench):.2f}")
print("=" * 60)

# Save the plot (optional)
plt.savefig('mse_vs_ipw_comparison.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')