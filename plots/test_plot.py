import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def draw_underbrace(ax, x_start, x_end, y, label='', color='black'):
    """Draw an underbrace below the x-axis."""
    mid = (x_start + x_end) / 2
    # Brace arms and span
    ax.annotate('', xy=(x_start, y), xytext=(x_end, y),
                arrowprops=dict(arrowstyle='-', color=color,
                                connectionstyle='bar,fraction=0.15'))
    # Label below the brace
    if label:
        ax.text(mid, y - 0.03, label, ha='center', va='top',
                transform=ax.get_xaxis_transform(), color=color, fontsize=11)

fig, ax = plt.subplots()
x = np.linspace(0, 10, 200)
ax.plot(x, np.sin(x))

# Disable clipping so the brace can appear below the axes
ax.set_clip_on(False)

# Draw underbrace in axes-fraction y coords (below the axis)
draw_underbrace(ax, 3, 6, y=-0.12, label='region of interest')

plt.tight_layout()
plt.show()