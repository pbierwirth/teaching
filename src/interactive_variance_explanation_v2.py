import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import norm
#%matplotlib qt

# --- Daten generieren ---
np.random.seed(10)
n = 2000
x = np.random.normal(0, 1, n)
noise_base = np.random.normal(0, 1, n)

# --- Startparameter ---
initial_r = 0.0
initial_x_pos = 0.0
slice_width = 0.2  # Breite des farbigen Balkens

def get_mask(x_val):
    return (x > x_val - slice_width/2) & (x < x_val + slice_width/2)

mask = get_mask(initial_x_pos)
y = (x - x.mean()) * initial_r + noise_base * np.sqrt(1 - initial_r**2)
vec = np.arange(-5, 5, 0.01)

# --- Layout der Abbildung ---
fig = plt.figure(figsize=(6, 7))
ax_m  = fig.add_axes([0.15, 0.35, 0.75, 0.60]) # Haupt-Plot
ax_s1 = fig.add_axes([0.15, 0.20, 0.75, 0.05]) # Slider 1 (Korrelation)
ax_s2 = fig.add_axes([0.15, 0.10, 0.75, 0.05]) # Slider 2 (X-Slice Position)

slider_r = Slider(ax_s1, 'Correlation', valmin=-0.99, valmax=0.99, valstep=0.01, valinit=initial_r, color='#2c8cde')
slider_x = Slider(ax_s2, 'X-Slice', valmin=-3, valmax=3, valstep=0.1, valinit=initial_x_pos, color='#756bb1')

# --- Initiale Plots zeichnen ---
# 1. Die Punktewolke
dots,   = ax_m.plot(x, y, '.', markersize=10, color='black', alpha=0.05, markeredgecolor='None')
dots_m, = ax_m.plot(x[mask], y[mask], '.', markersize=10, color='#756bb1', alpha=1)

# 2. Statische Randverteilung (Marginal Distribution) als Referenz (Gestrichelte Linie)
marginal_pdf = norm.pdf(vec, 0, 1)
dist_com, = ax_m.plot(initial_x_pos + (marginal_pdf / max(marginal_pdf) * 1.5), vec, color='k', linestyle='--', alpha=0.5)

# 3. Dynamische theoretische bedingte Verteilung (Conditional Distribution)
initial_theoretical_mu = initial_r * initial_x_pos
initial_theoretical_std = np.sqrt(max(0.0001, 1 - initial_r**2))
cond_pdf = norm.pdf(vec, initial_theoretical_mu, initial_theoretical_std)
dist, = ax_m.plot(initial_x_pos + (cond_pdf / max(cond_pdf) * 1.5), vec, color='#756bb1', linewidth=2)

# 4. Vertikale Hilfslinie
vline = ax_m.axvline(initial_x_pos, color='gray', linestyle=':', alpha=0.5)

ax_m.set_xlim(-4, 4)
ax_m.set_ylim(-5, 5)

# --- Update-Logik für die Slider ---
def update(val):
    r = slider_r.val
    x_pos = slider_x.val

    # 1. Y-Werte basierend auf Korrelation neu berechnen
    new_y = (x - x.mean()) * r + noise_base * np.sqrt(max(0.0001, 1 - r**2))
    dots.set_ydata(new_y)

    # 2. Maske für den Ausschnitt neu berechnen und lila Punkte updaten
    new_mask = get_mask(x_pos)
    dots_m.set_data(x[new_mask], new_y[new_mask])

    # 3. Die graue Referenz-Verteilung zur neuen X-Position schieben
    dist_com.set_xdata(x_pos + (marginal_pdf / max(marginal_pdf) * 1.5))

    # 4. DIE THEORETISCHE bedingte Verteilung berechnen!
    theoretical_mu = r * x_pos
    theoretical_std = np.sqrt(max(0.0001, 1 - r**2))

    new_cond_pdf = norm.pdf(vec, theoretical_mu, theoretical_std)

    # Die Kurve bleibt in ihrer Form stabil und rutscht nur hoch/runter
    dist.set_xdata(x_pos + (new_cond_pdf / max(new_cond_pdf) * 1.5))

    # 5. Vertikale Linie verschieben
    vline.set_xdata([x_pos, x_pos])

    fig.canvas.draw_idle()

slider_r.on_changed(update)
slider_x.on_changed(update)

plt.show()
