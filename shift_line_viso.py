import matplotlib.pyplot as plt
import numpy as np

# principal axis: y
# xs 1D image normalized coords
xs = np.linspace(-1, 1, num=11)

# "hypo" plane AKA line:
# x - 2 * y = c

# rays and plane intersections
# [\lambda x, \lambda].T [1, - 2] = c
# \lambda x - \lambda 2 = c
# \lambda = c / (x - 2)
c = -3
lmbds = c / (xs - 2)

raster = np.stack((xs, np.ones_like(xs)))
gt_points = np.stack((lmbds * xs, lmbds))

# points shifted = [(\lambda + \beta) x, (\lambda + \beta)]
points_shifted = []
betas = range(1, 4)
for beta in betas:
    points_shifted.append(np.stack(((lmbds + beta) * xs, (lmbds + beta))))

# let's draw it
plt.figure()
markersize = 0.8
plt.plot([0], [0], '+r', linewidth=1, markersize=10, label='Camera center')
plt.plot(gt_points[0], gt_points[1], '+g-', linewidth=1, markersize=markersize, label='GT points on the plane')
plt.plot(raster[0], raster[1], '+b-', linewidth=1, markersize=markersize, label='Image plane')

for ps, beta in zip(points_shifted, betas):
    plt.plot(ps[0], ps[1],
             color=(beta / max(betas), 0, 0), marker='+', linestyle='solid', linewidth=1,
             markersize=markersize, label=f'Shifted by {beta}')

rays = []
for i in range(len(xs)):
    rx = [0, raster[0, i], gt_points[0, i]]
    ry = [0, raster[1, i], gt_points[1, i]]
    for j in range(len(betas)):
        rx.append(points_shifted[j][0, i])
        ry.append(points_shifted[j][1, i])
    plt.plot(rx, ry, "k-.", linewidth=0.1)

plt.legend()
plt.savefig('data/exps_shift/plane.png')
plt.show()
plt.close()
