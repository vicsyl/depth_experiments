import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

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
def get_lbds_for_slope(slope, cc=c):
    return c / (xs - slope)
lmbds_gt = get_lbds_for_slope(slope=2)

raster = np.stack((xs, np.ones_like(xs)))
gt_points = np.stack((lmbds_gt * xs, lmbds_gt))

# points shifted = [(\lambda + \beta) x, (\lambda + \beta)]
for shift in np.linspace(1, 5):
    #for scale in np.linspace(-1, 1, num=3):
    for scale in [1.0]:
        for sheer in np.linspace(0, 3, num=5):
            lmbds = get_lbds_for_slope(slope=2 + sheer)
            lmbds = lmbds * scale + shift
            points = np.stack((lmbds * xs, lmbds))

            y = lmbds_gt
            x = lmbds.reshape(-1, 1)
            reg = LinearRegression().fit(x, y)

            # @ squeezes it
            y_hat = x @ reg.coef_ + reg.intercept_
            points_aligned = np.stack((y_hat * xs, y_hat))

            scale_al = reg.coef_.item()
            shift_al = reg.intercept_.item()
            print(f"results: {(scale, shift, sheer)} vs. {(scale_al, shift_al)}")

            # let's draw it
            plt.figure()
            markersize = 0.8
            plt.plot([0], [0], '+r', linewidth=1, markersize=10, label='Camera center')
            plt.plot(raster[0], raster[1], '+b-', linewidth=1, markersize=markersize, label='Image plane')
            plt.plot(gt_points[0], gt_points[1], '+g-', linewidth=1, markersize=markersize, label='GT points')
            plt.plot(points[0], points[1], '+r-', linewidth=1, markersize=markersize, label=f'points with {(shift, scale, sheer)=}')
            plt.plot(points_aligned[0], points_aligned[1], '+k-', linewidth=1, markersize=markersize, label=f'points aligned {(shift_al, scale_al)=}')

            # rays = []
            # for i in range(len(xs)):
            #     rx = [0, raster[0, i], gt_points[0, i]]
            #     ry = [0, raster[1, i], gt_points[1, i]]
            #     for j in range(len(shifts)):
            #         rx.append(points[j][0, i])
            #         ry.append(points[j][1, i])
            #     plt.plot(rx, ry, "k-.", linewidth=0.1)

            plt.legend()
            fn = f'data/exps_shift/plane_{(scale, shift, sheer)=}.png'
            plt.savefig(fn)
            # print(f'saved to {fn}')
            #plt.show()
            plt.close()
