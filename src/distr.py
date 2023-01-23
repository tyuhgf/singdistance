import time

import numpy as np
import torch

t0 = time.time()


alpha = [np.pi / 3]

phi1_span = (0, np.pi / 3)
phi1_steps = 100

area_span = (0, .2)
area_steps = 1500

phi1 = np.linspace(*phi1_span, phi1_steps)
phi2 = sum(alpha) - phi1
area = np.linspace(*area_span, area_steps)

dphi1, darea = phi1[1]-phi1[0], area[1]-area[0]
Phi1, Area = np.meshgrid(phi1, area, indexing='ij')
Phi2 = sum(alpha) - Phi1

triangle_areas = 1 / (1 / np.tan(phi1 / 2) + 1 / np.tan(phi2 / 2))  # bad idea to use singular distrs
indexes = np.argmin(np.abs(np.expand_dims(triangle_areas, 1) - np.expand_dims(area, 0)), axis=1)

print(time.time() - t0)

distributions_dim0_2pi3 = np.zeros_like(Area)
distributions_dim0_2pi3[np.arange(len(phi1)), indexes] += 1 / darea

f1 = distributions_dim0_2pi3 * Area ** (1 + 0)
f2 = distributions_dim0_2pi3 * Area ** (1 + 0)

f2_rev = f2[:, ::-1].copy()

f1_tensor = torch.tensor(np.expand_dims(f1, 1))
f1_pad = torch.nn.functional.pad(f1_tensor, (area_steps - 1, 0))
f2_tensor = torch.tensor(np.expand_dims(f2_rev, 1))

print(time.time() - t0)

conv = torch.nn.functional.conv1d(f1_pad, f2_tensor).numpy()

print(time.time() - t0)

phi10 = np.expand_dims(Phi1, 0)
phi11 = np.expand_dims(Phi1, 1)
phi20 = np.expand_dims(Phi2, 0)
phi21 = np.expand_dims(Phi2, 1)
conv *= sum([1 / np.tan(x/2 + 1e-6) for x in [phi10, phi11, phi20, phi21]])
conv /= np.expand_dims(Area + 1e-6, 0) ** (2 + 0)

print(time.time() - t0)


def diag_sums(q):
    n, m, k = q.shape
    q = np.concatenate([q, np.zeros([n, n, k])], axis=1)
    q = q.reshape([n * (n+m), k])[:-n]
    q = q.reshape([n, n + m - 1, k])
    return q.sum(axis=0)


rhs = diag_sums(conv)

print(time.time() - t0)
