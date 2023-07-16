import matplotlib.pyplot as plt
import numpy as np
# import sys
# sys.path += ["..", "../..", "../../.."]
# from experimentkit_in.generators.time_series import gen_lorenz
# xyzs= gen_lorenz(n_steps=4000)

# %% lorenz system 3D

ax = plt.figure().add_subplot(projection='3d')
ax.plot(*xyzs.T, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")
plt.show()

# %% lorenz show X Y Z

fig, axs = plt.subplots(3, 1)
fig.suptitle("Lorenz System")

n= 4000
labels = ['X', 'Y', 'Z']
for i in range(3):
    ax = axs[i]
    ax.plot(np.arange(xyzs[:n].shape[0]), xyzs[:n, i])
    ax.set_ylabel(labels[i], rotation=0)
    ax.grid()
fig.tight_layout()

# %% lorenz show X Y Z chunked

chunk_len = 200
n_chunks = 4
chunks = xyzs[:chunk_len * n_chunks].reshape(n_chunks, chunk_len, xyzs.shape[1])
# plot
fig, axs = plt.subplots(3, 1)
fig.suptitle("Signal split")

labels = ['X', 'Y', 'Z']
x_labels_pos = []
for ch_i, chunk in enumerate(chunks):
    for i in range(3): # up to -1
        ax = axs[i]
        x = np.arange(chunk_len) + chunk_len * ch_i
        ax.plot(x, chunk[:, i])
        ax.set_ylabel(labels[i], rotation=0)
        ax.axvline(x[-1], c='grey')
    x_labels_pos.append(x[-1] + 1)

for ax in axs.ravel():
    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(np.arange(0, len(x_labels_pos)))
    ax.grid()
fig.tight_layout()