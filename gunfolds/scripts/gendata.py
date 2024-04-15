import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from gunfolds.utils import graphkit as gk
from gunfolds.conversions import graph2adj
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import uuid  # Make sure to import the uuid module


'''def animate_matrix(
    dd, window_size, interval, stride, figsize=(8, 6), aspect_ratio=None
):
    fig, ax = plt.subplots(figsize=figsize)
    lines = [
        ax.plot([], [], ".-", ms=0.5, lw=0.3)[0] for _ in range(dd.shape[0])
    ]
    def init():
        ax.set_xlim(0, window_size)
        ax.set_ylim(np.min(dd), np.max(dd))
        if aspect_ratio is not None:
            ax.set_aspect(aspect_ratio)
        return lines
    def update(frame):
        start_col = frame * stride
        end_col = start_col + window_size
        for i, line in enumerate(lines):
            if end_col <= dd.shape[1]:
                line.set_data(np.arange(window_size), dd[i, start_col:end_col])
            else:
                # Avoid going beyond the number of columns in dd
                padding = np.empty(end_col - dd.shape[1])
                padding.fill(np.nan)  # Fill with NaNs
                data = np.hstack((dd[i, start_col:], padding))
                line.set_data(np.arange(window_size), data)
        return lines
    frames_count = (dd.shape[1] - window_size) // stride + 1
    ani = FuncAnimation(
        fig,
        update,
        frames=range(frames_count),
        init_func=init,
        blit=True,
        interval=interval,
    )
    plt.show()
    return ani'''


def animate_matrix(
    dd,
    window_size,
    interval,
    stride,
    figsize=(8, 6),
    aspect_ratio=None,
    save_animation=False,
    save_duration=4,
):
    fig, ax = plt.subplots(figsize=figsize)
    lines = [
        ax.plot([], [], ".-", ms=0.5, lw=0.3)[0] for _ in range(dd.shape[0])
    ]

    def init():
        ax.set_xlim(0, window_size)
        ax.set_ylim(np.min(dd), np.max(dd))
        if aspect_ratio is not None:
            ax.set_aspect(aspect_ratio)
        return lines

    def update(frame):
        start_col = frame * stride
        end_col = start_col + window_size
        for i, line in enumerate(lines):
            if end_col <= dd.shape[1]:
                line.set_data(np.arange(window_size), dd[i, start_col:end_col])
            else:
                # Avoid going beyond the number of columns in dd
                padding = np.empty(end_col - dd.shape[1])
                padding.fill(np.nan)  # Fill with NaNs
                data = np.hstack((dd[i, start_col:], padding))
                line.set_data(np.arange(window_size), data)
        return lines

    frames_count = (dd.shape[1] - window_size) // stride + 1
    frames_to_save = int((save_duration * 1000) / interval)
    frames_range = range(min(frames_to_save, frames_count))

    ani = FuncAnimation(
        fig,
        update,
        frames=frames_range,
        init_func=init,
        blit=True,
        interval=interval,
    )

    if save_animation:
        # Generate a random filename using uuid
        random_filename = f"animation_{uuid.uuid4()}.gif"
        # Save the animation as an animated GIF
        ani.save(random_filename, writer="imagemagick", fps=1000 / interval)
        print(f"Saved animation to {random_filename}")
        # Close the figure to prevent it from displaying after saving
        plt.close(fig)
    else:
        # Display the animation if not saving
        plt.show()

    return ani


def check_matrix_powers(W, A, powers, threshold):
    for n in powers:
        W_n = np.linalg.matrix_power(W, n)
        non_zero_indices = np.nonzero(W_n)
        if (np.abs(W_n[non_zero_indices]) < threshold).any():
            return False
    return True


def create_stable_weighted_matrix(
    A,
    threshold=0.1,
    powers=[1, 2, 3, 4],
    max_attempts=1000,
    damping_factor=0.99,
    random_state=None,
):
    np.random.seed(
        random_state
    )  # Set random seed for reproducibility if provided
    attempts = 0

    while attempts < max_attempts:
        # Generate a random matrix with the same sparsity pattern as A
        random_weights = np.random.randn(*A.shape)
        weighted_matrix = A * random_weights

        # Convert to sparse format for efficient eigenvalue computation
        weighted_sparse = sp.csr_matrix(weighted_matrix)

        # Compute the largest eigenvalue in magnitude
        eigenvalues, _ = eigs(weighted_sparse, k=1, which="LM")
        max_eigenvalue = np.abs(eigenvalues[0])

        # Scale the matrix so that the spectral radius is slightly less than 1
        if max_eigenvalue > 0:
            weighted_matrix *= damping_factor / max_eigenvalue
            # Check if the powers of the matrix preserve the threshold for non-zero entries of A
            if check_matrix_powers(weighted_matrix, A, powers, threshold):
                return weighted_matrix

        attempts += 1

    raise ValueError(
        f"Unable to create a matrix satisfying the condition after {max_attempts} attempts."
    )


def drawsamplesLG(A, nstd=0.1, samples=100):
    n = A.shape[0]
    data = np.zeros([n, samples])
    data[:, 0] = nstd * np.random.randn(A.shape[0])
    for i in range(1, samples):
        data[:, i] = A @ data[:, i - 1] + nstd * np.random.randn(A.shape[0])
    return data


def genData(A, rate=2, burnin=100, ssize=5000, noise=0.1, dist="normal"):
    Agt = A
    data = drawsamplesLG(Agt, samples=burnin + (ssize * rate), nstd=noise)
    data = data[:, burnin:]
    return data[:, ::rate]


u_rate = 1
noise_svar = 0.1
g = gk.ringmore(8, 2)
A = graph2adj(g)
W = create_stable_weighted_matrix(A, threshold=0.1, powers=[2, 3, 4])

dd = genData(W, rate=u_rate, ssize=8000, noise=noise_svar)

shift = 2.5
shift_values = shift * np.arange(dd.shape[0])[:, np.newaxis]
ddplot = dd + shift_values

animation = animate_matrix(
    ddplot[:, :2000],
    window_size=1000,
    interval=1,
    stride=3,
    figsize=(10, 3),
    aspect_ratio="auto",
       save_animation=True,  # Set to True to save the animation
       save_duration=4,  # Duration of the saved animation in seconds
)