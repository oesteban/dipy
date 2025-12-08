"""
===============================================
Predict unseen data with Generalized Q-Sampling Imaging
===============================================

The ability to predict unseen data can be very useful when combined with
the "Leave one out" framework.

This example procedure is heavily inspired by the
NiPreps book :footcite:p:`Joseph_NiPreps_digital_book_2025`

This idea works as follow:

    1. Leave one DWI orientation out.
    2. Fit the rest of the data to the GQI model.
    3. Predict the left out orientation.
    4. Run a volumetric registration algorithm between the original orientation and
       the predicted one.
    5. Repeat for the whole dataset until convergence.

However, to make this example more digest, we will only go through steps 1-3.
For the registration step, you can refer to the
:ref:`Affine Registration in 3D example
<sphx_glr_examples_built_registration_affine_registration_3d.py>`

We will also leave out 2 orientations instead of 1 to make the final comparison
more meaningful


First import the necessary python modules:
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
import dipy.reconst.gqi as gqi
from dipy.segment.mask import median_otsu

np.random.seed(42)

###############################################################################
# Load the data, bvals and bvecs needed for the tutorial
fraw, fbval, fbvec = get_fnames(name="stanford_hardi")

data, affine, voxel_size = load_nifti(fraw, return_voxsize=True)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs=bvecs)

print(f"data.shape {data.shape}")


###############################################################################
# Display 5 random volumes to see what we are going to work with
def plot_slice(data, bvals, vol_idx=0, z_slice=None, title_prefix="Volume"):
    """
    Plot a single 2D slice from a 4D diffusion volume.

    Parameters:
    - data: 4D array of shape (X, Y, Z, N)
    - bvals: 1D array of b-values (length N)
    - vol_idx: which volume (4th dim) to show
    - z_slice: axial slice index; if None, use middle slice
    - title_prefix: label for the plot title
    """
    if z_slice is None:
        z_slice = data.shape[2] // 2

    vol = data[:, :, z_slice, vol_idx]

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(vol, cmap="gray", origin="lower")
    ax.set_title(f"{title_prefix} | Volume {vol_idx} (b={bvals[vol_idx]:.0f})")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


# Select 5 random slices to plot
random_slices = np.random.choice(data.shape[-1], size=5, replace=False)
for random_idx in random_slices:
    plot_slice(data, bvals, random_idx)

###############################################################################
# Split the data and gtab into train and test sets
#
# We have to be careful to remove all the b0 volumes from both sets
b0_indices = np.where(gtab.b0s_mask)[0]
non_b0_indices = np.where(~gtab.b0s_mask)[0]

# For comparison convenience we will actually leave out 2 orientations
n_test = 2
n_train = len(non_b0_indices) - n_test

# Shuffle only the non-b0 indices
permuted_non_b0 = np.random.permutation(non_b0_indices)
train_idx = permuted_non_b0[:n_train]
test_idx = permuted_non_b0[n_train:]

print(f"Number of train gradients: {len(train_idx)}")
print(f"Number of test gradients: {len(test_idx)}")
print(f"Number of b=0 volumes: {len(b0_indices)}")

# Split the 4D data (along last axis)
train_data = data[..., train_idx]
test_data = data[..., test_idx]

# Split bvals and bvecs
train_bvals = bvals[train_idx]
train_bvecs = bvecs[train_idx]

test_bvals = bvals[test_idx]
test_bvecs = bvecs[test_idx]

train_gtab = gradient_table(bvals=train_bvals, bvecs=train_bvecs)
test_gtab = gradient_table(bvals=test_bvals, bvecs=test_bvecs)

print(f"Train data shape {train_data.shape}")
print(f"Test data shape {test_data.shape}")

print(f"Number of train voxels: {train_data.size}")
print(f"Number of test voxels: {test_data.size}")

###############################################################################
# Compute a brain mask to make the computation faster
masked_data, mask = median_otsu(data, vol_idx=[0])

###############################################################################
# For each of the available GQI methods (standard and gqi2) we want to:
#  - Create the according GQI model with the `train_gtabs`
#  - Fit the `train_data` to the GQO model
#  - Predict the unseen test data
#  - Also predict an interceptor data (data the model generates for bval and bvec of 0)

methods = ["standard", "gqi2"]
method_predicted_data = {method: {} for method in methods}

interceptor_gtab = gradient_table(bvals=[0], bvecs=[[0, 0, 0]])

for method in methods:
    # Build model
    model = gqi.GeneralizedQSamplingModel(
        train_gtab, method=method, sampling_length=0.9
    )
    fit = model.fit(train_data, mask=mask)

    predicted_data = fit.predict(test_gtab)
    interceptor_data = fit.predict(interceptor_gtab)

    method_predicted_data[method]["predicted_data"] = predicted_data
    method_predicted_data[method]["interceptor_data"] = interceptor_data


###############################################################################
# Plot a comparison of the predicted results, this includes:
#   - The left out data
#   - The predicted data
#   - The predictor data
#   - A correlation map between the original data and it's prediction
#   - A correlation map between the prediction and the predictor data
def local_correlation(real, pred, window_size=5):
    """
    Compute a local Pearson correlation map between `real` and `pred`.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")

    # Ensure float dtype to support NaN
    real = real.astype(np.float32)
    pred = pred.astype(np.float32)

    pad = window_size // 2
    real_padded = np.pad(real, pad, mode="constant", constant_values=np.nan)
    pred_padded = np.pad(pred, pad, mode="constant", constant_values=np.nan)

    corr_map = np.full_like(real, np.nan, dtype=np.float32)
    h, w = real.shape

    for i in range(h):
        for j in range(w):
            real_patch = real_padded[i : i + window_size, j : j + window_size].flatten()
            pred_patch = pred_padded[i : i + window_size, j : j + window_size].flatten()

            valid = ~(np.isnan(real_patch) | np.isnan(pred_patch))
            if valid.sum() < 2:
                corr_map[i, j] = np.nan
            else:
                r, _ = pearsonr(real_patch[valid], pred_patch[valid])
                corr_map[i, j] = r
    return corr_map


def plot_all_methods_comparison(
    test_data, test_bvals, method_predicted_data, vol_idx, z_slice=None, window_size=5
):
    """
    Plot real data, predictions from all methods, and local correlation maps.
    Each method gets its own row: [Real | Predicted | Correlation]

    Parameters:
    - test_data: ndarray of shape (H, W, D, V) — ground truth
    - test_bvals: optional b-values for title annotation
    - method_predicted_data: dict {method_name: prediction_array}
    - vol_idx: which volume (4th dim) to visualize
    - z_slice: which axial slice (3rd dim); defaults to middle
    - window_size: size of local neighborhood for correlation (odd integer, e.g., 5)
    """

    if z_slice is None:
        z_slice = test_data.shape[2] // 2

    real_vol = test_data[:, :, z_slice, vol_idx]
    methods = list(method_predicted_data.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(n_methods, 5, figsize=(16, 3.8 * n_methods))

    if n_methods == 1:
        axes = axes[None, :]

    for i, method in enumerate(methods):
        pred_vol = method_predicted_data[method]["predicted_data"][
            :, :, z_slice, vol_idx
        ]
        interceptor_vol = method_predicted_data[method]["interceptor_data"][
            :, :, z_slice, -1
        ]

        # Compute local correlation map
        real_pred_corr_map = local_correlation(
            real_vol, pred_vol, window_size=window_size
        )
        interceptor_pred_corr_map = local_correlation(
            interceptor_vol, pred_vol, window_size=window_size
        )

        # Real
        im_real = axes[i, 0].imshow(real_vol, cmap="gray", origin="lower")
        if i == 0:
            axes[i, 0].set_title("Real data", fontsize=12)
        axes[i, 0].axis("off")
        fig.colorbar(im_real, ax=axes[i, 0], fraction=0.046, pad=0.04)

        # Predicted
        im_pred = axes[i, 1].imshow(pred_vol, cmap="gray", origin="lower")
        axes[i, 1].set_title(f"{method.upper()} prediction", fontsize=12)
        axes[i, 1].axis("off")
        fig.colorbar(im_pred, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Interceptor
        im_interceptor = axes[i, 2].imshow(interceptor_vol, cmap="gray", origin="lower")
        axes[i, 2].set_title(f"{method.upper()} interceptor", fontsize=12)
        axes[i, 2].axis("off")
        fig.colorbar(im_interceptor, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # Local Correlation (real vs prediction)
        im_corr = axes[i, 3].imshow(
            real_pred_corr_map, cmap="RdBu_r", origin="lower", vmin=-1, vmax=1
        )
        axes[i, 3].set_title(
            "Local Pearson Correlation\n(real vs prediction)", fontsize=12
        )
        axes[i, 3].axis("off")
        fig.colorbar(im_corr, ax=axes[i, 3], fraction=0.046, pad=0.04)

        # Local Correlation (intercetor vs prediction)
        im_corr = axes[i, 4].imshow(
            interceptor_pred_corr_map, cmap="RdBu_r", origin="lower", vmin=-1, vmax=1
        )
        axes[i, 4].set_title(
            "Local Pearson Correlation\n(interceptor vs prediction)", fontsize=12
        )
        axes[i, 4].axis("off")
        fig.colorbar(im_corr, ax=axes[i, 4], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Test Volume {vol_idx} | Z Slice {z_slice} | \
          Corr window size {window_size} (b = {test_bvals[vol_idx]:.0f})",
        fontsize=14,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=4.0)
    plt.show()


for i in range(len(test_idx)):
    plot_all_methods_comparison(
        test_data, test_bvals, method_predicted_data, vol_idx=i, window_size=9
    )
###############################################################################
#
# References
# ----------
#
# .. footbibliography::
#
