"""
=======================================================
Predict unseen data with Generalized Q-Sampling Imaging
=======================================================

Predicting unseen diffusion data is particularly valuable in the context of
the "leave-one-out" (LOO) framework, a cross-validation technique adapted from
machine learning to diffusion MRI. In LOO, one or more diffusion gradient orientations
are artificially removed from the dataset, the model is trained on the remaining data,
and the missing signals are predicted. This process iterates across all orientations,
allowing evaluation of model generalization without requiring multiple scans.

Unlike the standard GQI reconstruction (see
:ref:`sphx_glr_examples_built_reconstruction_reconst_gqi.py`), which fits the model
to the full dataset to compute orientation distribution functions (ODFs) and
fiber peaks, this predictive approach simulates incomplete acquisitions (e.g., due to
motion artifacts or limited scan time). It assesses how well GQI can infer missing
diffusion signals from partial data, providing a quantitative measure of robustness.

This capability supports various applications, including:

    - Model cross-validation: Evaluating a model's adaptability to unseen
      gradient directions, as demonstrated in
      :ref:`sphx_glr_examples_built_reconstruction_kfold_xval.py`.
    - Data cleaning: Identifying and correcting noisy or erroneous diffusion volumes
      (e.g., motion artifacts) by comparing predicted vs. actual signals
      :footcite:p:`Amitay2012`.
    - Data augmentation: Generating synthetic diffusion data to enhance
      training datasets for downstream tasks like tractography.

For further reading, see Yeh et al. (2010) on GQI fundamentals :footcite:p:`Yeh2010`,
and general LOO cross-validation in Hastie et al. (2009) :footcite:p:`Hastie2009`.

This example demonstrates predicting unseen diffusion orientations using
Generalized Q-Sampling Imaging (GQI) within a leave-one-out framework.

We begin by loading and visualizing a sample diffusion MRI dataset. Next, we split
the data into training and test sets, masking brain regions for efficient computation.
The GQI model is then fitted to the training data and used to predict the held-out
orientations. Finally, we compare the predictions against ground truth using
visual plots and local correlation maps, highlighting the method's ability
to enhance signal quality.

The "leave one out" procedure works as follows:

    1. Leave one DWI orientation out.
    2. Fit the rest of the data to the GQI model.
    3. Predict the left out orientation.
    4. Run a volumetric registration algorithm between the original orientation and
       the predicted one.
    5. Repeat until all volumes have been visited

However, we will restrict the demonstration to steps 1–3 to maintain
a concise and readable example.

Registration is used in the full procedure because predicted volumes may exhibit
slight geometric distortions or misalignments compared to the originals, arising from
the modeling assumptions. Aligning them via volumetric registration ensures accurate
comparisons, such as correlation maps or error quantification, preventing artifacts
from confounding the evaluation of prediction quality.

Although registration is not demonstrated here, users interested in implementing step 4
can refer to the :ref:`sphx_glr_examples_built_registration_affine_registration_3d.py`
example for guidance on volumetric alignment techniques.

We will also leave out 2 orientations instead of 1 to make the final comparison
more meaningful.

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

###############################################################################
# Load the data, bvals and bvecs needed for the tutorial

fraw, fbval, fbvec = get_fnames(name="taiwan_ntu_dsi")

data, affine, voxel_size = load_nifti(fraw, return_voxsize=True)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
bvecs[1:] = bvecs[1:] / np.sqrt(np.sum(bvecs[1:] * bvecs[1:], axis=1))[:, None]
gtab = gradient_table(bvals, bvecs=bvecs)

print(f"data.shape {data.shape}")

###############################################################################
# Display 5 random volumes to see what we are going to work with


def plot_slice(data, bval, bvec, volume_idx, z_slice=None):
    """
    Plot a single axial slice from a 3D diffusion MRI volume.

    Parameters:
    - data: 3D array of the volume (X, Y, Z)
    - bval: b-value (in s/mm²) of the volume, shown in the title
    - bvec: 3-element b-vector (diffusion gradient direction), shown in the title
    - volume_idx: index of the volume in the original 4D dataset (used for labeling)
    - z_slice: axial slice index along the Z dimension; if None, uses the central slice
    """
    if z_slice is None:
        z_slice = data.shape[2] // 2

    vol = data[:, :, z_slice]

    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(vol, cmap="gray", origin="lower")
    ax.set_title(f"Volume {volume_idx} | bval ={bval} | bvec = {bvec}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


# Setup a fix seed for random operations
rng = np.random.default_rng(42)

# Select 5 random slices to plot
random_slices = rng.choice(data.shape[-1], size=5, replace=False)
for random_idx in random_slices:
    plot_slice(data[..., random_idx], bvals[random_idx], bvecs[random_idx], random_idx)

###############################################################################
# Split the data and gtab into train and test sets.
#
# We have to be careful to remove all the b0 volumes from both sets.

b0_indices = np.where(gtab.b0s_mask)[0]
non_b0_indices = np.where(~gtab.b0s_mask)[0]

# For comparison convenience we leave out 2 orientations
n_test = 2
n_train = len(non_b0_indices) - n_test

# Shuffle only the non-b0 indices
permuted_non_b0 = rng.permutation(non_b0_indices)
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

# Generate the new train and test gtabs
train_gtab = gradient_table(bvals=train_bvals, bvecs=train_bvecs)
test_gtab = gradient_table(bvals=test_bvals, bvecs=test_bvecs)

print(f"Train data shape {train_data.shape}")
print(f"Test data shape {test_data.shape}")

print(f"Number of train voxels: {train_data.size}")
print(f"Number of test voxels: {test_data.size}")

###############################################################################
# Now that our data is correctly prepared, we want to compute a brain mask to make
# the computation faster

masked_data, mask = median_otsu(data, vol_idx=[0])

###############################################################################
# For each of the available GQI methods (``standard`` and ``gqi2``) we want to:
#
#  1. Create the according GQI model with the ``train_gtabs``
#  2. Fit the ``train_data`` to the GQO model
#  3. Predict the unseen ``test data``

methods = ["standard", "gqi2"]
method_predicted_data = {method: {} for method in methods}

for method in methods:
    # Build model
    model = gqi.GeneralizedQSamplingModel(
        train_gtab, method=method, sampling_length=0.9
    )
    fit = model.fit(train_data, mask=mask)

    predicted_data = fit.predict(test_gtab)

    method_predicted_data[method]["predicted_data"] = predicted_data

###############################################################################
# We now plot a comparison of the predicted results, this includes:
#
#   - The left out data
#   - The predicted data
#   - A correlation map between the left data and it's prediction


def local_correlation(real, pred, window_size=5):
    """
    Compute a local Pearson correlation map between ``real`` and ``pred``.
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
    test_data,
    method_predicted_data,
    vol_idx,
    test_bval,
    test_bvec,
    z_slice=None,
    window_size=9,
):
    """
    Plot real data, predictions from all methods, and local Pearson correlation maps.
    Each method gets its own row with three panels:
        [Real | Predicted | Local Correlation].

    Parameters:
    - test_data: 3D array of ground-truth test volume (X, Y, Z)
    - method_predicted_data:
        dict mapping method names to prediction containers;
        each value must contain a key "predicted_data" with a 4D array (X, Y, Z, N_vols)
    - vol_idx: index of the test volume to visualize
    - test_bval: b-value (in s/mm²) of the selected volume, shown in the plot title
    - test_bvec: gradient direction of the selected volume, shown in the title
    - z_slice: axial slice index along the Z dimension; if None, uses the central slice
    - window_size: size of the square window for local correlation (must be odd)
    """

    if z_slice is None:
        z_slice = test_data.shape[2] // 2

    real_vol = test_data[:, :, z_slice]
    methods = list(method_predicted_data.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(n_methods, 3, figsize=(13, 3.8 * n_methods))

    if n_methods == 1:
        axes = axes[None, :]

    for i, method in enumerate(methods):
        pred_vol = method_predicted_data[method]["predicted_data"][
            :, :, z_slice, vol_idx
        ]

        # Compute local correlation map
        real_pred_corr_map = local_correlation(
            real_vol, pred_vol, window_size=window_size
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

        # Local Correlation (real vs prediction)
        im_corr = axes[i, 2].imshow(
            real_pred_corr_map, cmap="RdBu_r", origin="lower", vmin=-1, vmax=1
        )
        axes[i, 2].set_title(
            "Local Pearson Correlation\n(real vs prediction)", fontsize=12
        )
        axes[i, 2].axis("off")
        fig.colorbar(im_corr, ax=axes[i, 2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Test Volume {vol_idx} | Z Slice {z_slice} | "
        f"Corr window size {window_size} | "
        f"bval = {test_bval:.0f} s/mm^2 | "
        f"bvec = {test_bvec}",
        fontsize=14,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=4.0)
    plt.show()


for i in range(len(test_idx)):
    plot_all_methods_comparison(
        test_data[..., i],
        method_predicted_data,
        i,
        test_bvals[i],
        test_bvecs[i],
        window_size=9,
    )

###############################################################################
# It should be noted that, at the current time, while the ``standard`` GQI model
# prediction generates great images, this is not yet the case for the ``gqi2``
# GQI model.
#
# Moreover, we can see that the ``standard model``’s predicted left-out orientations
# have a higher SNR than the original data while still preserving signal information.
#
#
# References
# ----------
#
# .. footbibliography::
#
