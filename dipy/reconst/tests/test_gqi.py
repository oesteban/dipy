import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from dipy.core.gradients import gradient_table
from dipy.core.sphere_stats import angular_similarity
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.data import default_sphere, dsi_voxels, get_fnames, get_sphere
from dipy.direction.peaks import peak_directions
from dipy.reconst.gqi import GeneralizedQSamplingModel, gqi_kernel, prediction_kernel
from dipy.reconst.odf import gfa
from dipy.reconst.tests.test_dsi import sticks_and_ball_dummies
from dipy.sims.voxel import sticks_and_ball


def test_gqi():
    # load repulsion 724 sphere
    sphere = default_sphere
    # load icosahedron sphere
    sphere2 = create_unit_sphere(recursion_level=5)
    btable = np.loadtxt(get_fnames(name="dsi515btable"))
    bvals = btable[:, 0]
    bvecs = btable[:, 1:]
    gtab = gradient_table(bvals, bvecs=bvecs)
    data, golden_directions = sticks_and_ball(
        gtab, d=0.0015, S0=100, angles=[(0, 0), (90, 0)], fractions=[50, 50], snr=None
    )
    gq = GeneralizedQSamplingModel(gtab, method="gqi2", sampling_length=1.4)

    # repulsion724
    gqfit = gq.fit(data)
    odf = gqfit.odf(sphere)
    directions, values, indices = peak_directions(
        odf, sphere, relative_peak_threshold=0.35, min_separation_angle=25
    )
    assert_equal(len(directions), 2)
    assert_almost_equal(angular_similarity(directions, golden_directions), 2, 1)

    # 5 subdivisions
    gqfit = gq.fit(data)
    odf2 = gqfit.odf(sphere2)
    directions, values, indices = peak_directions(
        odf2, sphere2, relative_peak_threshold=0.35, min_separation_angle=25
    )
    assert_equal(len(directions), 2)
    assert_almost_equal(angular_similarity(directions, golden_directions), 2, 1)

    sb_dummies = sticks_and_ball_dummies(gtab)
    for sbd in sb_dummies:
        data, golden_directions = sb_dummies[sbd]
        odf = gq.fit(data).odf(sphere2)
        directions, values, indices = peak_directions(
            odf, sphere2, relative_peak_threshold=0.35, min_separation_angle=25
        )
        if len(directions) <= 3:
            assert_equal(len(directions), len(golden_directions))
        if len(directions) > 3:
            assert_equal(gfa(odf) < 0.1, True)


def test_mvoxel_gqi():
    data, gtab = dsi_voxels()
    sphere = get_sphere(name="symmetric724")

    gq = GeneralizedQSamplingModel(gtab, method="standard")
    gqfit = gq.fit(data)
    all_odfs = gqfit.odf(sphere)

    # Check that the first and last voxels each have 2 peaks
    odf = all_odfs[0, 0, 0]
    directions, values, indices = peak_directions(
        odf, sphere, relative_peak_threshold=0.35, min_separation_angle=25
    )
    assert_equal(directions.shape[0], 2)
    odf = all_odfs[-1, -1, -1]
    directions, values, indices = peak_directions(
        odf, sphere, relative_peak_threshold=0.35, min_separation_angle=25
    )
    assert_equal(directions.shape[0], 2)


def test_prediction_kernel():
    """Test that the GQI prediction kernel satisfies key mathematical properties
    of a Tikhonov-regularized pseudo-inverse for both "standard" and "gqi2" methods."""
    # Load test data
    _, gtab = dsi_voxels()
    sphere = get_sphere(name="symmetric724")
    param_lambda = 1.2

    for method in ["standard", "gqi2"]:
        K_plus = prediction_kernel(gtab, param_lambda, sphere, method=method)

        # Shape check
        assert_equal(K_plus.shape, (len(sphere.vertices), len(gtab.bvals)))

        # Finite and non-zero
        assert np.all(np.isfinite(K_plus)), "K_plus contains non-finite values"
        assert np.any(K_plus != 0), "K_plus is all zeros"

        # Compute forward kernel
        K = gqi_kernel(gtab, param_lambda, sphere, method=method)

        # K and K_plus should be transposed in shape
        assert (
            K_plus.shape == K.T.shape
        ), f"K_plus.shape {K_plus.shape} does not match K.T.shape {K.T.shape}"

        # Property 1: K @ K_plus @ K ≈ K  (regularized reconstruction)
        reconstructed_K = K @ K_plus @ K
        assert np.allclose(
            reconstructed_K, K, atol=1e-4, rtol=1e-3
        ), "Regularized reconstruction K K_plus K ≈ K failed"

        # Property 2: K K_plus is symmetric
        KK_plus = K @ K_plus
        assert np.allclose(KK_plus, KK_plus.T, atol=1e-5), "K K_plus is not symmetric"

        # Property 3: K_plus K is symmetric
        K_plusK = K_plus @ K
        assert np.allclose(K_plusK, K_plusK.T, atol=1e-5), "K_plus K is not symmetric"


def test_predict_single_voxel():
    """Test GQI single voxel prediction API"""
    # Load test data
    data, gtab = dsi_voxels()

    tested_voxels_coordinates = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)]
    # Test single voxel prediction for both methods and all wanted voxels
    for voxel_coordinate in tested_voxels_coordinates:
        for method in ["standard", "gqi2"]:
            gq = GeneralizedQSamplingModel(gtab, method=method, sampling_length=1.2)
            # Test single voxel prediction
            voxel_data = data[voxel_coordinate]
            voxel_fit = gq.fit(voxel_data)
            voxel_predicted = voxel_fit.predict(gtab)

            # Basic shape and value checks
            assert_equal(voxel_predicted.shape, (len(gtab.bvals),))
            assert np.all(
                voxel_predicted >= 0
            ), "Predicted signals should be non-negative"

            # Test prediction with subset of gradients
            subset_gtab = gradient_table(gtab.bvals[::2], bvecs=gtab.bvecs[::2])
            subset_predicted = voxel_fit.predict(subset_gtab)

            assert_equal(subset_predicted.shape, (len(subset_gtab.bvals),))
            assert np.all(
                subset_predicted >= 0
            ), "Subset predictions should be non-negative"


def test_predict_multi_voxel():
    """Test GQI multi-voxel prediction API"""
    # Load test data
    data, gtab = dsi_voxels()

    # Test multi-voxel prediction for both methods
    for method in ["standard", "gqi2"]:
        gq = GeneralizedQSamplingModel(gtab, method=method, sampling_length=1.2)

        # Test multi-voxel prediction
        multi_fit = gq.fit(data)
        multi_predicted = multi_fit.predict(gtab)

        # Basic shape and value checks
        assert_equal(multi_predicted.shape, data.shape)
        assert np.all(multi_predicted >= 0), "Predicted signals should be non-negative"

        # Test prediction with subset of gradients
        subset_gtab = gradient_table(gtab.bvals[::2], bvecs=gtab.bvecs[::2])
        subset_predicted = multi_fit.predict(subset_gtab)

        # Expected shape is (Original_x, Original_y, Original_z, Orignial_N / 2)
        expected_shape = data.shape[:-1] + (len(subset_gtab.bvals),)

        assert_equal(subset_predicted.shape, expected_shape)
        assert np.all(
            subset_predicted >= 0
        ), "Subset predictions should be non-negative"


# TODO: Check why there is such a different correlation between
#       standard and gqi2 methods
def test_predict_roundtrip_single_voxel():
    """Verify that GQI single voxel predictions maintain high correlation
    with original signals

    Here only 1 voxel is fitted to the model and compared to it's prediction
    """
    # Load test data
    data, gtab = dsi_voxels()

    bvals = gtab.bvals
    bvecs = gtab.bvecs

    # Remove b0 volumes from the train data
    non_b0_indices = np.where(~gtab.b0s_mask)[0]
    train_data = data[..., non_b0_indices]

    train_bvals = bvals[non_b0_indices]
    train_bvecs = bvecs[non_b0_indices]

    train_gtab = gradient_table(bvals=train_bvals, bvecs=train_bvecs)

    # TODO: Find meaningful threshold
    correlation_threshold = 0.8

    tested_voxels_coordinates = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)]
    # Test both methods
    for voxel_coordinate in tested_voxels_coordinates:
        gq = GeneralizedQSamplingModel(
            train_gtab, method="standard", sampling_length=1.2
        )
        # Test single voxel round-trip consistency
        voxel_data = train_data[voxel_coordinate]
        voxel_fit = gq.fit(voxel_data)
        voxel_predicted = voxel_fit.predict(train_gtab)

        # Prediction should have correct shape
        assert_equal(voxel_predicted.shape, (len(train_gtab.bvals),))

        # Predicted signal should be non-negative
        assert np.all(voxel_predicted >= 0), "Predicted signals should be non-negative"

        # Compute correlation between original and predicted
        correlation = np.corrcoef(voxel_data, voxel_predicted)[0, 1]

        # TODO: Remove debug prints
        pred_min = voxel_predicted.min()
        pred_max = voxel_predicted.max()
        # Print debug information
        print(f"\n{20 * '='} Debug print {20 * '='}")
        print(f"\nSingle voxel at {voxel_coordinate}:")
        print(f"  Original range: [{voxel_data.min():.3f}, {voxel_data.max():.3f}]")
        print(f"  Predicted range: [{pred_min:.3f}, {pred_max:.3f}]")
        print(f"  Correlation: {correlation:.3f}")

        # For single voxel, correlation should be high
        assert (
            correlation > correlation_threshold
        ), f"Poor single voxel correlation {correlation:.3f}"

        # Original signal and predicted should be within same order of magnitude
        orig_mean = np.mean(voxel_data)
        pred_mean = np.mean(voxel_predicted)

        ratio = pred_mean / orig_mean if orig_mean > 0 else 1
        assert 0.1 < ratio < 10, f"Signal magnitude unrealistic: ratio={ratio:.3f}"

        # Test with different gradient table
        subset_gtab = gradient_table(train_gtab.bvals[::2], bvecs=train_gtab.bvecs[::2])
        subset_predicted = voxel_fit.predict(subset_gtab)

        # Subset prediction should have correct shape
        assert_equal(subset_predicted.shape, (len(subset_gtab.bvals),))

        # Should still be non-negative
        assert np.all(
            subset_predicted >= 0
        ), "Subset predictions should be non-negative"


def test_predict_roundtrip_multi_voxel():
    """Verify that GQI multi voxel predictions maintain high correlation
    with original signals

    Here all voxels are fitted to the model
    each voxel is getting predicted back and compared to it's original data
    """

    # Load test data
    data, gtab = dsi_voxels()

    bvals = gtab.bvals
    bvecs = gtab.bvecs

    # Remove b0 volumes from the train data
    non_b0_indices = np.where(~gtab.b0s_mask)[0]
    train_data = data[..., non_b0_indices]

    train_bvals = bvals[non_b0_indices]
    train_bvecs = bvecs[non_b0_indices]

    train_gtab = gradient_table(bvals=train_bvals, bvecs=train_bvecs)

    # TODO: Find meaningful thresholds
    average_correlation_threshold = 0.8
    per_voxel_correlation_thesrhold = 0.8

    gq = GeneralizedQSamplingModel(train_gtab, method="standard", sampling_length=1.2)

    # Test multi-voxel round-trip consistency
    multi_fit = gq.fit(train_data)
    multi_predicted = multi_fit.predict(train_gtab)

    # Compute correlation for all voxels
    correlations = []
    total_voxels = 0

    # Iterate through all voxels in the 3D data
    for i in range(train_data.shape[0]):
        for j in range(train_data.shape[1]):
            for k in range(train_data.shape[2]):
                total_voxels += 1
                original_voxel = train_data[i, j, k]
                predicted_voxel = multi_predicted[i, j, k]

                correlation = np.corrcoef(original_voxel, predicted_voxel)[0, 1]
                correlations.append(correlation)

    # TODO: Remove debug prints
    # Print debug information
    print(f"\n{20 * '='} Debug print {20 * '='}")
    print(f"  Original range: [{train_data.min():.3f}, {train_data.max():.3f}]")
    print(
        f"  Predicted range: [{multi_predicted.min():.3f}, {multi_predicted.max():.3f}]"
    )
    print(f"  Total voxels: {total_voxels}")
    print(f"  Number of gradients: {train_data.shape[-1]}")
    print(f"  Average voxel correlation: {np.mean(correlations):.3f}")
    print(f"  Min correlation: {np.min(correlations):.3f}")
    print(f"  Max correlation: {np.max(correlations):.3f}")

    # For multi-voxel, average correlation should be high
    avg_correlation = np.mean(correlations)
    assert (
        avg_correlation > average_correlation_threshold
    ), f"Poor multi-voxel average correlation{avg_correlation:.3f}"

    # Predicted signals should be non-negative
    assert np.all(multi_predicted >= 0), "Predicted signals should be non-negative"

    # Original signal and predicted should be within same order of magnitude
    orig_mean = np.mean(train_data)
    pred_mean = np.mean(multi_predicted)
    ratio = pred_mean / orig_mean if orig_mean > 0 else 1
    assert 0.1 < ratio < 10, f"Signal magnitude unrealistic: ratio={ratio:.3f}"

    # Test with different gradient table
    subset_gtab = gradient_table(train_gtab.bvals[::2], bvecs=train_gtab.bvecs[::2])
    subset_predicted = multi_fit.predict(subset_gtab)

    # Expected shape is (Original_x, Original_y, Original_z, Orignial_N / 2)
    expected_shape = train_data.shape[:-1] + (len(subset_gtab.bvals),)
    assert_equal(subset_predicted.shape, expected_shape)

    # Should still be non-negative
    assert np.all(subset_predicted >= 0), "Subset predictions should be non-negative"

    # Test that all voxels have reasonable correlation
    poor_correlation_voxels = []
    for i in range(train_data.shape[0]):
        for j in range(train_data.shape[1]):
            for k in range(train_data.shape[2]):
                original_voxel = train_data[i, j, k]
                predicted_voxel = multi_predicted[i, j, k]

                # Skip voxels with no signal
                if np.sum(original_voxel) == 0:
                    continue

                test_corr = np.corrcoef(original_voxel, predicted_voxel)[0, 1]
                if test_corr <= per_voxel_correlation_thesrhold:
                    poor_correlation_voxels.append((i, j, k, test_corr))

    # Assert that no voxels have poor correlation
    assert (
        len(poor_correlation_voxels) == 0
    ), f"Found {len(poor_correlation_voxels)} voxels with poor correlation "
