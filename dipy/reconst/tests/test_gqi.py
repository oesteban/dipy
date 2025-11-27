import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from dipy.core.gradients import gradient_table
from dipy.core.sphere_stats import angular_similarity
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.data import default_sphere, dsi_voxels, get_fnames, get_sphere
from dipy.direction.peaks import peak_directions
from dipy.reconst.gqi import GeneralizedQSamplingModel
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


# TODO: Should it only test single voxel?
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


# TODO: Check why there is such a different correlation between
#       standard and gqi2 methods
def test_predict_meaningful_single_voxel():
    """Test that GQI single voxel predictions give meaningful results"""
    # Load synthetic data with known ground truth
    data, gtab = dsi_voxels()

    tested_voxels_coordinates = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)]
    # Test both methods
    for voxel_coordinate in tested_voxels_coordinates:
        for method in ["standard", "gqi2"]:
            gq = GeneralizedQSamplingModel(gtab, method=method, sampling_length=1.2)
            # Test single voxel round-trip consistency
            voxel_data = data[voxel_coordinate]
            voxel_fit = gq.fit(voxel_data)
            voxel_predicted = voxel_fit.predict(gtab)

            # Compute correlation between original and predicted
            correlation = np.corrcoef(voxel_data, voxel_predicted)[0, 1]

            # TODO: Remove debug prints
            pred_min = voxel_predicted.min()
            pred_max = voxel_predicted.max()
            # Print debug information
            print(f"\n{20 * '='} Debug print {20 * '='}")
            print(f"\nSingle voxel at {voxel_coordinate} prediction ({method}):")
            print(f"  Original range: [{voxel_data.min():.3f}, {voxel_data.max():.3f}]")
            print(f"  Predicted range: [{pred_min:.3f}, {pred_max:.3f}]")
            print(f"  Correlation: {correlation:.3f}")

            # For single voxel, correlation should be high
            # TODO: Find meaningful threshold
            assert (
                correlation > 0.5
            ), f"Poor single voxel correlation {correlation:.3f} for {method} method"

            # Test physical realism - signals should be non-negative
            assert np.all(
                voxel_predicted >= 0
            ), "Predicted signals should be non-negative"

            # Test reasonable signal range
            orig_mean = np.mean(voxel_data[voxel_data > 0])  # Exclude b0 images
            pred_mean = np.mean(voxel_predicted[voxel_predicted > 0])

            # Should be within same order of magnitude
            ratio = pred_mean / orig_mean if orig_mean > 0 else 1
            assert 0.1 < ratio < 10, f"Signal magnitude unrealistic: ratio={ratio:.3f}"

            # Test with different gradient table
            subset_gtab = gradient_table(gtab.bvals[::2], bvecs=gtab.bvecs[::2])
            subset_predicted = voxel_fit.predict(subset_gtab)

            # Subset prediction should have correct shape
            assert_equal(subset_predicted.shape, (len(subset_gtab.bvals),))

            # Should still be non-negative
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
        expected_shape = data.shape[:-1] + (len(subset_gtab.bvals),)
        assert_equal(subset_predicted.shape, expected_shape)
        assert np.all(
            subset_predicted >= 0
        ), "Subset predictions should be non-negative"


def test_predict_meaningful_multi_voxel():
    """Test that GQI multi-voxel predictions give meaningful results"""
    # Load synthetic data with known ground truth
    data, gtab = dsi_voxels()

    # Test both methods
    for method in ["standard", "gqi2"]:
        gq = GeneralizedQSamplingModel(gtab, method=method, sampling_length=1.2)

        # Test multi-voxel round-trip consistency
        multi_fit = gq.fit(data)
        multi_predicted = multi_fit.predict(gtab)

        # Compute correlation for all voxels
        correlations = []
        total_voxels = 0
        valid_voxels = 0

        # Iterate through all voxels in the 3D data
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    total_voxels += 1
                    original_voxel = data[i, j, k]
                    predicted_voxel = multi_predicted[i, j, k]

                    # Skip voxels with no signal (all zeros)
                    if np.sum(original_voxel) == 0:
                        continue

                    valid_voxels += 1
                    correlation = np.corrcoef(original_voxel, predicted_voxel)[0, 1]
                    correlations.append(correlation)

        # TODO: Remove debug prints

        # Print debug information
        print(f"\n{20 * '='} Debug print {20 * '='}")
        print(f"\nMulti-voxel prediction ({method}):")
        print(f"  Original range: [{data.min():.3f}, {data.max():.3f}]")
        print(
            f"  Predicted range: "
            f"[{multi_predicted.min():.3f}, {multi_predicted.max():.3f}]"
        )
        print(f"  Total voxels: {total_voxels}")
        print(f"  Valid voxels (with signal): {valid_voxels}")
        print(f"  Average voxel correlation: {np.mean(correlations):.3f}")
        print(f"  Min correlation: {np.min(correlations):.3f}")
        print(f"  Max correlation: {np.max(correlations):.3f}")

        # For multi-voxel, average correlation should be high
        avg_correlation = np.mean(correlations)
        # TODO: Find meaningful threshold
        assert avg_correlation > 0.5, (
            f"Poor multi-voxel average correlation"
            f"{avg_correlation:.3f} for {method} method"
        )

        # Test physical realism - signals should be non-negative
        assert np.all(multi_predicted >= 0), "Predicted signals should be non-negative"

        # Test reasonable signal range
        orig_mean = np.mean(data[data > 0])  # Exclude b0 images
        pred_mean = np.mean(multi_predicted[multi_predicted > 0])

        # Should be within same order of magnitude
        ratio = pred_mean / orig_mean if orig_mean > 0 else 1
        assert 0.1 < ratio < 10, f"Signal magnitude unrealistic: ratio={ratio:.3f}"

        # Test with different gradient table
        subset_gtab = gradient_table(gtab.bvals[::2], bvecs=gtab.bvecs[::2])
        subset_predicted = multi_fit.predict(subset_gtab)

        # Subset prediction should have correct shape
        expected_shape = data.shape[:-1] + (len(subset_gtab.bvals),)
        assert_equal(subset_predicted.shape, expected_shape)

        # Should still be non-negative
        assert np.all(
            subset_predicted >= 0
        ), "Subset predictions should be non-negative"

        # TODO: Is this necessary ?

        # Test that all valid voxels have reasonable correlation
        poor_correlation_voxels = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    original_voxel = data[i, j, k]
                    predicted_voxel = multi_predicted[i, j, k]

                    # Skip voxels with no signal
                    if np.sum(original_voxel) == 0:
                        continue

                    test_corr = np.corrcoef(original_voxel, predicted_voxel)[0, 1]
                    # TODO: Find meaningful threshold
                    if test_corr <= 0.45:
                        poor_correlation_voxels.append((i, j, k, test_corr))

        # Assert that no voxels have poor correlation
        assert len(poor_correlation_voxels) == 0, (
            f"Found {len(poor_correlation_voxels)} voxels with poor correlation "
            f"(<= 0.5) for {method} method. Examples: {poor_correlation_voxels[:5]}"
        )
