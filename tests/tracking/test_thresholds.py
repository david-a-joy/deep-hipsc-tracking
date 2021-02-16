import unittest

from deep_hipsc_tracking.tracking import thresholds

# Tests


class TestDetectorThresholds(unittest.TestCase):

    def test_finds_inverted(self):

        res = thresholds.DetectorThresholds.by_microscope('inverted')

        self.assertEqual(res.microscope, 'inverted')
        self.assertEqual(res.training_set, 'peaks')

        res = thresholds.DetectorThresholds.by_training_set('peaks')

        self.assertEqual(res.microscope, 'inverted')
        self.assertEqual(res.training_set, 'peaks')

    def test_finds_confocal(self):

        res = thresholds.DetectorThresholds.by_microscope('confocal')

        self.assertEqual(res.microscope, 'confocal')
        self.assertEqual(res.training_set, 'confocal')

        res = thresholds.DetectorThresholds.by_training_set('confocal')

        self.assertEqual(res.microscope, 'confocal')
        self.assertEqual(res.training_set, 'confocal')

    def test_all_thresholds_have_matching_keys(self):

        for cls in thresholds.DetectorThresholds.all_thresholds():
            bad_keys = []
            detectors = [d.split('-', 1)[0] for d in cls.detectors]
            for attr in ['training_detectors', 'detector_thresholds']:
                data = getattr(cls, attr)
                if not all([d in data for d in detectors]):
                    bad_keys.append(attr)
            self.assertEqual(bad_keys, [], 'Class {} doesnt define all detectors: {}'.format(cls, bad_keys))

    def test_all_thresholds_have_expected_attrs(self):

        attrs = ['microscope', 'training_set', 'train_rootdir',
                 'training_detectors', 'detectors', 'detector_weights',
                 'detector_thresholds']
        for cls in thresholds.DetectorThresholds.all_thresholds():
            missing_attrs = []
            for attr in attrs:
                if getattr(cls, attr, None) is None:
                    missing_attrs.append(attr)
            self.assertEqual(missing_attrs, [], 'Class {} missing attributes: {}'.format(cls, missing_attrs))
