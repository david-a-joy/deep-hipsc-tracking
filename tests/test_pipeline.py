import pathlib
import unittest

from deep_hipsc_tracking import pipeline

# Tests


class TestPipelineStages(unittest.TestCase):

    def test_stages_exist(self):

        cls = pipeline.ImagePipeline

        exp = [
            'write_config_file',
            'extract_frames',
            'ensemble_detect_cells',
            'track_cells',
            'mesh_cells',
        ]

        self.assertEqual(cls.pipeline_stages, exp)

        for stage in cls.pipeline_stages:
            self.assertTrue(hasattr(cls, stage))

    def test_can_instantiate_class(self):

        obj = pipeline.ImagePipeline(pathlib.Path('/fake'))

        self.assertEqual(obj.script_dir.name, 'scripts')
        self.assertEqual(obj.log_file, pathlib.Path('/fake/deep_tracking.log'))
        self.assertEqual(obj.config_file, pathlib.Path('/fake/deep_tracking.ini'))
