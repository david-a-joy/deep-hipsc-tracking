""" Preset data for a given experiment """

from . import helpers

from deep_hipsc_tracking import presets

# Tests


class TestPresetToFromFile(helpers.FileSystemTestCase):

    def test_list_presets(self):

        res = presets.list_presets()
        exp = ['confocal', 'inverted']
        self.assertEqual(set(res), set(exp))

    def test_get_preset(self):

        res = presets.get_preset('fake')
        self.assertIsNone(res)

        res = presets.get_preset('confocal')
        self.assertIsInstance(res, presets.ConfocalPreset)

        res = presets.get_preset('inverted')
        self.assertIsInstance(res, presets.InvertedPreset)

    def test_mock_class_roundtrips(self):

        class MockPreset(presets.BasePreset):

            name = 'mock'
            space_scale = 0.91
            time_scale = 5.0
            magnification = 10

        conf = MockPreset(foo='bar')
        conf.bees = False

        config_file = self.tempdir / 'config.ini'
        self.assertFalse(config_file.is_file())

        conf.to_file(config_file)

        with config_file.open('rt') as fp:
            res = fp.read().splitlines()
        exp = [
            '[base]',
            'name = "mock"',
            'space_scale = 0.91',
            'time_scale = 5.0',
            'magnification = 10',
            'foo = "bar"',
            'bees = false',
            '',
        ]
        self.assertEqual(set(res), set(exp))

        self.assertTrue(config_file.is_file())

        new_conf = presets.load_preset(config_file)

        self.assertEqual(new_conf.name, 'mock')
        self.assertEqual(new_conf.space_scale, 0.91)
        self.assertEqual(new_conf.time_scale, 5.0)
        self.assertEqual(new_conf.magnification, 10)
        self.assertEqual(new_conf.foo, 'bar')
        self.assertFalse(new_conf.bees)

    def test_default_presets_roundtrip(self):

        default_presets = presets.list_presets()
        self.assertGreater(len(default_presets), 0)

        for preset in default_presets:
            conf_file = self.tempdir / f'test_{preset}.ini'
            self.assertFalse(conf_file.is_file())

            conf = presets.get_preset(preset)
            conf.to_file(conf_file)
            self.assertTrue(conf_file.is_file())

            new_conf = presets.load_preset(conf_file)

            self.assertEqual(conf, new_conf)

    def test_mock_class_handles_subsections(self):

        class MockPreset(presets.BasePreset):
            segmentation = {
                'foo': 3,
                'bar': False,
            }
            tracking = {
                'foo': 'stuff',
                'bar': [1, 2, 3],
            }

        conf = MockPreset(meshing={'foo': 2.0, 'bar': True})

        config_file = self.tempdir / 'config.ini'
        self.assertFalse(config_file.is_file())

        conf.to_file(config_file)

        with config_file.open('rt') as fp:
            res = fp.read().splitlines()
        exp = [
            '[segmentation]',
            'foo = 3',
            'bar = false',
            '',
            '[tracking]',
            'foo = "stuff"',
            'bar = [1, 2, 3]',
            '',
            '[meshing]',
            'foo = 2.0',
            'bar = true',
            '',
        ]
        self.assertEqual(set(res), set(exp))

        self.assertTrue(config_file.is_file())

        new_conf = presets.load_preset(config_file)

        self.assertEqual(new_conf.segmentation, {'foo': 3, 'bar': False})
        self.assertEqual(new_conf.tracking, {'foo': 'stuff', 'bar': [1, 2, 3]})
        self.assertEqual(new_conf.meshing, {'foo': 2.0, 'bar': True})
