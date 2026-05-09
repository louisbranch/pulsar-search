from unittest.mock import patch

import pytest

from pulsar.act.act import ACT


class TestActInit:
    def test_dataset_is_set_on_config(self):
        from enlib import config
        with patch.object(config, 'set') as mock_set:
            ACT(dataset='dr6v4')
        # First call sets the dataset; nothing else (root/file_override) is touched.
        mock_set.assert_called_once_with('dataset', 'dr6v4')

    def test_root_is_set_when_provided(self):
        from enlib import config
        with patch.object(config, 'set') as mock_set:
            ACT(dataset='dr6v4', root='/data/act')
        # Two calls: dataset + root.
        calls = [c.args for c in mock_set.call_args_list]
        assert ('dataset', 'dr6v4') in calls
        assert ('root', '/data/act') in calls

    def test_pointing_file_uses_at_prefix(self):
        from enlib import config
        with patch.object(config, 'set') as mock_set:
            ACT(dataset='dr6v4', pointing_file='/path/to/p.txt')
        calls = [c.args for c in mock_set.call_args_list]
        assert ('file_override', '@/path/to/p.txt') in calls

    def test_verbose_flag_stored(self):
        act = ACT(verbose=True)
        assert act.verbose is True

    def test_pointing_model_returns_pmat_class(self):
        from pulsar.act.pointing_models import PmatTotTransient
        act = ACT()
        assert act.pointing_model() is PmatTotTransient


class TestActDelegations:
    def test_tods_forwards_to_module_function(self):
        # ACT.tods just delegates to pulsar.act.io.tods. Patch that function
        # and assert ACT calls it with the expected kwargs.
        with patch('pulsar.act.act.tods') as mock_tods:
            mock_tods.return_value = iter([])
            act = ACT(verbose=True)
            list(act.tods('@some_query', limit=5, dets=[1, 2], ids=['a']))
        mock_tods.assert_called_once_with('@some_query', limit=5, dets=[1, 2], ids=['a'], verbose=True)

    def test_tod_ids_forwards_to_module_function(self):
        with patch('pulsar.act.act.tod_ids') as mock_tod_ids:
            mock_tod_ids.return_value = ['id1', 'id2']
            act = ACT()
            assert act.tod_ids('@q') == ['id1', 'id2']
            mock_tod_ids.assert_called_once_with('@q')

    def test_create_map_forwards_to_module_function(self):
        with patch('pulsar.act.act.create_map') as mock_create_map:
            ACT().create_map('arg', kw='value')
        mock_create_map.assert_called_once_with('arg', kw='value')

    def test_plot_map_forwards_to_module_function(self):
        with patch('pulsar.act.act.plot_map') as mock_plot_map:
            ACT().plot_map('arg')
        mock_plot_map.assert_called_once_with('arg')
