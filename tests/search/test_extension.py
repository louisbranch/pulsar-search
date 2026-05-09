import os
from unittest.mock import MagicMock, patch

import matplotlib
import pytest

matplotlib.use('Agg')

from pulsar.search import extension as ext_module
from pulsar.search.extension import (
    ExtensionManager,
    PlottingExtension,
    ProfilingExtension,
    SamplingExtension,
    TimingExtension,
)
from ..test_mocks import create_mock_tod


@pytest.fixture
def manager():
    return ExtensionManager()


@pytest.fixture
def tod():
    return create_mock_tod(id='ext_tod_1')


class TestExtensionManager:
    def test_register_extension(self, manager):
        a, b = MagicMock(), MagicMock()
        manager.register_extension(a)
        manager.register_extension(b)
        assert manager.extensions == [a, b]

    def test_wrap_step_orders_before_call_after(self, manager, tod):
        ext = MagicMock()
        manager.register_extension(ext)
        calls = []
        ext.before_step.side_effect = lambda *a, **kw: calls.append('before')
        ext.after_step.side_effect = lambda *a, **kw: calls.append('after')

        @manager.wrap_step('step1', None, tod)
        def work(x):
            calls.append('work')
            return x + 1

        result = work(41)
        assert result == 42
        assert calls == ['before', 'work', 'after']
        ext.before_step.assert_called_once_with('step1', None, tod)
        ext.after_step.assert_called_once_with('step1', None, tod)


class TestSamplingExtension:
    def test_should_sample_call_is_deterministic(self, tod):
        a = SamplingExtension(sampling_rate=0.5)
        b = SamplingExtension(sampling_rate=0.5)
        a.should_sample_call(tod)
        b.should_sample_call(tod)
        assert a.should_sample == b.should_sample
        assert a.detectors[tod.id] == b.detectors[tod.id]

    def test_should_sample_call_caches_detector(self, tod):
        s = SamplingExtension(sampling_rate=1.0)
        s.should_sample_call(tod)
        first = s.detectors[tod.id]
        # Second call returns cached value rather than re-assigning
        assert s.should_sample_call(tod) == first
        assert s.detector_index(tod) == first

    def test_before_step_with_steps_none_always_runs(self, tod):
        s = SamplingExtension(sampling_rate=1.0, steps=None)
        s.before_step('any_step', None, tod)
        assert s.should_sample is True

    def test_before_step_skips_steps_not_in_list(self, tod):
        s = SamplingExtension(sampling_rate=1.0, steps=['filter'])
        s.before_step('flux', None, tod)
        # should_sample remains the default False because the branch was skipped
        assert s.should_sample is False
        # Detector cache also untouched
        assert tod.id not in s.detectors

    def test_after_step_resets_should_sample_only_for_listed_steps(self, tod):
        s = SamplingExtension(sampling_rate=1.0, steps=['filter'])
        s.should_sample = True
        s.after_step('flux', None, tod)
        assert s.should_sample is True
        s.after_step('filter', None, tod)
        assert s.should_sample is False


class TestTimingExtension:
    def test_logs_when_sampled(self, tod):
        t = TimingExtension(sampling_rate=1.0)
        with patch('time.time', side_effect=[100.0, 102.5]), \
             patch.object(ext_module.log, 'debug') as mock_debug:
            t.before_step('s', None, tod)
            t.after_step('s', None, tod)
        assert mock_debug.called
        assert '2.50' in mock_debug.call_args[0][0]

    def test_noop_when_not_sampled(self, tod):
        t = TimingExtension(sampling_rate=0.0)
        with patch.object(ext_module.log, 'debug') as mock_debug:
            t.before_step('s', None, tod)
            t.after_step('s', None, tod)
        mock_debug.assert_not_called()


class TestProfilingExtension:
    def test_raises_when_psutil_missing(self):
        with patch.object(ext_module, 'psutil', None):
            with pytest.raises(ImportError):
                ProfilingExtension(sampling_rate=1.0)

    def test_records_memory_delta_when_sampled(self, tod):
        fake_psutil = MagicMock()
        fake_psutil.Process.return_value.memory_info.side_effect = [
            MagicMock(rss=10 * 1024 ** 2),
            MagicMock(rss=12 * 1024 ** 2),
        ]
        with patch.object(ext_module, 'psutil', fake_psutil), \
             patch.object(ext_module.log, 'debug') as mock_debug:
            p = ProfilingExtension(sampling_rate=1.0)
            p.before_step('s', None, tod)
            p.after_step('s', None, tod)
        assert mock_debug.called
        assert '2.00 MB' in mock_debug.call_args[0][0]


class TestPlottingExtension:
    def test_after_step_writes_png_and_increments_count(self, tmpdir, tod):
        tod.sampling_rate = 400.0
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            p = PlottingExtension(sampling_rate=1.0, output_path=str(tmpdir))
            p.before_step('flux', None, tod)
            p.after_step('flux', None, tod)
            assert mock_savefig.called
            saved_path = mock_savefig.call_args[0][0]
            assert os.path.dirname(saved_path) == str(tmpdir)
            assert saved_path.endswith('.png')
            assert p.count == 1

    def test_filename_includes_seed_when_scenario_provided(self, tmpdir, tod):
        tod.sampling_rate = 400.0
        scenario = MagicMock()
        scenario.title = 'sc'
        scenario.seed = 42
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            p = PlottingExtension(sampling_rate=1.0, before=True, after=False,
                                  output_path=str(tmpdir))
            p.before_step('flux', scenario, tod)
            saved_path = mock_savefig.call_args[0][0]
            assert '42_' in os.path.basename(saved_path)
