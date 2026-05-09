from unittest.mock import MagicMock, patch

import pytest

from pulsar.act.io import query_path, tod_ids, tods


class TestQueryPath:
    def test_adds_at_prefix(self):
        assert query_path('foo') == '@foo'

    def test_does_not_double_prefix(self):
        assert query_path('@foo') == '@foo'


class TestTodIds:
    def test_returns_db_ids(self):
        from enact import filedb
        fake_db = MagicMock(ids=['a', 'b', 'c'])
        with patch.object(filedb.scans, 'select', return_value=fake_db) as mock_select:
            ids = tod_ids('my_query')
        mock_select.assert_called_once_with('@my_query')
        assert ids == ['a', 'b', 'c']


class TestTods:
    """Exercise the loop in `tods()` — each branch (success, missing dets,
    missing samples, DataMissing exception, id filter, limit) is covered with
    patched enact stubs."""

    def _setup_filedb(self, ids):
        from enact import filedb
        fake_db = MagicMock(ids=ids)
        # Each id maps to an "entry" carrying tag + id (used for band/array).
        filedb.data = {i: {'tag': 'f090', 'id': f'1234.{i}.ar5'} for i in ids}
        return patch.object(filedb.scans, 'select', return_value=fake_db)

    def _act_scan_factory(self, ndet=4, nsamp=16):
        from enact import actscan
        scan = MagicMock()
        scan.dets = list(range(ndet))
        scan.ndet = ndet
        scan.nsamp = nsamp
        return patch.object(actscan, 'ACTScan', return_value=scan)

    def test_yields_a_tod_per_id(self):
        with self._setup_filedb(['t1', 't2']), self._act_scan_factory():
            out = list(tods('q'))
        assert len(out) == 2
        assert out[0].id == 't1'
        assert out[1].id == 't2'

    def test_filters_by_ids_argument(self):
        with self._setup_filedb(['a', 'b', 'c']), self._act_scan_factory():
            out = list(tods('q', ids=['b']))
        assert [t.id for t in out] == ['b']

    def test_honors_limit(self):
        with self._setup_filedb(['a', 'b', 'c', 'd']), self._act_scan_factory():
            out = list(tods('q', limit=2))
        assert [t.id for t in out] == ['a', 'b']

    def test_skips_when_too_few_detectors(self):
        # When dets is None and the scan reports fewer than 2 detectors,
        # io.tods raises DataMissing internally and skips the entry.
        with self._setup_filedb(['only_one']), self._act_scan_factory(ndet=1):
            out = list(tods('q'))
        assert out == []

    def test_skips_when_no_samples(self):
        with self._setup_filedb(['empty']), self._act_scan_factory(ndet=4, nsamp=0):
            out = list(tods('q'))
        assert out == []

    def test_skips_when_data_missing_raised(self):
        from enact import actscan
        from enlib.errors import DataMissing
        with self._setup_filedb(['bad']):
            with patch.object(actscan, 'ACTScan', side_effect=DataMissing('boom')):
                out = list(tods('q'))
        assert out == []

    def test_band_and_array_inferred_from_entry(self):
        with self._setup_filedb(['t1']), self._act_scan_factory():
            out = list(tods('q'))
        assert out[0].band == 'f090'
        # The array is the last 3 chars of the id ('1234.t1.ar5').
        assert out[0].array == 'ar5'
