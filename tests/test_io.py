import pulsar

def test_tods():
    result = pulsar.tods("/path/to/tods")
    assert len(list(result)) == 5

def test_tod_ids():
    result = pulsar.tod_ids("/path/to/tods")
    assert result == ["mock_tod_1", "mock_tod_2", "mock_tod_3"]