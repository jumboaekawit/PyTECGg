import pytest
from pytecgg.context import GNSSContext


@pytest.mark.parametrize(
    "input_systems, expected",
    [
        (["GPS", "Galileo"], ["G", "E"]),
        (["gps", "GALILEO", "BEIDOU"], ["G", "E", "C"]),
        (["QZSS"], ["J"]),
        (["Glonass"], ["R"]),
        (["G", "GPS", "E"], ["G", "E"]),
    ],
)
def test_systems_normalization(rec_pos, input_systems, expected):
    """Verify that GNSS systems are normalized correctly."""
    ctx = GNSSContext(rec_pos, "TEST", "3.04", systems=input_systems)
    assert sorted(ctx.systems) == sorted(expected)


def test_invalid_system(rec_pos):
    """Verify that an invalid GNSS system raises a ValueError."""
    with pytest.raises(ValueError, match="not recognised"):
        GNSSContext(rec_pos, "TEST", "3.04", systems=["Starlink"])


def test_receiver_pos_validation_length():
    """Verify the validation of receiver_pos length."""
    with pytest.raises(
        ValueError, match="'receiver_pos' must be a tuple/list of three floats"
    ):
        GNSSContext((0, 0), "TEST", "3.04", systems=["G"])


@pytest.mark.parametrize("bad_height", [2_000_000, -100, 1_000])
def test_ipp_height_warnings(rec_pos, bad_height):
    """Verify that unusual IPP heights generate a warning."""
    with pytest.warns(UserWarning, match="unusual"):
        GNSSContext(rec_pos, "TEST", "3.04", h_ipp=bad_height, systems=["G"])


def test_symbol_to_name_mapping(rec_pos):
    """
    Verify that the symbol_to_name property returns the correct mapping: GNSS systems within context only.
    """
    ctx = GNSSContext(rec_pos, "TEST", "3.04", systems=["G"])
    mapping = ctx.symbol_to_name
    assert mapping["G"] == "GPS"
    assert "E" not in mapping


def test_init(
    rec_pos,
):
    """Verify that glonass_channels and freq_meta are initialized as empty dicts."""
    ctx = GNSSContext(rec_pos, "TEST", "3.04", systems=["G"])
    assert ctx.glonass_channels == {}
    assert ctx.freq_meta == {}


def test_receiver_name_normalization(
    rec_pos,
):
    """Verify that the receiver name is stripped and truncated."""
    ctx = GNSSContext(rec_pos, "  Torremaggiore ", "3.04", systems=["G"])
    assert ctx.receiver_name == "torr"


@pytest.mark.parametrize(
    "empty_input, expected_error",
    [
        ([], "At least one GNSS system must be specified"),
        ([""], "not recognised"),
        ([" "], "not recognised"),
    ],
)
def test_empty_or_blank_system(rec_pos, empty_input, expected_error):
    """Verify that an empty systems list raises a ValueError."""
    with pytest.raises(ValueError, match=expected_error):
        GNSSContext(rec_pos, "TEST", "3.04", systems=empty_input)


def test_rinex_version_cast(rec_pos):
    """Verify that the RINEX version is cast to a string."""
    ctx = GNSSContext(rec_pos, "TEST", 3.04, systems=["G"])
    assert ctx.rinex_version == "3.04"
