from datetime import datetime, timedelta, timezone

import pytest
import polars as pl

from pytecgg.context import GNSSContext
from pytecgg.parsing import read_rinex_nav
from pytecgg.satellites.ephemeris import (
    _KEPLERIAN_CADENCE_S,
    _drop_week_rollover_bug,
    _get_gps_time,
    prepare_ephemeris,
)
from pytecgg.satellites.positions import satellite_coordinates


def test_get_gps_time():
    dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    gps_week, gps_seconds = _get_gps_time(dt)
    assert gps_week == 2295
    assert gps_seconds == 129600.0


def test_get_gps_time_type_safety():
    """
    Verify that _get_gps_time raises TypeError when provided with an invalid type.
    """
    with pytest.raises(TypeError):
        _get_gps_time(123456789)


def test_prepare_ephemeris_nav_v3(nav_v3_file):
    nav_data = read_rinex_nav(nav_v3_file)

    df_gps = nav_data.get("GPS")
    assert df_gps is not None
    assert isinstance(df_gps["epoch"].dtype, pl.Datetime)

    ctx = GNSSContext(
        receiver_pos=(0.0, 0.0, 0.0),
        receiver_name="TEST",
        rinex_version="3.04",
        systems=["G"],
    )

    ephemeris = prepare_ephemeris(nav_data, ctx)
    assert isinstance(ephemeris, dict)

    for sat, eph in ephemeris.items():
        assert sat.startswith("G")
        assert len(sat) == 3
        # Keplerian SVs now carry the full list of broadcast records so
        # downstream code can pick the one nearest to obs_time.
        assert isinstance(eph, list) and eph, f"expected non-empty list for {sat}"
        for record in eph:
            assert record["datetime"].tzinfo is not None
            assert record["constellation"] == "GPS"


def test_qzss_navigation_to_satellite_coordinates(nav_v3_file):
    nav_data = read_rinex_nav(nav_v3_file)

    qzss_data = nav_data.get("QZSS")
    assert qzss_data is not None
    assert not qzss_data.is_empty()

    ctx = GNSSContext(
        receiver_pos=(0.0, 0.0, 0.0),
        receiver_name="TEST",
        rinex_version="3.04",
        systems=["QZSS"],
    )
    ephemeris = prepare_ephemeris(nav_data, ctx)

    assert ephemeris
    assert all(sv.startswith("J") for sv in ephemeris)

    sv, records = next(iter(ephemeris.items()))
    assert isinstance(records, list) and records
    record = records[0]
    assert record["constellation"] == "QZSS"

    coordinates = satellite_coordinates(
        pl.Series("sv", [sv]),
        pl.Series("epoch", [record["datetime"]]),
        ephemeris,
    )
    assert coordinates.height == 1
    assert coordinates.select(
        pl.all_horizontal(
            pl.col("sat_x").is_finite(),
            pl.col("sat_y").is_finite(),
            pl.col("sat_z").is_finite(),
        )
    ).item()


def _bad_row(epoch: datetime, cadence: float) -> dict:
    return {"epoch": epoch, "toe": cadence}


def _good_row(epoch: datetime, toe: float) -> dict:
    return {"epoch": epoch, "toe": toe}


def test_drop_week_rollover_bug_drops_corrupted_beidou_row():
    """At BDT Sunday 00:00:00, toe = cadence is the rinex-crate bug signature."""
    cadence = _KEPLERIAN_CADENCE_S["BEIDOU"]
    rollover_sunday = datetime(2025, 1, 5, 0, 0, 0)  # Sunday 00:00:00
    df = pl.DataFrame(
        [
            _bad_row(rollover_sunday, cadence),
            _good_row(datetime(2025, 1, 5, 1, 0, 0), cadence),
            _good_row(datetime(2025, 1, 5, 2, 0, 0), 7200.0),
        ]
    )

    with pytest.warns(UserWarning, match="week rollover"):
        out = _drop_week_rollover_bug(df, "BEIDOU", "C01")

    assert out.height == 2
    assert rollover_sunday not in out["epoch"].to_list()


def test_drop_week_rollover_bug_leaves_other_records_alone():
    """A toe equal to cadence on any non-rollover epoch must NOT be dropped."""
    cadence = _KEPLERIAN_CADENCE_S["GALILEO"]
    df = pl.DataFrame(
        [
            # Sunday but not at 00:00:00 -> keep
            _bad_row(datetime(2025, 1, 5, 0, 10, 0), cadence),
            # Monday at 00:00:00 -> keep (weekday() != 7)
            _bad_row(datetime(2025, 1, 6, 0, 0, 0), cadence),
        ]
    )

    out = _drop_week_rollover_bug(df, "GALILEO", "E01")

    assert out.height == 2


def test_drop_week_rollover_bug_noop_for_glonass():
    """GLONASS has no cadence entry, so the filter is a no-op."""
    df = pl.DataFrame([{"epoch": datetime(2025, 1, 5, 0, 0, 0), "toe": 600.0}])

    out = _drop_week_rollover_bug(df, "GLONASS", "R01")

    assert out.height == 1


def test_prepare_ephemeris_populates_leap_seconds(nav_v3_file):
    """Each record must carry leap_seconds, defaulting when ctx leaves it None."""
    nav_data = read_rinex_nav(nav_v3_file)

    ctx_default = GNSSContext(
        receiver_pos=(0.0, 0.0, 0.0),
        receiver_name="TEST",
        rinex_version="3.04",
        systems=["G"],
    )
    ephem_default = prepare_ephemeris(nav_data, ctx_default)
    any_sv = next(iter(ephem_default))
    assert ephem_default[any_sv][0]["leap_seconds"] == 18  # DEFAULT_LEAP_SECONDS_UTC_GPST

    ctx_override = GNSSContext(
        receiver_pos=(0.0, 0.0, 0.0),
        receiver_name="TEST",
        rinex_version="3.04",
        systems=["G"],
        leap_seconds=37,
    )
    ephem_override = prepare_ephemeris(nav_data, ctx_override)
    assert ephem_override[any_sv][0]["leap_seconds"] == 37


@pytest.mark.parametrize(
    "constellation, symbol",
    [("NAVIC", "I"), ("SBAS", "S")],
)
def test_supported_constellation_navigation_to_coordinates(
    nav_v3_file, constellation, symbol
):
    nav_data = read_rinex_nav(nav_v3_file)
    assert constellation in nav_data

    ctx = GNSSContext(
        (0.0, 0.0, 0.0), "TEST", "3.05", systems=[constellation]
    )
    ephemeris = prepare_ephemeris(nav_data, ctx)

    assert ephemeris
    sv, records = next(iter(ephemeris.items()))
    assert sv.startswith(symbol)
    record = records[0]
    coordinates = satellite_coordinates(
        pl.Series("sv", [sv]),
        pl.Series("epoch", [record["datetime"]]),
        ephemeris,
    )

    assert coordinates.select(
        pl.all_horizontal(
            pl.col("sat_x").is_finite(),
            pl.col("sat_y").is_finite(),
            pl.col("sat_z").is_finite(),
        )
    ).item()


def test_navic_ephemeris_to_satellite_coordinates():
    epoch = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
    orbital_fields = {
        "toe": 129600.0,
        "sqrta": 6493.0,
        "deltaN": 0.0,
        "m0": 0.1,
        "e": 0.01,
        "omega": 0.2,
        "cuc": 0.0,
        "cus": 0.0,
        "crc": 0.0,
        "crs": 0.0,
        "cic": 0.0,
        "cis": 0.0,
        "i0": 0.5,
        "idot": 0.0,
        "omega0": 1.0,
        "omegaDot": 0.0,
    }
    nav = {"NAVIC": pl.DataFrame([{"epoch": epoch, "sv": "1", **orbital_fields}])}
    ctx = GNSSContext((0.0, 0.0, 0.0), "TEST", "3.04", systems=["I"])

    ephemeris = prepare_ephemeris(nav, ctx)
    coordinates = satellite_coordinates(
        pl.Series("sv", ["I01"]),
        pl.Series("epoch", [epoch]),
        ephemeris,
    )

    assert ephemeris["I01"][0]["constellation"] == "NAVIC"
    assert coordinates.select(
        pl.all_horizontal(
            pl.col("sat_x").is_finite(),
            pl.col("sat_y").is_finite(),
            pl.col("sat_z").is_finite(),
        )
    ).item()


def test_sbas_ephemeris_uses_quadratic_ecef_propagation():
    epoch = datetime(2025, 1, 1, 0, tzinfo=timezone.utc)
    nav = {
        "SBAS": pl.DataFrame(
            [
                {
                    "epoch": epoch,
                    "sv": "120",
                    "satPosX": 20_000_000.0,
                    "satPosY": 10_000_000.0,
                    "satPosZ": 15_000_000.0,
                    "velX": 10.0,
                    "velY": -20.0,
                    "velZ": 30.0,
                    "accelX": 1.0,
                    "accelY": 2.0,
                    "accelZ": -1.0,
                }
            ]
        )
    }
    ctx = GNSSContext((0.0, 0.0, 0.0), "TEST", "3.04", systems=["SBAS"])
    ephemeris = prepare_ephemeris(nav, ctx)

    coordinates = satellite_coordinates(
        pl.Series("sv", ["S120"]),
        pl.Series("epoch", [epoch + timedelta(seconds=10)]),
        ephemeris,
    )

    assert coordinates["sat_x"][0] == pytest.approx(20_000_150.0)
    assert coordinates["sat_y"][0] == pytest.approx(9_999_900.0)
    assert coordinates["sat_z"][0] == pytest.approx(15_000_250.0)


def test_prepare_ephemeris_warns_when_required_orbit_fields_are_missing():
    nav = {
        "NAVIC": pl.DataFrame(
            [{"epoch": datetime(2025, 1, 1), "sv": "1", "toe": 0.0}]
        )
    }
    ctx = GNSSContext((0.0, 0.0, 0.0), "TEST", "3.04", systems=["I"])

    with pytest.warns(RuntimeWarning, match="required position fields"):
        assert prepare_ephemeris(nav, ctx) == {}
