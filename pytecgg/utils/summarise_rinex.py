import polars as pl


def summarise_rinex_data(
    obs: pl.DataFrame,
    nav: dict[str, pl.DataFrame],
) -> None:
    """
    Print a diagnostic summary of GNSS RINEX data.

    This utility provides a high-level overview of the dataset's temporal coverage,
    constellation and signal availability. It is designed to help users
    verify if the loaded observations contain the desidered frequency pairs and tracking
    channels (e.g., C, L, S, I, Q) required for accurate TEC calibration.

    Parameters
    ----------
    obs : pl.DataFrame
        Observation data in long format, as returned by `read_rinex_obs`.
    nav : dict[str, pl.DataFrame]
        Navigation data dictionary keyed by constellation symbol, as returned
        by `read_rinex_nav`.

    Returns
    -------
    None
        The summary is printed directly to the standard output.
    """
    print("=" * 55)
    print("🛰️  GNSS RINEX DATA SUMMARY")
    print("=" * 55)

    epochs = obs["epoch"].unique().sort()
    start_t, end_t = epochs.min(), epochs.max()
    duration = end_t - start_t
    sampling = epochs.diff().median()

    print(f"\n📅 Temporal Coverage (OBS)\n")
    print(f"   - Start:    {start_t}")
    print(f"   - End:      {end_t}")
    print(f"   - Duration: {duration}")
    print(f"   - Sampling: {sampling.total_seconds() if sampling else 'N/A'}s")

    print(f"\n📡 Constellations Breakdown (OBS)\n")
    const_stats = (
        obs.with_columns(pl.col("sv").str.slice(0, 1).alias("const"))
        .group_by("const")
        .agg(
            [
                pl.col("sv").n_unique().alias("svs"),
                pl.col("observable").unique().sort().alias("signals"),
                pl.len().alias("records"),
            ]
        )
        .sort("const")
    )

    header = f"{'Sys':<4} | {'SVs':<4} | {'Total Records':<15} | {'Available Signals'}"
    print(header)
    print("-" * len(header))

    for row in const_stats.to_dicts():
        signals_str = ", ".join(row["signals"])
        print(
            f"{row['const']:<4} | {row['svs']:<4} | {row['records']:<15,} | {signals_str}"
        )

    print(f"\n📖 Ephemeris Coverage (NAV)\n")
    for const, df_nav in nav.items():
        nav_svs = df_nav["sv"].n_unique()
        print(f"   - {const:<8}: {nav_svs:>3} satellites have at least an ephemeris")
