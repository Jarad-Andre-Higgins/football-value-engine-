"""Tests for entity resolution, the shipped team registry and the audit CLI."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from src import logging_config
from src.entities import (
    DEFAULT_REGISTRY_PATH,
    Team,
    TeamRegistry,
    UnresolvedTeamError,
    audit_names,
    main,
    normalize_name,
    scan_football_data,
    strip_noise,
)

# Every spelling football-data.co.uk is expected to use for the top five leagues,
# 2019/20 to 2026/27. Guards the shipped registry against accidental alias deletions.
FOOTBALL_DATA_NAMES = {
    "England": [
        "Arsenal",
        "Aston Villa",
        "Bournemouth",
        "Brentford",
        "Brighton",
        "Burnley",
        "Chelsea",
        "Coventry",
        "Crystal Palace",
        "Everton",
        "Fulham",
        "Hull",
        "Ipswich",
        "Leeds",
        "Leicester",
        "Liverpool",
        "Luton",
        "Man City",
        "Man United",
        "Newcastle",
        "Norwich",
        "Nott'm Forest",
        "Sheffield United",
        "Southampton",
        "Sunderland",
        "Tottenham",
        "Watford",
        "West Brom",
        "West Ham",
        "Wolves",
    ],
    "Spain": [
        "Alaves",
        "Almeria",
        "Ath Bilbao",
        "Ath Madrid",
        "Barcelona",
        "Betis",
        "Cadiz",
        "Celta",
        "La Coruna",
        "Eibar",
        "Elche",
        "Espanol",
        "Getafe",
        "Girona",
        "Granada",
        "Huesca",
        "Las Palmas",
        "Leganes",
        "Levante",
        "Malaga",
        "Mallorca",
        "Osasuna",
        "Oviedo",
        "Santander",
        "Real Madrid",
        "Sevilla",
        "Sociedad",
        "Valencia",
        "Valladolid",
        "Vallecano",
        "Villarreal",
    ],
    "Italy": [
        "Atalanta",
        "Benevento",
        "Bologna",
        "Brescia",
        "Cagliari",
        "Como",
        "Cremonese",
        "Crotone",
        "Empoli",
        "Fiorentina",
        "Frosinone",
        "Genoa",
        "Inter",
        "Juventus",
        "Lazio",
        "Lecce",
        "Milan",
        "Monza",
        "Napoli",
        "Parma",
        "Pisa",
        "Roma",
        "Salernitana",
        "Sampdoria",
        "Sassuolo",
        "Spal",
        "Spezia",
        "Torino",
        "Udinese",
        "Venezia",
        "Verona",
    ],
    "Germany": [
        "Bielefeld",
        "Augsburg",
        "Bayern Munich",
        "Bochum",
        "Darmstadt",
        "Dortmund",
        "Ein Frankfurt",
        "Elversberg",
        "Fortuna Dusseldorf",
        "Freiburg",
        "Greuther Furth",
        "Hamburg",
        "Heidenheim",
        "Hertha",
        "Hoffenheim",
        "Holstein Kiel",
        "FC Koln",
        "Leverkusen",
        "M'gladbach",
        "Mainz",
        "Paderborn",
        "RB Leipzig",
        "Schalke 04",
        "St Pauli",
        "Stuttgart",
        "Union Berlin",
        "Werder Bremen",
        "Wolfsburg",
    ],
    "France": [
        "Ajaccio",
        "Amiens",
        "Angers",
        "Auxerre",
        "Bordeaux",
        "Brest",
        "Clermont",
        "Dijon",
        "Le Havre",
        "Le Mans",
        "Lens",
        "Lille",
        "Lorient",
        "Lyon",
        "Marseille",
        "Metz",
        "Monaco",
        "Montpellier",
        "Nantes",
        "Nice",
        "Nimes",
        "Paris FC",
        "Paris SG",
        "Reims",
        "Rennes",
        "St Etienne",
        "Strasbourg",
        "Toulouse",
        "Troyes",
    ],
}

SMALL_REGISTRY = [
    Team("manchester-city", "Manchester City", "England", ("Man City", "Man. City")),
    Team(
        "manchester-united", "Manchester United", "England", ("Man United", "Man Utd")
    ),
    Team("arsenal", "Arsenal", "England", ("Arsenal FC",)),
    Team("inter", "Inter Milan", "Italy", ("Inter", "Internazionale")),Team("milan", "AC Milan", "Italy", ("Milan",)),
    Team("koln", "1. FC Köln", "Germany", ("FC Koln", "Koln")),
    Team("paris-fc", "Paris FC", "France"),
    Team("psg", "Paris Saint-Germain", "France", ("Paris SG", "PSG")),
    Team("nacional-uy", "Nacional", "Uruguay"),
    Team("nacional-pt", "Nacional", "Portugal"),
]


@pytest.fixture
def registry() -> TeamRegistry:
    return TeamRegistry(SMALL_REGISTRY)


@pytest.fixture
def isolated_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep CLI tests from writing to the real logs folder."""
    root = logging.getLogger()
    original_level, original_hook = root.level, sys.excepthook
    monkeypatch.setattr(logging_config, "LOG_DIR", tmp_path / "logs")
    yield
    logging_config._remove_engine_handlers(root)
    root.setLevel(original_level)
    sys.excepthook = original_hook


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Atlético Madrid", "atletico madrid"),
        ("Nott'm Forest", "nottm forest"),
        ("M'gladbach", "mgladbach"),
        ("St. Étienne", "st etienne"),
        ("Brighton & Hove Albion", "brighton and hove albion"),
        ("  Man   City ", "man city"),
        ("Bodø/Glimt", "bodo glimt"),
        ("Straße", "strasse"),
        ("1. FC Köln", "1 fc koln"),
    ],
)
def test_normalize_name(raw: str, expected: str) -> None:
    assert normalize_name(raw) == expected


@pytest.mark.parametrize(
    ("normalized", "expected"),
    [
        ("1 fc koln", "koln"),
        ("mainz 05", "mainz"),
        ("arsenal fc", "arsenal"),
        ("fc", "fc"),
    ],
)
def test_strip_noise(normalized: str, expected: str) -> None:
    assert strip_noise(normalized) == expected


# --------------------------------------------------------------------------- #
# Resolution rules
# --------------------------------------------------------------------------- #
def test_resolves_names_aliases_and_cosmetic_variants(registry: TeamRegistry) -> None:
    assert registry.resolve("Manchester City") == "manchester-city"
    assert registry.resolve("MAN CITY") == "manchester-city"
    assert registry.resolve("man. city") == "manchester-city"
    assert registry.resolve("Arsenal FC") == "arsenal"
    assert registry.resolve("Arsenal") == "arsenal"  # via the legal-token-free form
    assert registry.resolve("1. FC Köln") == "koln"
    assert registry.resolve("Internazionale") == "inter"


def test_unknown_and_vague_names_are_never_guessed(registry: TeamRegistry) -> None:
    assert registry.resolve("Man") is None
    assert registry.resolve("Manchester") is None
    assert registry.resolve("Paris") is None  # PSG or Paris FC: refuse to pick
    assert registry.resolve("Man Cit") is None  # a typo is a suggestion, not a match
    assert registry.resolve("") is None


def test_similar_clubs_stay_distinct(registry: TeamRegistry) -> None:
    assert registry.resolve("Man City") != registry.resolve("Man United")
    assert registry.resolve("Inter") != registry.resolve("Milan")
    assert registry.resolve("Paris FC") != registry.resolve("Paris SG")


def test_country_scopes_shared_spellings(registry: TeamRegistry) -> None:
    assert registry.resolve("Nacional") is None  # ambiguous without a country
    assert registry.resolve("Nacional", country="Uruguay") == "nacional-uy"
    assert registry.resolve("Nacional", country="Portugal") == "nacional-pt"
    assert registry.resolve("Man City", country="Italy") is None  # wrong country


def test_name_of_returns_the_display_name(registry: TeamRegistry) -> None:
    assert registry.name_of("koln") == "1. FC Köln"


# --------------------------------------------------------------------------- #
# Suggestions and require()
# --------------------------------------------------------------------------- #
def test_suggestions_rank_the_closest_team_first(registry: TeamRegistry) -> None:
    best = registry.suggest("Manchestr Utd")[0]
    assert best.team_id == "manchester-united"
    assert 0.5 <= best.score <= 1.0


def test_suggestions_respect_country(registry: TeamRegistry) -> None:
    ids = [s.team_id for s in registry.suggest("Man Cty", country="England")]
    assert "manchester-city" in ids
    assert not any(team_id in ids for team_id in ("inter", "milan"))


def test_require_raises_with_suggestions(registry: TeamRegistry) -> None:
    with pytest.raises(UnresolvedTeamError, match="Closest: Manchester United"):
        registry.require("Manchestr Utd")
    assert registry.require("Man Utd") == "manchester-united"


# --------------------------------------------------------------------------- #
# Registry validation
# --------------------------------------------------------------------------- #
def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate team id"):
        TeamRegistry([Team("a", "Alpha", "X"), Team("a", "Beta", "X")])


def test_a_spelling_claimed_by_two_teams_in_one_country_is_rejected() -> None:
    teams = [Team("a", "Alpha", "X", ("Shared",)), Team("b", "Beta", "X", ("Shared",))]
    with pytest.raises(ValueError, match="Ambiguous spellings"):
        TeamRegistry(teams)


def test_unusable_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="unusable"):
        TeamRegistry([Team("a", "Alpha", "X", ("...",))])


def test_from_yaml_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "teams.yaml"
    path.write_text(
        "teams:\n"
        '  - {id: arsenal, country: England, name: "Arsenal", aliases: ["Arsenal FC"]}\n'
        "  - {id: chelsea, country: England, name: Chelsea}\n",
        encoding="utf-8",
    )
    loaded = TeamRegistry.from_yaml(path)
    assert len(loaded) == 2
    assert loaded.resolve("arsenal fc") == "arsenal"


@pytest.mark.parametrize(
    "text",
    ["just: a mapping\n", "teams:\n  - {id: a, country: X}\n", "teams: not-a-list\n"],
)
def test_malformed_yaml_is_rejected(tmp_path: Path, text: str) -> None:
    path = tmp_path / "teams.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        TeamRegistry.from_yaml(path)


# --------------------------------------------------------------------------- #
# The shipped registry
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def shipped() -> TeamRegistry:
    return TeamRegistry.from_yaml(DEFAULT_REGISTRY_PATH)


def test_shipped_registry_resolves_every_expected_football_data_name(
    shipped: TeamRegistry,
) -> None:
    for country, names in FOOTBALL_DATA_NAMES.items():
        ids = [shipped.resolve(name, country=country) for name in names]
        missing = [n for n, team_id in zip(names, ids) if team_id is None]
        assert not missing, f"{country}: unresolved {missing}"
        assert len(set(ids)) == len(ids), f"{country}: two spellings share one team"


@pytest.mark.parametrize(
    ("raw", "country", "expected"),
    [
        ("Man City", "England", "manchester-city"),
        ("Man United", "England", "manchester-united"),
        ("Nott'm Forest", "England", "nottingham-forest"),
        ("Ath Madrid", "Spain", "atletico-madrid"),
        ("Ath Bilbao", "Spain", "athletic-club"),
        ("Sociedad", "Spain", "real-sociedad"),
        ("Inter", "Italy", "inter"),
        ("Milan", "Italy", "milan"),
        ("Ein Frankfurt", "Germany", "eintracht-frankfurt"),
        ("M'gladbach", "Germany", "gladbach"),
        ("FC Koln", "Germany", "koln"),
        ("Paris SG", "France", "paris-saint-germain"),
        ("Paris FC", "France", "paris-fc"),
        ("St Etienne", "France", "st-etienne"),
    ],
)
def test_shipped_registry_spot_checks(
    shipped: TeamRegistry, raw: str, country: str, expected: str
) -> None:
    assert shipped.resolve(raw, country=country) == expected


@pytest.mark.parametrize(
    "vague", ["Real", "Union", "Racing", "Paris", "Atletico", "Man"]
)
def test_shipped_registry_refuses_vague_names(shipped: TeamRegistry, vague: str
) -> None:
    assert shipped.resolve(vague) is None


# --------------------------------------------------------------------------- #
# Scanning and auditing data files
# --------------------------------------------------------------------------- #
def write_results(directory: Path, name: str, rows: list[tuple[str, str]]) -> None:
    lines = ["Div,Date,HomeTeam,AwayTeam,FTHG,FTAG"]
    lines += [f"E0,15/08/2025,{home},{away},1,0" for home, away in rows]
    (directory / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_scan_counts_spellings_and_skips_foreign_files(tmp_path: Path) -> None:
    write_results(
        tmp_path, "E0_2526.csv", [("Man City", "Arsenal"), ("Arsenal", "Man City")]
    )
    (tmp_path / "notes.csv").write_text("a,b\n", encoding="utf-8")
    scan = scan_football_data(tmp_path)
    counts = {row.raw_name: row.matches for row in scan.itertuples()}
    assert counts == {"Man City": 2, "Arsenal": 2}
    assert set(scan["league"]) == {"E0"}


def test_audit_flags_unresolved_names_with_suggestions(
    tmp_path: Path, registry: TeamRegistry
) -> None:
    write_results(
        tmp_path, "E0_2526.csv", [("Man City", "Arsenal"), ("Man Utd", "Manchestr Utd")]
    )
    result = audit_names(registry, scan_football_data(tmp_path))
    assert not result.ok
    assert result.n_names == 4
    assert result.n_resolved == 3
    row = result.unresolved.iloc[0]
    assert row["raw_name"] == "Manchestr Utd"
    assert "Manchester United" in row["suggestions"]


def test_audit_flags_two_spellings_of_one_team_in_one_season(
    tmp_path: Path, registry: TeamRegistry
) -> None:
    write_results(
        tmp_path,
        "E0_2526.csv",
        [("Man City", "Manchester City"), ("Arsenal", "Man City")],
    )
    result = audit_names(registry, scan_football_data(tmp_path))
    assert result.unresolved.empty
    assert len(result.collisions) == 1
    assert "manchester-city" in result.collisions[0]


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def write_registry(path: Path) -> Path:
    path.write_text(
        "teams:\n"
        "  - {id: arsenal, country: England, name: Arsenal, aliases: []}\n"
        '  - {id: manchester-city, country: England, name: "Manchester City", aliases: ["Man City"]}\n',
        encoding="utf-8",
    )
    return path


def cli_args(tmp_path: Path, raw_dir: Path) -> list[str]:
    return [
        "--registry",
        str(tmp_path / "teams.yaml"),
        "--raw-dir",
        str(raw_dir),
        "--queue",
        str(tmp_path / "queue.csv"),
    ]


@pytest.mark.usefixtures("isolated_logging")
def test_cli_returns_zero_when_everything_resolves(tmp_path: Path) -> None:
    write_registry(tmp_path / "teams.yaml")
    raw = tmp_path / "raw"
    raw.mkdir()
    write_results(raw, "E0_2526.csv", [("Man City", "Arsenal")])
    assert main(cli_args(tmp_path, raw)) == 0
    assert not (tmp_path / "queue.csv").exists()


@pytest.mark.usefixtures("isolated_logging")
def test_cli_writes_a_review_queue_and_returns_one(tmp_path: Path) -> None:
    write_registry(tmp_path / "teams.yaml")
    raw = tmp_path / "raw"
    raw.mkdir()
    write_results(raw, "E0_2526.csv", [("Man City", "Arsenl")])
    assert main(cli_args(tmp_path, raw)) == 1
    queue = pd.read_csv(tmp_path / "queue.csv")
    assert list(queue["raw_name"]) == ["Arsenl"]
    assert "Arsenal" in queue.loc[0, "suggestions"]


@pytest.mark.usefixtures("isolated_logging")
def test_cli_returns_one_when_there_is_no_data(tmp_path: Path) -> None:
    write_registry(tmp_path / "teams.yaml")
    raw = tmp_path / "raw"
    raw.mkdir()
    assert main(cli_args(tmp_path, raw)) == 1


@pytest.mark.usefixtures("isolated_logging")
def test_cli_returns_two_when_the_registry_is_missing(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    assert main(cli_args(tmp_path, raw)) == 2
