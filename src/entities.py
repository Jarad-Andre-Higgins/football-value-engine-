"""Entity resolution: map every spelling of a team name to one canonical team id.

Data sources and bookmakers spell clubs differently ("Man City", "Man. City",
"Manchester City FC"). Joining on raw names silently loses matches or, worse,
mixes up teams, so every name goes through this registry first. The rules are
deliberately strict, because one wrong merge poisons every rating built on top:

1. Names are compared only after normalisation: case, accents, punctuation and
   a few legal tokens such as "FC" are ignored.
2. A name matches only if exactly one team claims it (within the league's
   country when known). Unknown or ambiguous names resolve to `None`; the
   registry never guesses.
3. Fuzzy matching produces *suggestions* for a human. The fix is always a new
   alias in `data/reference/teams.yaml`.

Audit the downloaded data with::

    python -m src.entities
"""

from __future__ import annotations

import argparse
import difflib
import logging
import re
import sys
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import yaml

from src.config import DATA_DIR, PROCESSED_DIR
from src.ingest import FOOTBALL_DATA_DIR
from src.leagues import LEAGUES
from src.logging_config import setup_logging

logger = logging.getLogger("src.entities")

DEFAULT_REGISTRY_PATH: Final[Path] = DATA_DIR / "reference" / "teams.yaml"
DEFAULT_QUEUE_PATH: Final[Path] = PROCESSED_DIR / "unresolved_team_names.csv"

# Legal-form and prefix tokens that never tell two top-flight clubs apart.
NOISE_TOKENS: Final[frozenset[str]] = frozenset(
    {"fc", "cf", "afc", "sc", "ac", "as", "ssc", "us", "ud", "cd", "ca", "rc"}
    | {"rcd", "sv", "vfb", "vfl", "fsv", "tsg"}
)
_LETTER_MAP: Final[dict[int, str]] = str.maketrans(
    {"ø": "o", "æ": "ae", "œ": "oe", "đ": "d", "ł": "l", "ı": "i", "ð": "d", "þ": "th"}
)
_FILE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<league>[A-Z0-9]+)_(?P<season>\d{4})\.csv$"
)


def normalize_name(raw: str) -> str:
    """Return the comparison form of a team name.

    Lower-cases, removes accents, drops apostrophes and dots, turns every other
    punctuation run into one space and spells `&` as `and`. For example
    `"Nott'm Forest"` becomes `"nottm forest"` and `"Atlético Madrid"`
    becomes `"atletico madrid"`.
    """
    text = unicodedata.normalize("NFKD", raw)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold().translate(_LETTER_MAP)
    text = text.replace("&", " and ")
    text = re.sub(r"[’'`.]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def strip_noise(normalized: str) -> str:
    """Drop legal tokens (``fc``, `ac`...) and bare numbers from a normalised name.

    `"1 fc koln"` becomes `"koln"` and `"mainz 05"` becomes `"mainz"`.
    A name made only of noise is returned unchanged.
    """
    tokens = [
        token
        for token in normalized.split()
        if token not in NOISE_TOKENS and not token.isdigit()
    ]
    return " ".join(tokens) or normalized


@dataclass(frozen=True)
class Team:
    """One club in the registry.

    Attributes:
        id: Stable ASCII slug, for example `manchester-city`.
        name: Display name.
        country: Country the club belongs to; scopes matching.
        aliases: Other spellings seen in data sources and bookmakers.
    """

    id: str
    name: str
    country: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Suggestion:
    """A possible match for an unresolved name, for a human to review."""

    team_id: str
    name: str
    score: float


class UnresolvedTeamError(ValueError):
    """Raised by :meth:`TeamRegistry.require` for names that cannot be resolved."""


class TeamRegistry:
    """Strict lookup from team-name spellings to canonical team ids.

    Args:
        teams: The clubs to register.

    Raises:ValueError: On duplicate ids, unusable names, or a spelling claimed by
            two teams of the same country.
    """

    def __init__(self, teams: Iterable[Team]) -> None:
        self._teams: dict[str, Team] = {}
        index: dict[str, list[str]] = defaultdict(list)
        for team in teams:
            if team.id in self._teams:
                raise ValueError(f"Duplicate team id '{team.id}'")
            self._teams[team.id] = team
            for spelling in (team.name, *team.aliases):
                key = normalize_name(spelling)
                if not key:
                    raise ValueError(
                        f"Team '{team.id}' has an unusable name: {spelling!r}"
                    )
                if team.id not in index[key]:
                    index[key].append(team.id)
        self._index: dict[str, list[str]] = dict(index)
        self._reject_ambiguous_spellings()

    @classmethod
    def from_yaml(cls, path: Path) -> TeamRegistry:
        """Load the registry from `teams.yaml`.

        Raises:
            ValueError: If the file is malformed or contains ambiguous aliases.
        """
        with path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
        try:
            entries = list(document["teams"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path}: expected a top-level 'teams' list") from exc
        teams: list[Team] = []
        for position, entry in enumerate(entries, start=1):
            try:
                aliases = tuple(str(alias) for alias in entry.get("aliases", []))
                teams.append(
                    Team(
                        str(entry["id"]),
                        str(entry["name"]),
                        str(entry["country"]),
                        aliases,
                    )
                )
            except (KeyError, TypeError, AttributeError) as exc:
                raise ValueError(
                    f"{path}: team entry #{position} is malformed ({exc!r})"
                ) from exc
        return cls(teams)

    def __len__(self) -> int:
        return len(self._teams)

    @property
    def teams(self) -> list[Team]:
        """All registered teams."""
        return list(self._teams.values())

    def name_of(self, team_id: str) -> str:
        """Return the display name of `team_id`."""
        return self._teams[team_id].name

    def resolve(self, raw_name: str, *, country: str | None = None) -> str | None:
        """Return the team id for `raw_name`, or `None` if it is not certain.

        Tries the normalised spelling first, then the spelling without legal
        tokens. A spelling claimed by several teams (after scoping to
        `country``) is ambiguous and also returns ``None`.

        Args:
            raw_name: The name exactly as the source wrote it.
            country: Restrict matching to teams of this country, when known.
        """
        normalized = normalize_name(raw_name)
        for key in dict.fromkeys((normalized, strip_noise(normalized))):
            candidates = self._index.get(key, [])
            if country is not None:
                candidates = [
                    c for c in candidates if self._teams[c].country == country
                ]
            if len(candidates) == 1:
                return candidates[0]
            if candidates:
                return None
        return None

    def require(self, raw_name: str, *, country: str | None = None) -> str:
        """Like :meth:`resolve`, but raise with suggestions when it cannot.

        Raises:
            UnresolvedTeamError: If the name is unknown or ambiguous.
        """
        team_id = self.resolve(raw_name, country=country)
        if team_id is not None:
            return team_id
        close = self.suggest(raw_name, country=country)
        hint = "; ".join(f"{s.name} ({s.score:.2f})" for s in close) or "no close match"
        scope = f" in {country}" if country else ""
        raise UnresolvedTeamError(f"Cannot resolve team '{raw_name}'{scope}. Closest: {hint}"
        )

    def suggest(
        self,
        raw_name: str,
        *,
        country: str | None = None,
        limit: int = 3,
        min_score: float = 0.6,
    ) -> list[Suggestion]:
        """Rank the registered teams most similar to `raw_name`.

        Suggestions are for humans only; they are never applied automatically.
        """
        key = normalize_name(raw_name)
        best: dict[str, float] = {}
        for candidate, team_ids in self._index.items():
            score = difflib.SequenceMatcher(None, key, candidate).ratio()
            if score < min_score:
                continue
            for team_id in team_ids:
                if country is not None and self._teams[team_id].country != country:
                    continue
                best[team_id] = max(best.get(team_id, 0.0), score)
        ranked = sorted(best.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return [
            Suggestion(tid, self._teams[tid].name, round(score, 3))
            for tid, score in ranked
        ]

    def _reject_ambiguous_spellings(self) -> None:
        problems = []
        for key, team_ids in self._index.items():
            countries = [self._teams[team_id].country for team_id in team_ids]
            if len(countries) != len(set(countries)):
                problems.append(f"'{key}' is claimed by {', '.join(team_ids)}")
        if problems:
            raise ValueError("Ambiguous spellings:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------------------- #
# Auditing the downloaded data
# --------------------------------------------------------------------------- #
def scan_football_data(raw_dir: Path) -> pd.DataFrame:
    """Count every team spelling in the downloaded football-data files.

    Returns:
        DataFrame with columns `league`, `file`, `raw_name`, `matches`.
    """
    rows: list[dict[str, object]] = []
    for path in sorted(raw_dir.glob("*.csv")):
        found = _FILE_PATTERN.match(path.name)
        if found is None:
            logger.warning(
                "Skipping %s: expected a <LEAGUE>_<SEASON>.csv name", path.name
            )
            continue
        frame = pd.read_csv(
            path,
            usecols=["HomeTeam", "AwayTeam"],
            encoding="utf-8-sig",
            encoding_errors="replace",
        ).dropna()
        names = (
            pd.concat([frame["HomeTeam"], frame["AwayTeam"]]).astype(str).str.strip()
        )
        for raw_name, count in names.value_counts().items():
            rows.append(
                {
                    "league": found["league"],
                    "file": path.name,
                    "raw_name": raw_name,
                    "matches": int(count),
                }
            )
    return pd.DataFrame(rows, columns=["league", "file", "raw_name", "matches"])


@dataclass(frozen=True)
class AuditResult:
    """Outcome of checking every scanned spelling against the registry.

    Attributes:
        n_names: Distinct `(league, spelling)` pairs scanned.
        n_resolved: How many of them resolved to a team.
        unresolved: Review queue with `league`, `raw_name`, `matches`,
            `files` and `suggestions` columns.
        collisions: Files where two different spellings resolved to one team.
    """

    n_names: int
    n_resolved: int
    unresolved: pd.DataFrame
    collisions: list[str]

    @property
    def ok(self) -> bool:
        """True when everything resolved and no collisions were found."""
        return self.unresolved.empty and not self.collisions


def audit_names(registry: TeamRegistry, scan: pd.DataFrame) -> AuditResult:
    """Resolve every scanned spelling and report what needs a human."""
    pairs = scan[["league", "raw_name"]].drop_duplicates()
    countries = {code: league.country for code, league in LEAGUES.items()}
    resolved: dict[tuple[str, str], str | None] = {
        (league, raw): registry.resolve(raw, country=countries.get(league))
        for league, raw in pairs.itertuples(index=False)
    }
    team_ids = [
        resolved[(lg, raw)] for lg, raw in zip(scan["league"], scan["raw_name"])
    ]
    scan = scan.assign(team_id=team_ids)

    collisions: list[str] = []
    for file, group in scan.dropna(subset=["team_id"]).groupby("file"):
        spellings_per_team = group.groupby("team_id")["raw_name"].nunique()
        for team_id in spellings_per_team[spellings_per_team > 1].index:
            names = sorted(group.loc[group["team_id"] == team_id, "raw_name"])
            collisions.append(f"{file}: {', '.join(names)} all resolve to '{team_id}'")

    missing = scan[scan["team_id"].isna()]
    rows = []
    for (league, raw), group in missing.groupby(["league", "raw_name"]):
        close = registry.suggest(raw, country=countries.get(league))
        rows.append(
            {
                "league": league,
                "raw_name": raw,
                "matches": int(group["matches"].sum()),
                "files": int(group["file"].nunique()),
                "suggestions": "; ".join(f"{s.name} ({s.score:.2f})" for s in close),
            }
        )
    unresolved = pd.DataFrame(
        rows, columns=["league", "raw_name", "matches", "files", "suggestions"]
    )
    n_resolved = sum(team_id is not None for team_id in resolved.values())
    return AuditResult(len(resolved), n_resolved, unresolved, collisions)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check every team name in data/raw against the team registry."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--raw-dir", type=Path, default=FOOTBALL_DATA_DIR)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the audit and return the process exit code.

    Returns `0` when every name resolves cleanly, `1` when names need
    review (or collide), and `2` when the registry file is missing.
    """
    args = _parse_args(argv)
    log = setup_logging("entities")
    if not args.registry.exists():
        log.error("Registry not found: %s", args.registry)
        return 2
    registry = TeamRegistry.from_yaml(args.registry)
    log.info("Registry: %d teams loaded from %s", len(registry), args.registry)

    scan = scan_football_data(args.raw_dir)
    if scan.empty:
        log.error(
            "No results files in %s; run python -m src.ingest first", args.raw_dir
        )
        return 1
    result = audit_names(registry, scan)
    log.info(
        "Audit: %d distinct spellings, %d resolved, %d need review",
        result.n_names,
        result.n_resolved,
        len(result.unresolved),
    )
    for message in result.collisions:
        log.error("Collision: %s", message)
    for row in result.unresolved.itertuples(index=False):
        log.error(
            "Unresolved %s | %r | %d matches | closest: %s",
            row.league,
            row.raw_name,
            row.matches,
            row.suggestions or "none",
        )
    if not result.unresolved.empty:
        args.queue.parent.mkdir(parents=True, exist_ok=True)
        result.unresolved.to_csv(args.queue, index=False)
        log.error("Review queue written to %s", args.queue)
    if result.ok:
        log.info("Every team name resolves cleanly")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
