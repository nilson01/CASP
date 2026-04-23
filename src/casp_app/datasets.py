from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from .config import ApplicationConfig, default_raw_root
from .schema import MovieRecord, RatingRecord, UserRecord


def raw_archive_path(config: ApplicationConfig) -> Path:
    return default_raw_root() / config.raw_archive_name


def raw_data_dir(config: ApplicationConfig) -> Path:
    return default_raw_root() / config.raw_data_subdir


def extraction_manifest_path(config: ApplicationConfig) -> Path:
    return raw_data_dir(config) / "extraction_manifest.json"


def expected_raw_files(config: ApplicationConfig) -> dict[str, Path]:
    root = raw_data_dir(config)
    return {
        "users": root / "users.dat",
        "movies": root / "movies.dat",
        "ratings": root / "ratings.dat",
    }


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_file_status(config: ApplicationConfig) -> list[dict]:
    rows = [
        {
            "logical_name": "archive",
            "path": str(raw_archive_path(config)),
            "exists": raw_archive_path(config).exists(),
        }
    ]
    for logical_name, path in expected_raw_files(config).items():
        rows.append(
            {
                "logical_name": logical_name,
                "path": str(path),
                "exists": path.exists(),
            }
        )
    return rows


def raw_data_available(config: ApplicationConfig) -> bool:
    return all(path.exists() for path in expected_raw_files(config).values())


def dataset_preflight_message(config: ApplicationConfig, mode: str) -> str:
    archive = raw_archive_path(config)
    root = raw_data_dir(config)
    return (
        f"Cannot run application mode '{mode}' because the MovieLens 1M raw files are missing. "
        f"Place {archive} and extract it so that {root / 'users.dat'}, {root / 'movies.dat'}, "
        f"and {root / 'ratings.dat'} exist, or run the download/extract helpers later."
    )


def require_raw_files(config: ApplicationConfig, mode: str) -> None:
    if not raw_data_available(config):
        raise FileNotFoundError(dataset_preflight_message(config, mode))


def download_raw_archive(config: ApplicationConfig, force: bool = False) -> Path:
    archive = raw_archive_path(config)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() and not force:
        return archive
    urllib.request.urlretrieve(config.raw_download_url, archive)
    return archive


def extract_raw_archive(config: ApplicationConfig, force: bool = False) -> list[dict]:
    archive = raw_archive_path(config)
    if not archive.exists():
        raise FileNotFoundError(
            f"Cannot extract MovieLens 1M because {archive} does not exist."
        )
    target_dir = raw_data_dir(config)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zip_handle:
        for member in zip_handle.infolist():
            if not member.filename.startswith(f"{config.raw_data_subdir}/"):
                continue
            relative = Path(member.filename).relative_to(config.raw_data_subdir)
            destination = target_dir / relative
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not force:
                continue
            with zip_handle.open(member) as source, destination.open("wb") as sink:
                shutil.copyfileobj(source, sink)
    return extracted_file_manifest_rows(config)


def extracted_file_manifest_rows(config: ApplicationConfig) -> list[dict]:
    rows = []
    for logical_name, path in expected_raw_files(config).items():
        rows.append(
            {
                "logical_name": logical_name,
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return rows


def dataset_manifest_rows(config: ApplicationConfig) -> list[dict]:
    raw_status = raw_file_status(config)
    rows = [
        {"field": "dataset_id", "value": config.dataset_id},
        {"field": "raw_download_url", "value": config.raw_download_url},
        {"field": "raw_archive_path", "value": str(raw_archive_path(config))},
        {"field": "raw_data_dir", "value": str(raw_data_dir(config))},
        {"field": "raw_data_available", "value": raw_data_available(config)},
        {"field": "min_history_length", "value": config.min_history_length},
        {"field": "positive_rating_threshold", "value": config.positive_rating_threshold},
        {"field": "candidate_set_size", "value": config.candidate_set_size},
        {"field": "context_limit", "value": config.context_limit},
        {"field": "min_supporting_generators", "value": config.min_supporting_generators},
        {"field": "fallback_min_supporting_generators", "value": config.fallback_min_supporting_generators},
        {"field": "fallback_context_floor", "value": config.fallback_context_floor},
        {"field": "fallback_pool_strategy", "value": config.fallback_pool_strategy},
        {"field": "fallback_head_generator_index", "value": config.fallback_head_generator_index},
        {"field": "fallback_singleton_caps", "value": "|".join(str(value) for value in config.fallback_singleton_caps)},
        {"field": "collaborative_neighbor_strategy", "value": config.collaborative_neighbor_strategy},
    ]
    archive = raw_archive_path(config)
    if archive.exists():
        rows.append({"field": "raw_archive_sha256", "value": compute_sha256(archive)})
    for row in raw_status:
        rows.append({"field": f"raw_file::{row['logical_name']}", "value": row["path"]})
        rows.append({"field": f"raw_file_exists::{row['logical_name']}", "value": row["exists"]})
    if raw_data_available(config):
        users = load_users(config)
        movies = load_movies(config)
        ratings = load_ratings(config)
        genre_counter = Counter(
            genre
            for movie in movies
            for genre in movie.genres
        )
        rows.extend(
            [
                {"field": "num_users", "value": len(users)},
                {"field": "num_movies", "value": len(movies)},
                {"field": "num_ratings", "value": len(ratings)},
                {"field": "num_genres", "value": len(genre_counter)},
            ]
        )
    return rows


def provenance_rows(config: ApplicationConfig) -> list[dict]:
    archive = raw_archive_path(config)
    rows = [
        {"field": "source_url", "value": config.raw_download_url},
        {"field": "archive_path", "value": str(archive)},
        {"field": "archive_exists", "value": archive.exists()},
        {"field": "archive_sha256", "value": compute_sha256(archive) if archive.exists() else ""},
        {"field": "extraction_manifest_path", "value": str(extraction_manifest_path(config))},
    ]
    for row in extracted_file_manifest_rows(config):
        rows.append({"field": f"extracted::{row['logical_name']}::path", "value": row["path"]})
        rows.append({"field": f"extracted::{row['logical_name']}::exists", "value": row["exists"]})
        rows.append({"field": f"extracted::{row['logical_name']}::bytes", "value": row["bytes"]})
    return rows


def _parse_year(title: str) -> int | None:
    if len(title) >= 6 and title.endswith(")") and title[-5] == "(":
        year_token = title[-4:]
        if year_token.isdigit():
            return int(year_token)
    return None


def load_users(config: ApplicationConfig) -> list[UserRecord]:
    path = expected_raw_files(config)["users"]
    rows = []
    with path.open(encoding="latin-1") as handle:
        for line in handle:
            user_id, sex, age, occupation, zip_code = line.rstrip("\n").split("::")
            rows.append(
                UserRecord(
                    user_id=int(user_id),
                    sex=sex,
                    age=int(age),
                    occupation=occupation,
                    zip_code=zip_code,
                )
            )
    return rows


def load_movies(config: ApplicationConfig) -> list[MovieRecord]:
    path = expected_raw_files(config)["movies"]
    rows = []
    with path.open(encoding="latin-1") as handle:
        for line in handle:
            movie_id, title, genres = line.rstrip("\n").split("::")
            genre_tuple = tuple(token for token in genres.split("|") if token)
            rows.append(
                MovieRecord(
                    movie_id=int(movie_id),
                    title=title,
                    release_year=_parse_year(title),
                    genres=genre_tuple,
                )
            )
    return rows


def load_ratings(config: ApplicationConfig) -> list[RatingRecord]:
    path = expected_raw_files(config)["ratings"]
    rows = []
    with path.open(encoding="latin-1") as handle:
        for line in handle:
            user_id, movie_id, rating, timestamp = line.rstrip("\n").split("::")
            rows.append(
                RatingRecord(
                    user_id=int(user_id),
                    movie_id=int(movie_id),
                    rating=float(rating),
                    timestamp=int(timestamp),
                )
            )
    return rows
