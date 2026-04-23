from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class UserRecord:
    user_id: int
    sex: str
    age: int
    occupation: str
    zip_code: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MovieRecord:
    movie_id: int
    title: str
    release_year: int | None
    genres: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["genres"] = "|".join(self.genres)
        return payload


@dataclass(frozen=True)
class RatingRecord:
    user_id: int
    movie_id: int
    rating: float
    timestamp: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RequestContextRow:
    request_id: int
    user_id: int
    timestamp: int
    history_length: int
    positive_history_length: int
    recent_mean_rating: float
    top_genre_1: str
    top_genre_2: str
    observed_movie_id: int
    observed_rating: float
    observed_reward: int

    def to_dict(self) -> dict:
        return asdict(self)
