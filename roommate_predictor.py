from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


IRRELEVANT_COLUMNS = {
    "Timestamp",
    "Email address",
    "Outlook email ID",
    "Hostel Block",
    "Floor",
    "Room number",
    "Current Roommate Name",
    "Please provide an honest feedback about your current roommate",
    "What is their usual sleep time? ",
    "Do they prefer lights on while sleeping?",
    "How clean do they keep the room?",
    "How noisy are they?(1-5)",
    "How often do they invite friends to the room?",
    "How warm/cold do they like the room?",
}

FEATURE_ALIASES = {
    "sleep_time": [
        "What is your usual sleep time? ",
        "sleep time",
        "sleep schedule",
    ],
    "lights_preference": [
        "Do you prefer lights on while sleeping?",
        "lights preference",
    ],
    "cleanliness_level": [
        "How clean do you keep your room?",
        "cleanliness level",
        "cleanliness",
    ],
    "noise_tolerance": [
        "How much noise can you tolerate while studying?",
        "noise tolerance",
    ],
    "social_activity": [
        "How often do you invite friends to your room?",
        "social activity",
        "social behavior",
    ],
    "sharing_comfort": [
        "How comfortable are you in sharing your belongings",
        "sharing comfort",
    ],
    "temperature_preference": [
        "How warm/cold do you like your room?",
        "temperature preference",
    ],
    "study_hours": [
        "How many hours do you study daily?",
        "study hours",
        "study habits",
    ],
}

DISPLAY_NAMES = {
    "sleep_time": "sleep schedule",
    "lights_preference": "lights preference",
    "cleanliness_level": "cleanliness",
    "noise_tolerance": "noise tolerance",
    "social_activity": "social activity",
    "sharing_comfort": "sharing comfort",
    "temperature_preference": "room temperature preference",
    "study_hours": "study habits",
}

FEATURE_WEIGHTS = {
    "sleep_time": 8.0,
    "cleanliness_level": 7.0,
    "noise_tolerance": 6.0,
    "social_activity": 5.0,
    "lights_preference": 4.0,
    "temperature_preference": 3.0,
    "sharing_comfort": 2.0,
    "study_hours": 1.0,
}


@dataclass
class MatchResult:
    full_name: str
    cluster: int
    compatibility_score: float
    explanation: str
    current_roommate_name: str
    roommate_feedback: str


class RoommateCompatibilityPredictor:
    def __init__(self, random_state: int = 42, max_iterations: int = 100) -> None:
        self.random_state = random_state
        self.max_iterations = max_iterations
        self.raw_df: pd.DataFrame | None = None
        self.model_df: pd.DataFrame | None = None
        self.feature_columns: list[str] = []
        self.scaled_matrix: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None
        self.feature_weights: np.ndarray | None = None
        self.centroids: np.ndarray | None = None
        self.cluster_labels: np.ndarray | None = None

    def fit(self, csv_path: str | Path) -> pd.DataFrame:
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        self.raw_df = df.copy()

        cleaned = self._prepare_dataframe(df)
        self.model_df = cleaned
        self.feature_columns = [
            column
            for column in cleaned.columns
            if column not in {"Full name", "Sex", "Current Roommate Name", "Roommate Feedback"}
        ]

        feature_matrix = cleaned[self.feature_columns].to_numpy(dtype=float)
        self.means = feature_matrix.mean(axis=0)
        self.stds = feature_matrix.std(axis=0)
        self.stds[self.stds == 0] = 1.0
        base_scaled_matrix = (feature_matrix - self.means) / self.stds
        self.feature_weights = np.array(
            [FEATURE_WEIGHTS.get(feature, 1.0) for feature in self.feature_columns],
            dtype=float,
        )
        self.scaled_matrix = base_scaled_matrix * self.feature_weights

        cluster_count = self._select_cluster_count(self.scaled_matrix)
        self.cluster_labels, self.centroids = self._kmeans(self.scaled_matrix, cluster_count)
        self.model_df["cluster"] = self.cluster_labels
        return self.model_df.copy()

    def available_users(self) -> list[str]:
        if self.model_df is None:
            return []
        return sorted(self.model_df["Full name"].dropna().astype(str).unique().tolist())

    def recommend_for_user(self, user_name: str, top_n: int | None = 3) -> list[MatchResult]:
        if self.model_df is None or self.scaled_matrix is None or self.cluster_labels is None:
            raise ValueError("Model has not been fitted yet.")

        matches = self.model_df.index[self.model_df["Full name"] == user_name].tolist()
        if not matches:
            raise ValueError(f"User '{user_name}' was not found in the dataset.")

        user_index = matches[0]
        user_vector = self.scaled_matrix[user_index]
        return self._recommend_from_vector(
            user_vector=user_vector,
            raw_profile=self.model_df.loc[user_index, self.feature_columns].to_dict(),
            requested_sex=float(self.model_df.loc[user_index, "Sex"]),
            exclude_index=user_index,
            exclude_name=None,
            top_n=top_n,
        )

    def recommend_for_profile(self, profile: dict[str, Any], top_n: int | None = 3) -> list[MatchResult]:
        if self.means is None or self.stds is None:
            raise ValueError("Model has not been fitted yet.")

        raw_profile = {}
        entered_name = str(profile.get("Full name", "")).strip().lower()
        requested_sex = self._encode_sex(profile.get("Sex"))
        for feature in self.feature_columns:
            if feature not in profile:
                raise ValueError(f"Missing value for '{feature}'.")
            raw_profile[feature] = float(profile[feature])

        ordered = np.array([raw_profile[feature] for feature in self.feature_columns], dtype=float)
        scaled = ((ordered - self.means) / self.stds) * self.feature_weights
        return self._recommend_from_vector(
            user_vector=scaled,
            raw_profile=raw_profile,
            requested_sex=requested_sex,
            exclude_index=None,
            exclude_name=entered_name or None,
            top_n=top_n,
        )

    def top_outside_cluster_for_user(self, user_name: str, top_n: int = 3) -> list[MatchResult]:
        if self.model_df is None or self.scaled_matrix is None or self.cluster_labels is None:
            raise ValueError("Model has not been fitted yet.")

        matches = self.model_df.index[self.model_df["Full name"] == user_name].tolist()
        if not matches:
            raise ValueError(f"User '{user_name}' was not found in the dataset.")

        user_index = matches[0]
        user_vector = self.scaled_matrix[user_index]
        return self._outside_cluster_matches(
            user_vector=user_vector,
            raw_profile=self.model_df.loc[user_index, self.feature_columns].to_dict(),
            requested_sex=float(self.model_df.loc[user_index, "Sex"]),
            exclude_index=user_index,
            exclude_name=None,
            top_n=top_n,
        )

    def top_outside_cluster_for_profile(self, profile: dict[str, Any], top_n: int = 3) -> list[MatchResult]:
        if self.means is None or self.stds is None:
            raise ValueError("Model has not been fitted yet.")

        raw_profile = {}
        entered_name = str(profile.get("Full name", "")).strip().lower()
        requested_sex = self._encode_sex(profile.get("Sex"))
        for feature in self.feature_columns:
            if feature not in profile:
                raise ValueError(f"Missing value for '{feature}'.")
            raw_profile[feature] = float(profile[feature])

        ordered = np.array([raw_profile[feature] for feature in self.feature_columns], dtype=float)
        scaled = (ordered - self.means) / self.stds
        return self._outside_cluster_matches(
            user_vector=scaled,
            raw_profile=raw_profile,
            requested_sex=requested_sex,
            exclude_index=None,
            exclude_name=entered_name or None,
            top_n=top_n,
        )

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        working = df.copy()
        metadata = working.reindex(
            columns=[
                "Current Roommate Name",
                "Please provide an honest feedback about your current roommate",
            ],
            fill_value="",
        ).copy()
        drop_columns = [column for column in working.columns if column in IRRELEVANT_COLUMNS]
        working = working.drop(columns=drop_columns, errors="ignore")

        if "Full name" not in working.columns:
            raise ValueError("The dataset must include a 'Full name' column.")

        normalized = pd.DataFrame()
        normalized["Full name"] = working["Full name"].fillna("Unknown User").astype(str).str.strip()
        normalized["Current Roommate Name"] = (
            metadata["Current Roommate Name"].fillna("None").astype(str).str.strip()
        )
        normalized["Roommate Feedback"] = (
            metadata["Please provide an honest feedback about your current roommate"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        if "Sex" in working.columns:
            normalized["Sex"] = (
                working["Sex"]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"male": 0, "female": 1})
                .fillna(2)
                .astype(float)
            )
        else:
            normalized["Sex"] = 2.0

        for feature_name, aliases in FEATURE_ALIASES.items():
            source_column = next((alias for alias in aliases if alias in working.columns), None)
            if source_column is None:
                continue

            numeric = pd.to_numeric(working[source_column], errors="coerce")
            normalized[feature_name] = numeric.fillna(numeric.median())

        available_lifestyle_features = [
            column
            for column in normalized.columns
            if column not in {"Full name", "Sex", "Current Roommate Name", "Roommate Feedback"}
        ]
        if len(available_lifestyle_features) < 3:
            raise ValueError(
                "Not enough meaningful lifestyle features were found for clustering."
            )

        return normalized.dropna(subset=["Full name"]).reset_index(drop=True)

    def _select_cluster_count(self, matrix: np.ndarray) -> int:
        row_count = len(matrix)
        if row_count <= 3:
            return 1

        max_clusters = min(6, row_count - 1)
        best_k = 2
        best_score = -1.0

        for candidate in range(2, max_clusters + 1):
            labels, centroids = self._kmeans(matrix, candidate)
            score = self._silhouette_score(matrix, labels, centroids)
            if score > best_score:
                best_score = score
                best_k = candidate

        return best_k

    def _kmeans(self, matrix: np.ndarray, cluster_count: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        chosen = rng.choice(len(matrix), size=cluster_count, replace=False)
        centroids = matrix[chosen].copy()

        for _ in range(self.max_iterations):
            distances = np.linalg.norm(matrix[:, None, :] - centroids[None, :, :], axis=2)
            labels = distances.argmin(axis=1)

            updated_centroids = centroids.copy()
            for cluster_id in range(cluster_count):
                members = matrix[labels == cluster_id]
                if len(members) == 0:
                    updated_centroids[cluster_id] = matrix[rng.integers(0, len(matrix))]
                else:
                    updated_centroids[cluster_id] = members.mean(axis=0)

            if np.allclose(updated_centroids, centroids):
                centroids = updated_centroids
                break
            centroids = updated_centroids

        distances = np.linalg.norm(matrix[:, None, :] - centroids[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        return labels, centroids

    def _silhouette_score(
        self,
        matrix: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
    ) -> float:
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            return -1.0

        scores = []
        for index, point in enumerate(matrix):
            same_cluster = matrix[labels == labels[index]]
            if len(same_cluster) <= 1:
                continue

            same_cluster_without_self = np.delete(
                same_cluster,
                np.where(np.all(same_cluster == point, axis=1))[0][:1],
                axis=0,
            )
            if len(same_cluster_without_self) == 0:
                continue

            a = np.mean(np.linalg.norm(same_cluster_without_self - point, axis=1))
            b = np.inf
            for other_label in unique_labels:
                if other_label == labels[index]:
                    continue
                other_cluster = matrix[labels == other_label]
                if len(other_cluster) == 0:
                    continue
                distance = np.mean(np.linalg.norm(other_cluster - point, axis=1))
                b = min(b, distance)

            if b == np.inf:
                continue
            scores.append((b - a) / max(a, b))

        if not scores:
            return -1.0
        return float(np.mean(scores))

    def _recommend_from_vector(
        self,
        user_vector: np.ndarray,
        raw_profile: dict[str, float],
        requested_sex: float | None,
        exclude_index: int | None,
        exclude_name: str | None,
        top_n: int | None,
    ) -> list[MatchResult]:
        if self.model_df is None or self.scaled_matrix is None or self.centroids is None:
            raise ValueError("Model has not been fitted yet.")

        cluster_id = int(
            np.linalg.norm(self.centroids - user_vector, axis=1).argmin()
        )
        cluster_members = self.model_df.index[self.model_df["cluster"] == cluster_id].tolist()

        results: list[MatchResult] = []
        for candidate_index in cluster_members:
            if exclude_index is not None and candidate_index == exclude_index:
                continue
            if exclude_name is not None:
                candidate_name = str(self.model_df.loc[candidate_index, "Full name"]).strip().lower()
                if candidate_name == exclude_name:
                    continue
            if (
                requested_sex is not None
                and "Sex" in self.model_df.columns
                and float(self.model_df.loc[candidate_index, "Sex"]) != requested_sex
            ):
                continue

            candidate_vector = self.scaled_matrix[candidate_index]
            similarity = self._cosine_similarity(user_vector, candidate_vector)
            score = round(max(0.0, similarity) * 100, 2)
            candidate_row = self.model_df.loc[candidate_index]
            explanation = self._build_explanation(
                raw_profile=raw_profile,
                candidate_profile=candidate_row[self.feature_columns].to_dict(),
            )
            results.append(
                MatchResult(
                    full_name=str(candidate_row["Full name"]),
                    cluster=cluster_id,
                    compatibility_score=score,
                    explanation=explanation,
                    current_roommate_name=str(candidate_row.get("Current Roommate Name", "None")),
                    roommate_feedback=str(candidate_row.get("Roommate Feedback", "")),
                )
            )

        results.sort(key=lambda item: item.compatibility_score, reverse=True)
        if top_n is None:
            return results
        return results[:top_n]

    def _outside_cluster_matches(
        self,
        user_vector: np.ndarray,
        raw_profile: dict[str, float],
        requested_sex: float | None,
        exclude_index: int | None,
        exclude_name: str | None,
        top_n: int,
    ) -> list[MatchResult]:
        if self.model_df is None or self.scaled_matrix is None or self.centroids is None:
            raise ValueError("Model has not been fitted yet.")

        cluster_id = int(np.linalg.norm(self.centroids - user_vector, axis=1).argmin())
        matches: list[MatchResult] = []

        for candidate_index in self.model_df.index.tolist():
            if int(self.model_df.loc[candidate_index, "cluster"]) == cluster_id:
                continue
            if exclude_index is not None and candidate_index == exclude_index:
                continue
            if exclude_name is not None:
                candidate_name = str(self.model_df.loc[candidate_index, "Full name"]).strip().lower()
                if candidate_name == exclude_name:
                    continue
            if (
                requested_sex is not None
                and "Sex" in self.model_df.columns
                and float(self.model_df.loc[candidate_index, "Sex"]) != requested_sex
            ):
                continue

            candidate_vector = self.scaled_matrix[candidate_index]
            similarity = self._cosine_similarity(user_vector, candidate_vector)
            score = round(max(0.0, similarity) * 100, 2)
            candidate_row = self.model_df.loc[candidate_index]
            explanation = self._build_explanation(
                raw_profile=raw_profile,
                candidate_profile=candidate_row[self.feature_columns].to_dict(),
            )
            matches.append(
                MatchResult(
                    full_name=str(candidate_row["Full name"]),
                    cluster=int(candidate_row["cluster"]),
                    compatibility_score=score,
                    explanation=explanation,
                    current_roommate_name=str(candidate_row.get("Current Roommate Name", "None")),
                    roommate_feedback=str(candidate_row.get("Roommate Feedback", "")),
                )
            )

        matches.sort(key=lambda item: item.compatibility_score, reverse=True)
        return matches[:top_n]

    def _build_explanation(
        self,
        raw_profile: dict[str, float],
        candidate_profile: dict[str, float],
    ) -> str:
        closeness = []
        for feature in self.feature_columns:
            difference = abs(float(raw_profile[feature]) - float(candidate_profile[feature]))
            weighted_difference = difference / FEATURE_WEIGHTS.get(feature, 1.0)
            closeness.append((weighted_difference, feature))

        closeness.sort(key=lambda item: item[0])
        best_features = [DISPLAY_NAMES.get(feature, feature) for _, feature in closeness[:3]]
        if len(best_features) == 1:
            summary = best_features[0]
        elif len(best_features) == 2:
            summary = f"{best_features[0]} and {best_features[1]}"
        else:
            summary = f"{best_features[0]}, {best_features[1]}, and {best_features[2]}"

        return f"Strong alignment in {summary}."

    @staticmethod
    def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
        first_norm = np.linalg.norm(first)
        second_norm = np.linalg.norm(second)
        if first_norm == 0 or second_norm == 0:
            return 0.0
        return float(np.dot(first, second) / (first_norm * second_norm))

    @staticmethod
    def _encode_sex(value: Any) -> float | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        mapping = {"male": 0.0, "female": 1.0, "other": 2.0, "unknown": 2.0}
        return mapping.get(normalized, 2.0)
