#!/usr/bin/env python3
"""
Convex database integration for updating event analytics.
Handles event updates with aggregated statistics and demographics.
"""

from __future__ import annotations

from typing import Any

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit("The 'requests' package is required. Install it with: pip install requests") from exc

from app import config

# Get Convex configuration from config module
BASE_URL = config.CONVEX_BASE_URL.rstrip("/")
DEFAULT_EVENT_ID = config.CONVEX_EVENT_ID
DEFAULT_USER_ID = config.CONVEX_USER_ID

JsonDict = dict[str, Any]


def _calculate_median(ages: list[int]) -> float:
    """
    Calculate the median of a list of ages.

    Args:
        ages: List of integer ages

    Returns:
        Median value as float, or 0.0 if list is empty

    """
    if not ages:
        return 0.0

    sorted_ages = sorted(ages)
    n = len(sorted_ages)

    if n % 2 == 0:
        # Even number of elements: average of two middle elements
        mid1 = sorted_ages[n // 2 - 1]
        mid2 = sorted_ages[n // 2]
        return float((mid1 + mid2) / 2)
    else:
        # Odd number of elements: middle element
        return float(sorted_ages[n // 2])


def post_json(path: str, payload: JsonDict) -> JsonDict:
    """
    Post JSON payload to Convex endpoint.

    Args:
        path: API endpoint path (e.g., "/events/update-analysis")
        payload: JSON payload to send

    Returns:
        JSON response from the server

    Raises:
        RuntimeError: If the request fails

    """
    url = f"{BASE_URL}{path}"
    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as error:
        message = response.text.strip()
        raise RuntimeError(f"{path} failed with {response.status_code}: {message or 'no response body'}") from error
    except requests.RequestException as error:
        raise RuntimeError(f"Request to {path} failed: {error}") from error


def build_age_distribution_payload(faces_data: list) -> dict:
    """
    Build age distribution payload in Convex format.

    Args:
        faces_data: List of face data dictionaries

    Returns:
        Dictionary with age distribution in format: {a0_10: count, a11_20: count, ...}

    """
    age_distribution = {
        "a0_10": 0,
        "a11_20": 0,
        "a21_30": 0,
        "a31_40": 0,
        "a41_50": 0,
        "a51_60": 0,
        "a61_70": 0,
        "a71_80": 0,
        "a81_90": 0,
        "a91_100": 0,
        "a101": 0,
    }

    for face in faces_data:
        age_str = face.get("age", "")
        if age_str and age_str.strip():
            try:
                age = int(age_str)
                if 0 <= age <= 120:
                    if age < 10:
                        age_distribution["a0_10"] += 1
                    elif age < 20:
                        age_distribution["a11_20"] += 1
                    elif age < 30:
                        age_distribution["a21_30"] += 1
                    elif age < 40:
                        age_distribution["a31_40"] += 1
                    elif age < 50:
                        age_distribution["a41_50"] += 1
                    elif age < 60:
                        age_distribution["a51_60"] += 1
                    elif age < 70:
                        age_distribution["a61_70"] += 1
                    elif age < 80:
                        age_distribution["a71_80"] += 1
                    elif age < 90:
                        age_distribution["a81_90"] += 1
                    elif age < 100:
                        age_distribution["a91_100"] += 1
                    else:
                        age_distribution["a101"] += 1
            except (ValueError, TypeError):
                pass

    return age_distribution


def build_gender_distribution_payload(faces_data: list) -> dict:
    """
    Build gender distribution by age group payload in Convex format.

    Args:
        faces_data: List of face data dictionaries

    Returns:
        Dictionary with gender distribution by age in format:
        {g0_10: {male: count, female: count, unknown: count}, ...}

    """
    gender_distribution = {
        "g0_10": {"male": 0, "female": 0, "unknown": 0},
        "g11_20": {"male": 0, "female": 0, "unknown": 0},
        "g21_30": {"male": 0, "female": 0, "unknown": 0},
        "g31_40": {"male": 0, "female": 0, "unknown": 0},
        "g41_50": {"male": 0, "female": 0, "unknown": 0},
        "g51_60": {"male": 0, "female": 0, "unknown": 0},
        "g61_70": {"male": 0, "female": 0, "unknown": 0},
        "g71_80": {"male": 0, "female": 0, "unknown": 0},
        "g81_90": {"male": 0, "female": 0, "unknown": 0},
        "g91_100": {"male": 0, "female": 0, "unknown": 0},
        "g101": {"male": 0, "female": 0, "unknown": 0},
    }

    for face in faces_data:
        age_str = face.get("age", "")
        gender = face.get("gender", "").strip().upper()

        if age_str and age_str.strip():
            try:
                age = int(age_str)
                if 0 <= age <= 120:
                    # Determine age bracket
                    if age < 10:
                        bracket = "g0_10"
                    elif age < 20:
                        bracket = "g11_20"
                    elif age < 30:
                        bracket = "g21_30"
                    elif age < 40:
                        bracket = "g31_40"
                    elif age < 50:
                        bracket = "g41_50"
                    elif age < 60:
                        bracket = "g51_60"
                    elif age < 70:
                        bracket = "g61_70"
                    elif age < 80:
                        bracket = "g71_80"
                    elif age < 90:
                        bracket = "g81_90"
                    elif age < 100:
                        bracket = "g91_100"
                    else:
                        bracket = "g101"

                    # Increment appropriate gender count
                    if gender == "M":
                        gender_distribution[bracket]["male"] += 1
                    elif gender == "F":
                        gender_distribution[bracket]["female"] += 1
                    else:
                        gender_distribution[bracket]["unknown"] += 1
            except (ValueError, TypeError):
                pass

    return gender_distribution


def build_event_update_payload(
    faces_data: list, total_faces: int, event_id: str, user_id: str, event_price: float = 0.0
) -> JsonDict:
    """
    Build complete event update payload for Convex /events/update-analysis endpoint.

    Args:
        faces_data: List of face data dictionaries
        total_faces: Total number of unique faces
        event_id: Event ID to update
        user_id: User ID to update totals for
        event_price: Price associated with the event

    Returns:
        Complete event update payload dictionary

    """
    # Calculate age statistics
    ages = []
    for face in faces_data:
        age_str = face.get("age", "")
        if age_str and age_str.strip():
            try:
                age = int(age_str)
                if 0 <= age <= 120:
                    ages.append(age)
            except (ValueError, TypeError):
                pass

    # Calculate gender counts
    gender_counts = {"male": 0, "female": 0, "unknown": 0}
    for face in faces_data:
        gender = face.get("gender", "").strip().upper()
        if gender == "M":
            gender_counts["male"] += 1
        elif gender == "F":
            gender_counts["female"] += 1
        else:
            gender_counts["unknown"] += 1

    # Calculate data quality
    faces_with_age = sum(1 for face in faces_data if face.get("age", "").strip())
    faces_with_gender = sum(1 for face in faces_data if face.get("gender", "").strip())
    faces_with_camera = sum(1 for face in faces_data if face.get("cameras", "").strip())

    quality_score = 0.0
    if total_faces > 0:
        quality_score = (
            (faces_with_age / total_faces * 0.3333)
            + (faces_with_gender / total_faces * 0.3333)
            + (faces_with_camera / total_faces * 0.3334)
        )

    # Build the event update payload
    event_payload = {
        "eventId": event_id,
        "userId": user_id,
        "price": event_price,
        "data_quality": round(quality_score, 2),
        "status": "completed",
        "favourite": False,  # Default value, can be made configurable if needed
        "age": {
            "min": min(ages) if ages else 0,
            "max": max(ages) if ages else 0,
            "average": round(sum(ages) / len(ages), 1) if ages else 0.0,
            "median": _calculate_median(ages),
        },
        "age_distribution": build_age_distribution_payload(faces_data),
        "gender": gender_counts,
        "gender_distribution": build_gender_distribution_payload(faces_data),
    }

    return event_payload


def update_event(
    faces_data: list,
    total_faces: int,
    event_price: float = 0.0,
    event_id: str | None = None,
    user_id: str | None = None,
) -> str:
    """
    Update an existing event in Convex database with analysis results.

    Args:
        faces_data: List of face data dictionaries
        total_faces: Total number of unique faces
        event_price: Price associated with the event
        event_id: Event ID to update (uses CONVEX_EVENT_ID env var if not provided)
        user_id: User ID to update totals for (uses CONVEX_USER_ID env var if not provided)

    Returns:
        Response from Convex

    Raises:
        RuntimeError: If event update fails
        ValueError: If event_id or user_id is not provided and corresponding env vars are not set

    """
    # Use provided event_id or fall back to environment variable
    if event_id is None:
        event_id = DEFAULT_EVENT_ID

    if not event_id:
        raise ValueError("event_id must be provided or CONVEX_EVENT_ID environment variable must be set")

    # Use provided user_id or fall back to environment variable
    if user_id is None:
        user_id = DEFAULT_USER_ID

    if not user_id:
        raise ValueError("user_id must be provided or CONVEX_USER_ID environment variable must be set")

    print(f"\n📤 Updating event in Convex: '{event_id}' for user: '{user_id}'")

    event_payload = build_event_update_payload(faces_data, total_faces, event_id, user_id, event_price)
    response = post_json("/events/update-analysis", event_payload)

    print(f"✅ Event updated successfully: {event_id}")
    return response


def ingest_event_data(
    faces_data: list,
    total_faces: int,
    event_price: float = 0.0,
    event_id: str | None = None,
    user_id: str | None = None,
) -> str:
    """
    Update event with analysis data.

    Args:
        faces_data: List of face data dictionaries
        total_faces: Total number of unique faces
        event_price: Price associated with the event
        event_id: Event ID to update (uses CONVEX_EVENT_ID env var if not provided)
        user_id: User ID to update totals for (uses CONVEX_USER_ID env var if not provided)

    Returns:
        Response from Convex

    Raises:
        RuntimeError: If event update fails
        ValueError: If event_id or user_id is not provided and corresponding env vars are not set

    """
    print(f"\n{'=' * 80}")
    print("📊 UPDATING EVENT IN CONVEX DATABASE")
    print(f"{'=' * 80}")
    print(f"Convex URL: {BASE_URL}")
    print(f"Event ID: {event_id or DEFAULT_EVENT_ID}")
    print(f"User ID: {user_id or DEFAULT_USER_ID}")

    # Update event with analysis results
    response = update_event(faces_data, total_faces, event_price, event_id, user_id)

    print(f"\n{'=' * 80}")
    print("✅ CONVEX EVENT UPDATE COMPLETE")
    print(f"{'=' * 80}\n")

    return response
