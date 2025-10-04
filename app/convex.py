#!/usr/bin/env python3
"""
Convex database integration for storing event analytics and user data.
Handles event creation and user updates with aggregated statistics.
"""

from __future__ import annotations

import time
from typing import Any

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit("The 'requests' package is required. Install it with: pip install requests") from exc

from app import config

# Get Convex configuration from config module
BASE_URL = config.CONVEX_BASE_URL.rstrip("/")
DEFAULT_USER_ID = config.CONVEX_USER_ID

JsonDict = dict[str, Any]


def post_json(path: str, payload: JsonDict) -> JsonDict:
    """
    Post JSON payload to Convex endpoint.

    Args:
        path: API endpoint path (e.g., "/ingest/events")
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


def build_event_payload(
    faces_data: list, total_faces: int, _event_name: str = "Face Detection Event", event_price: float = 0.0
) -> JsonDict:
    """
    Build complete event payload for Convex /ingest/events endpoint.

    Args:
        faces_data: List of face data dictionaries
        total_faces: Total number of unique faces
        event_name: Name of the event
        event_price: Price associated with the event

    Returns:
        Complete event payload dictionary

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

    # Build the event payload
    event_payload = {
        "name": str(time.time()),
        "price": event_price,
        "age": {
            "average": round(sum(ages) / len(ages), 1) if ages else 0.0,
            "median": float(sorted(ages)[len(ages) // 2]) if ages else 0.0,
            "min": min(ages) if ages else 0,
            "max": max(ages) if ages else 0,
        },
        "age_distribution": build_age_distribution_payload(faces_data),
        "gender": gender_counts,
        "gender_distribution": build_gender_distribution_payload(faces_data),
        "data_quality": round(quality_score, 2),
    }

    return event_payload


def create_event(
    faces_data: list, total_faces: int, event_name: str = "Face Detection Event", event_price: float = 0.0
) -> str:
    """
    Create a new event in Convex database.

    Args:
        faces_data: List of face data dictionaries
        total_faces: Total number of unique faces
        event_name: Name of the event
        event_price: Price associated with the event

    Returns:
        Event ID from Convex

    Raises:
        RuntimeError: If event creation fails

    """
    print(f"\n📤 Creating event in Convex: '{event_name}'")

    event_payload = build_event_payload(faces_data, total_faces, event_name, event_price)
    response = post_json("/ingest/events", event_payload)

    event_id = response.get("id")
    if not event_id:
        raise RuntimeError("Event creation succeeded but no ID was returned")

    print(f"✅ Event created successfully with ID: {event_id}")
    return event_id


def update_user_with_event(user_id: str, event_id: str, faces_data: list, _total_faces: int) -> str:
    """
    Update user with new event data, adding to their totals.

    Note: This function sends the current event's data as the new totals.
    In a real implementation, you would need to fetch existing user data,
    add the new event data to it, and then send the combined totals.

    Args:
        user_id: User ID to update
        event_id: Event ID to associate with user
        faces_data: List of face data dictionaries
        total_faces: Total number of unique faces

    Returns:
        User mutation ID from Convex

    Raises:
        RuntimeError: If user update fails

    """
    print(f"\n📤 Updating user {user_id} with event {event_id}")

    # Build user update payload
    # Note: In production, you'd fetch existing totals and add to them
    user_payload = {
        "userId": user_id,
        "total_gender": {},
        "total_gender_distribution": {},
        "total_age": {},
        "total_age_distribution": {},
        "events": [event_id],
    }

    # Calculate totals (in production, these would be cumulative)
    gender_counts = {"male": 0, "female": 0, "unknown": 0}
    for face in faces_data:
        gender = face.get("gender", "").strip().upper()
        if gender == "M":
            gender_counts["male"] += 1
        elif gender == "F":
            gender_counts["female"] += 1
        else:
            gender_counts["unknown"] += 1

    user_payload["total_gender"] = gender_counts
    user_payload["total_gender_distribution"] = build_gender_distribution_payload(faces_data)
    user_payload["total_age_distribution"] = build_age_distribution_payload(faces_data)

    # Calculate age totals
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

    user_payload["total_age"] = {"min": min(ages) if ages else 0, "max": max(ages) if ages else 0}

    response = post_json("/ingest/users/update", user_payload)

    mutation_id = response.get("id")
    if not mutation_id:
        raise RuntimeError("User update succeeded but no ID was returned")

    print(f"✅ User updated successfully, mutation ID: {mutation_id}")
    return mutation_id


def ingest_event_data(
    faces_data: list,
    total_faces: int,
    event_name: str = "Face Detection Event",
    event_price: float = 0.0,
    user_id: str | None = None,
) -> tuple[str, str]:
    """
    Complete workflow: Create event and update user.

    Args:
        faces_data: List of face data dictionaries
        total_faces: Total number of unique faces
        event_name: Name of the event
        event_price: Price associated with the event
        user_id: User ID to update (uses CONVEX_USER_ID env var if not provided)

    Returns:
        Tuple of (event_id, user_mutation_id)

    Raises:
        RuntimeError: If ingestion fails
        ValueError: If user_id is not provided and CONVEX_USER_ID is not set

    """
    # Use provided user_id or fall back to environment variable
    if user_id is None:
        user_id = DEFAULT_USER_ID

    if not user_id:
        raise ValueError("user_id must be provided or CONVEX_USER_ID environment variable must be set")

    print(f"\n{'=' * 80}")
    print("📊 INGESTING DATA TO CONVEX DATABASE")
    print(f"{'=' * 80}")
    print(f"Convex URL: {BASE_URL}")
    print(f"User ID: {user_id}")

    # Step 1: Create event
    event_id = create_event(faces_data, total_faces, event_name, event_price)

    # Step 2: Update user with event
    user_mutation_id = update_user_with_event(user_id, event_id, faces_data, total_faces)

    print(f"\n{'=' * 80}")
    print("✅ CONVEX INGESTION COMPLETE")
    print(f"{'=' * 80}\n")

    return event_id, user_mutation_id
