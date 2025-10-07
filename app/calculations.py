"""
Comprehensive data analysis and calculations for face detection data.
All calculations are performed on data from Redis DB 0 (face storage).
"""

import traceback
from operator import itemgetter

import redis

from app import config
from app.convex import ingest_event_data
from app.redis_client import redis_client


def run_all_calculations():
    """
    Run comprehensive calculations on all face data stored in Redis DB 0.
    Calculates demographics, statistics, and insights from the collected data.
    """
    print("\n" + "=" * 80)
    print("📊 RUNNING COMPREHENSIVE DATA ANALYSIS")
    print("=" * 80)

    try:
        # Create a raw Redis connection without decode_responses to handle binary data
        raw_storage = redis.Redis(
            host=redis_client.host,
            port=redis_client.port,
            db=0,  # Face storage database
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            decode_responses=False,  # Don't decode binary data
        )

        # Get all face keys from Redis DB 0 (should be purely face data)
        all_keys = raw_storage.keys("*")
        total_faces = len(all_keys)

        if total_faces == 0:
            print("\n⚠️  No face data found in storage. Nothing to calculate.")
            print("=" * 80)
            return

        print(f"\n🔍 Found {total_faces} unique faces in database")
        print("📥 Retrieving face data...")

        # Retrieve all face data
        faces_data = []
        for key in all_keys:
            try:
                face_hash = raw_storage.hgetall(key)
                if face_hash:
                    # Decode only the text fields we need, skip binary 'vec' field
                    faces_data.append({
                        "id": key.decode("utf-8") if isinstance(key, bytes) else key,
                        "age": face_hash.get(b"age", b"").decode("utf-8") if b"age" in face_hash else "",
                        "gender": face_hash.get(b"gender", b"").decode("utf-8") if b"gender" in face_hash else "",
                        "cameras": face_hash.get(b"cameras", b"").decode("utf-8") if b"cameras" in face_hash else "",
                        "seen_count": face_hash.get(b"seen_count", b"1").decode("utf-8")
                        if b"seen_count" in face_hash
                        else "1",
                        "confidence": face_hash.get(b"confidence", b"").decode("utf-8")
                        if b"confidence" in face_hash
                        else "",
                    })
            except (redis.exceptions.RedisError, UnicodeDecodeError, ValueError) as e:
                print(f"⚠️  Warning: Failed to retrieve data for key {key}: {e}")
                continue

        print(f"✅ Successfully retrieved {len(faces_data)} face records")

        # Run all calculation modules
        calculate_total_count(total_faces)
        calculate_age_statistics(faces_data, total_faces)
        calculate_age_distribution(faces_data, total_faces)
        calculate_gender_distribution(faces_data, total_faces)
        calculate_gender_by_age_group(faces_data)
        calculate_camera_distribution(faces_data, total_faces)
        calculate_data_quality_metrics(faces_data, total_faces)
        print_summary(faces_data, total_faces)

        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80 + "\n")

        # Update event in Convex database
        try:
            # Get event configuration from config
            event_price = config.CONVEX_EVENT_PRICE
            event_id = config.CONVEX_EVENT_ID
            user_id = config.CONVEX_USER_ID

            # Only attempt update if both event_id and user_id are configured
            if event_id and user_id:
                response = ingest_event_data(
                    faces_data=faces_data,
                    total_faces=total_faces,
                    event_price=event_price,
                    event_id=event_id,
                    user_id=user_id,
                )
                print("✅ Event and user totals successfully updated in Convex")
                print(f"   Response: {response}")
            else:
                missing_vars = []
                if not event_id:
                    missing_vars.append("CONVEX_EVENT_ID")
                if not user_id:
                    missing_vars.append("CONVEX_USER_ID")
                print(f"\n⚠️  Skipping Convex update: {', '.join(missing_vars)} not configured")
                print("   Set the missing environment variables to enable Convex integration")
        except (ValueError, RuntimeError, ConnectionError) as convex_error:
            print(f"\n⚠️  Warning: Failed to update event in Convex: {convex_error}")
            print("   Analysis results are still available above")
            traceback.print_exc()

    except (redis.exceptions.RedisError, ValueError) as e:
        print(f"\n❌ Error during calculations: {e}")
        traceback.print_exc()
        print("=" * 80 + "\n")


def calculate_total_count(total_faces: int):
    """Calculate and display total face count."""
    print("\n" + "-" * 80)
    print("📈 TOTAL COUNTS")
    print("-" * 80)
    print(f"Total Unique Faces Detected: {total_faces}")


def calculate_age_statistics(faces_data: list, total_faces: int):
    """Calculate and display age statistics."""
    print("\n" + "-" * 80)
    print("👶 AGE ANALYSIS")
    print("-" * 80)

    ages = []
    for face in faces_data:
        age_str = face["age"]
        if age_str and age_str.strip():
            try:
                age = int(age_str)
                if 0 <= age <= 120:  # Sanity check
                    ages.append(age)
            except (ValueError, TypeError):
                pass

    if ages:
        avg_age = sum(ages) / len(ages)
        min_age = min(ages)
        max_age = max(ages)
        median_age = sorted(ages)[len(ages) // 2]

        print(f"Average Age: {avg_age:.2f} years")
        print(f"Median Age: {median_age} years")
        print(f"Age Range: {min_age} - {max_age} years")
        print(f"Faces with Age Data: {len(ages)} ({len(ages) / total_faces * 100:.1f}%)")
    else:
        print("No valid age data available")


def calculate_age_distribution(faces_data: list, _total_faces: int):
    """Calculate and display age distribution in 10-year brackets."""
    print("\n" + "-" * 80)
    print("📊 AGE DISTRIBUTION (10-Year Brackets)")
    print("-" * 80)

    ages = []
    for face in faces_data:
        age_str = face["age"]
        if age_str and age_str.strip():
            try:
                age = int(age_str)
                if 0 <= age <= 120:
                    ages.append(age)
            except (ValueError, TypeError):
                pass

    if ages:
        age_brackets = {
            "0-10": 0,
            "10-20": 0,
            "20-30": 0,
            "30-40": 0,
            "40-50": 0,
            "50-60": 0,
            "60-70": 0,
            "70-80": 0,
            "80-90": 0,
            "90-100": 0,
            "100+": 0,
        }

        for age in ages:
            if age < 10:
                age_brackets["0-10"] += 1
            elif age < 20:
                age_brackets["10-20"] += 1
            elif age < 30:
                age_brackets["20-30"] += 1
            elif age < 40:
                age_brackets["30-40"] += 1
            elif age < 50:
                age_brackets["40-50"] += 1
            elif age < 60:
                age_brackets["50-60"] += 1
            elif age < 70:
                age_brackets["60-70"] += 1
            elif age < 80:
                age_brackets["70-80"] += 1
            elif age < 90:
                age_brackets["80-90"] += 1
            elif age < 100:
                age_brackets["90-100"] += 1
            else:
                age_brackets["100+"] += 1

        for bracket, count in age_brackets.items():
            percentage = (count / len(ages)) * 100
            bar = "█" * int(percentage / 2)  # Visual bar chart
            print(f"{bracket:>8} years: {count:4d} ({percentage:5.1f}%) {bar}")
    else:
        print("No age data available for distribution analysis")


def calculate_gender_distribution(faces_data: list, total_faces: int):
    """Calculate and display gender distribution."""
    print("\n" + "-" * 80)
    print("⚧ GENDER DISTRIBUTION")
    print("-" * 80)

    gender_counts = {"M": 0, "F": 0, "Unknown": 0}

    for face in faces_data:
        gender = face["gender"].strip().upper() if face["gender"] else ""
        if gender == "M":
            gender_counts["M"] += 1
        elif gender == "F":
            gender_counts["F"] += 1
        else:
            gender_counts["Unknown"] += 1

    total_with_gender = gender_counts["M"] + gender_counts["F"]

    for gender, count in gender_counts.items():
        percentage = (count / total_faces) * 100
        bar = "█" * int(percentage / 2)

        if gender == "M":
            label = "Male"
        elif gender == "F":
            label = "Female"
        else:
            label = "Unknown"

        print(f"{label:>10}: {count:4d} ({percentage:5.1f}%) {bar}")

    if total_with_gender > 0:
        male_percentage = (gender_counts["M"] / total_with_gender) * 100
        female_percentage = (gender_counts["F"] / total_with_gender) * 100
        print("\nOf faces with gender data:")
        print(f"  Male: {male_percentage:.1f}%")
        print(f"  Female: {female_percentage:.1f}%")


def calculate_gender_by_age_group(faces_data: list):
    """Calculate and display gender distribution by age group (10-year brackets)."""
    print("\n" + "-" * 80)
    print("👥 GENDER DISTRIBUTION BY AGE GROUP (10-Year Brackets)")
    print("-" * 80)

    age_gender_data = []
    for face in faces_data:
        age_str = face["age"]
        gender = face["gender"].strip().upper() if face["gender"] else ""

        if age_str and age_str.strip() and gender in {"M", "F"}:
            try:
                age = int(age_str)
                if 0 <= age <= 120:
                    age_gender_data.append({"age": age, "gender": gender})
            except (ValueError, TypeError):
                pass

    if age_gender_data:
        # Use same age brackets as age distribution
        age_brackets = [
            ("0-10", 0, 10),
            ("10-20", 10, 20),
            ("20-30", 20, 30),
            ("30-40", 30, 40),
            ("40-50", 40, 50),
            ("50-60", 50, 60),
            ("60-70", 60, 70),
            ("70-80", 70, 80),
            ("80-90", 80, 90),
            ("90-100", 90, 100),
            ("100+", 100, 200),
        ]

        for bracket_name, min_age, max_age in age_brackets:
            group_data = [d for d in age_gender_data if min_age <= d["age"] < max_age]

            if group_data:
                male_count = sum(1 for d in group_data if d["gender"] == "M")
                female_count = sum(1 for d in group_data if d["gender"] == "F")
                total_group = len(group_data)

                male_pct = (male_count / total_group) * 100
                female_pct = (female_count / total_group) * 100

                print(f"\nAge {bracket_name:>8}: {total_group} faces")
                print(f"  Male:   {male_count:3d} ({male_pct:5.1f}%)")
                print(f"  Female: {female_count:3d} ({female_pct:5.1f}%)")
    else:
        print("Insufficient data for gender-by-age analysis")


def calculate_camera_distribution(faces_data: list, total_faces: int):
    """Calculate and display camera distribution."""
    print("\n" + "-" * 80)
    print("📹 CAMERA DISTRIBUTION")
    print("-" * 80)

    camera_counts = {}
    for face in faces_data:
        cameras = face["cameras"]
        if cameras:
            # Cameras might be comma-separated if face seen on multiple cameras
            for cam in cameras.split(","):
                cam = cam.strip()  # noqa: PLW2901
                if cam:
                    camera_counts[cam] = camera_counts.get(cam, 0) + 1

    if camera_counts:
        print(f"Total Cameras: {len(camera_counts)}")
        print("\nFaces detected per camera:")

        # Sort by count descending
        sorted_cameras = sorted(camera_counts.items(), key=itemgetter(1), reverse=True)

        for camera, count in sorted_cameras:
            percentage = (count / total_faces) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {camera:>20}: {count:4d} ({percentage:5.1f}%) {bar}")
    else:
        print("No camera data available")


def calculate_data_quality_metrics(faces_data: list, total_faces: int):
    """Calculate and display data quality metrics."""
    print("\n" + "-" * 80)
    print("✅ DATA QUALITY METRICS")
    print("-" * 80)

    faces_with_age = sum(1 for face in faces_data if face["age"] and face["age"].strip())
    faces_with_gender = sum(1 for face in faces_data if face["gender"] and face["gender"].strip())
    faces_with_camera = sum(1 for face in faces_data if face["cameras"] and face["cameras"].strip())
    faces_with_complete_data = sum(
        1
        for face in faces_data
        if face["age"]
        and face["age"].strip()
        and face["gender"]
        and face["gender"].strip()
        and face["cameras"]
        and face["cameras"].strip()
    )

    print(f"Faces with Age Data: {faces_with_age} ({faces_with_age / total_faces * 100:.1f}%)")
    print(f"Faces with Gender Data: {faces_with_gender} ({faces_with_gender / total_faces * 100:.1f}%)")
    print(f"Faces with Camera Data: {faces_with_camera} ({faces_with_camera / total_faces * 100:.1f}%)")
    print(f"Faces with Complete Data: {faces_with_complete_data} ({faces_with_complete_data / total_faces * 100:.1f}%)")

    # Overall data quality score
    quality_score = (
        (faces_with_age / total_faces * 33.33)
        + (faces_with_gender / total_faces * 33.33)
        + (faces_with_camera / total_faces * 33.34)
    )
    print(f"\nOverall Data Quality Score: {quality_score:.1f}%")


def print_summary(faces_data: list, total_faces: int):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)

    # Calculate summary metrics
    ages = [
        int(face["age"])
        for face in faces_data
        if face["age"] and face["age"].strip() and face["age"].isdigit() and 0 <= int(face["age"]) <= 120
    ]

    gender_counts = {"M": 0, "F": 0}
    for face in faces_data:
        gender = face["gender"].strip().upper() if face["gender"] else ""
        if gender == "M":
            gender_counts["M"] += 1
        elif gender == "F":
            gender_counts["F"] += 1

    camera_counts = {}
    for face in faces_data:
        cameras = face["cameras"]
        if cameras:
            for cam in cameras.split(","):
                cam = cam.strip()  # noqa: PLW2901
                if cam:
                    camera_counts[cam] = camera_counts.get(cam, 0) + 1

    faces_with_age = sum(1 for face in faces_data if face["age"] and face["age"].strip())
    faces_with_gender = sum(1 for face in faces_data if face["gender"] and face["gender"].strip())
    faces_with_camera = sum(1 for face in faces_data if face["cameras"] and face["cameras"].strip())

    quality_score = (
        (faces_with_age / total_faces * 33.33)
        + (faces_with_gender / total_faces * 33.33)
        + (faces_with_camera / total_faces * 33.34)
    )

    # Print summary
    print(f"✓ Total Unique Faces: {total_faces}")
    if ages:
        avg_age = sum(ages) / len(ages)
        print(f"✓ Average Age: {avg_age:.1f} years")
    if gender_counts["M"] + gender_counts["F"] > 0:
        print(f"✓ Gender Ratio (M:F): {gender_counts['M']}:{gender_counts['F']}")
    if camera_counts:
        print(f"✓ Cameras Used: {len(camera_counts)}")
    print(f"✓ Data Completeness: {quality_score:.1f}%")
