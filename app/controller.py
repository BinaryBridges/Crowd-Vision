import os
import time
from datetime import UTC, datetime

import redis

from app.calculations import run_all_calculations
from app.redis_client import redis_client


def start_controller():
    """Main loop for the controller that monitors worker completion."""
    print("🎯 Starting Controller - Worker Completion Monitor")
    print("=" * 60)

    # Wait for Redis to be ready
    if not redis_client.wait_for_redis(timeout=300):
        print("❌ Redis failed to become ready. Exiting.")
        return 1

    print("✅ Redis is ready, starting monitoring loop")

    # Track system state
    last_status_time = 0
    last_cleanup_time = 0
    grace_period_start = None
    grace_period_duration = int(os.getenv("CONTROLLER_GRACE_PERIOD", "60"))  # Default: 60 seconds
    cleanup_interval = 300  # Clean up old messages every 5 minutes
    check_interval = int(os.getenv("CONTROLLER_CHECK_INTERVAL", "5"))  # Default: 5 seconds
    status_print_interval = int(os.getenv("CONTROLLER_STATUS_INTERVAL", "30"))  # Default: 30 seconds
    system_was_idle = False

    print("⚙️  Configuration:")
    print(f"   Check interval: {check_interval}s")
    print(f"   Status print interval: {status_print_interval}s")
    print(f"   Grace period: {grace_period_duration}s")

    try:
        while True:
            current_time = time.time()

            # Check system completion status
            status = redis_client.check_system_completion()

            # Print periodic status updates
            if current_time - last_status_time >= status_print_interval:
                print_status_update(status)
                last_status_time = current_time

            # Periodic stream cleanup (every 5 minutes)
            if current_time - last_cleanup_time >= cleanup_interval:
                try:
                    removed = redis_client.cleanup_old_messages(
                        max_age_seconds=3600
                    )  # Remove messages older than 1 hour
                    if removed > 0:
                        print(f"🧹 Cleaned up {removed} old messages from stream")
                except redis.exceptions.RedisError as e:
                    print(f"⚠️  Stream cleanup warning: {e}")
                last_cleanup_time = current_time

            # Check if system is complete
            if status["system_complete"]:
                if not system_was_idle:
                    # System just became idle, start grace period
                    grace_period_start = current_time
                    system_was_idle = True
                    print(f"🔄 System appears idle, starting {grace_period_duration}s grace period...")
                    print(f"   Videos complete: {status['videos']['all_complete']}")
                    print(f"   Stream idle: {status['stream']['is_idle']}")
                    print(f"   Motion workers: {status['workers']['motion_detection']}")
                elif current_time - grace_period_start >= grace_period_duration:
                    # Grace period completed, system is truly done
                    print_completion_summary(status)

                    run_all_calculations()

                    break
                else:
                    # Still in grace period
                    remaining = grace_period_duration - (current_time - grace_period_start)
                    print(f"⏳ Grace period: {remaining:.0f}s remaining...")
            else:
                # System is not idle, reset grace period
                if system_was_idle:
                    print("🔄 New work detected, resetting grace period")
                system_was_idle = False
                grace_period_start = None

            # Sleep before next check
            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n🛑 Controller interrupted by user")
        return 0
    except (redis.exceptions.RedisError, ValueError, RuntimeError) as e:
        print(f"\n❌ Controller error: {e}")
        return 1

    print("🎯 Controller completed successfully")
    return 0


def print_status_update(status: dict) -> None:
    """Print a formatted status update."""
    timestamp = datetime.now(tz=UTC).strftime("%H:%M:%S")

    print(f"\n📊 Status Update [{timestamp}]")
    print("-" * 40)

    # Video processing status
    videos = status["videos"]
    print(f"📹 Videos: {videos['completed']}/{videos['total']} completed")
    if videos["available"] > 0:
        print(f"   📋 Available: {videos['available']}")
    if videos["locked"] > 0:
        print(f"   🔒 Locked: {videos['locked']}")

    # Stream processing status
    stream = status["stream"]
    total_unprocessed = stream.get("total_unprocessed", stream["stream_length"] + stream["pending_messages"])
    print(
        f"🌊 Stream: {stream['stream_length']} queued, {stream['pending_messages']} pending ({total_unprocessed} total unprocessed)"
    )

    # Worker status
    workers = status["workers"]
    print(f"👥 Workers: {workers['motion_detection']} motion, {workers['image_processor']} image")

    # Overall status
    if status["system_complete"]:
        print("✅ System Status: IDLE")
    else:
        print("🔄 System Status: ACTIVE")


def print_completion_summary(status: dict) -> None:
    """Print the final completion summary."""
    print("\n" + "=" * 60)
    print("🎉 ALL WORKERS COMPLETED - System is fully idle")
    print("=" * 60)

    # Summary statistics
    videos = status["videos"]
    print("📊 Summary:")
    print(f"   📹 Videos processed: {videos['completed']}")

    # Get face count from Redis
    try:
        # Count faces in the database
        face_keys = redis_client.storage.keys("*")
        # Filter out non-face keys (worker heartbeats, etc.)
        face_count = len([k for k in face_keys if not k.startswith(("worker:", "video_", "sim:"))])
        print(f"   👤 Faces detected: {face_count}")
    except redis.exceptions.RedisError:
        print("   👤 Faces detected: Unable to count")

    # Worker summary
    workers = status["workers"]
    print(f"   👥 Final worker count: {workers['total_active']} active")

    print("\n⏱️  System processing complete!")
    print("=" * 60)


def reconcile():
    """Optional: coordination/scheduling."""


def shutdown():
    """Optional: graceful shutdown."""
