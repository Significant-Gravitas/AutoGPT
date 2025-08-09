#!/usr/bin/env python3
"""
Unified scheduler debugging tool
- Test deployment
- Collect thread dumps (signal-based, works when FastAPI is stuck)  
- Monitor periodic dumps
"""
import subprocess
import sys
import time
from datetime import datetime

import requests


def find_scheduler_pod():
    """Find the running scheduler pod"""
    result = subprocess.run(
        "kubectl get pods -n dev-agpt --no-headers".split(),
        capture_output=True,
        text=True,
    )

    for line in result.stdout.split("\n"):
        if "scheduler-server" in line and "Running" in line:
            return line.split()[0]
    return None


def test_deployment():
    """Test if the deployment has debugging enabled"""
    print("🧪 TESTING SCHEDULER DEBUG DEPLOYMENT")
    print("=" * 50)

    pod_name = find_scheduler_pod()
    if not pod_name:
        print("❌ No scheduler pod found")
        return False

    print(f"📍 Pod: {pod_name}")

    # Check if faulthandler is enabled
    print("🔍 Checking faulthandler setup...")
    log_result = subprocess.run(
        f"kubectl logs -n dev-agpt {pod_name} --tail=50".split(),
        capture_output=True,
        text=True,
    )

    faulthandler_enabled = "Faulthandler enabled" in log_result.stdout
    periodic_enabled = "Periodic thread dump monitor started" in log_result.stdout

    if faulthandler_enabled:
        print("✅ Faulthandler is enabled")
    else:
        print("❌ Faulthandler not found in logs")

    if periodic_enabled:
        print("✅ Periodic monitoring is enabled")
    else:
        print("❌ Periodic monitoring not found in logs")

    # Test signal sending
    print("\\n📡 Testing signal delivery...")
    signal_result = subprocess.run(
        f"kubectl exec -n dev-agpt {pod_name} -- kill -USR2 1".split(),
        capture_output=True,
        text=True,
    )

    if signal_result.returncode == 0:
        print("✅ Signal sent successfully")
        time.sleep(2)

        # Check for thread dump in logs
        new_logs = subprocess.run(
            f"kubectl logs -n dev-agpt {pod_name} --tail=20".split(),
            capture_output=True,
            text=True,
        )
        if "SIGNAL THREAD DUMP" in new_logs.stdout:
            print("✅ Thread dump appeared in logs!")
        else:
            print("⚠️  No thread dump found (might take a moment)")
    else:
        print(f"❌ Signal failed: {signal_result.stderr}")

    # Test HTTP API (should work when not stuck)
    print("\\n🌐 Testing HTTP API...")
    pf_process = None
    try:
        pf_process = subprocess.Popen(
            f"kubectl port-forward -n dev-agpt {pod_name} 8003:8003".split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(2)

        response = requests.get("http://localhost:8003/debug_thread_dump", timeout=10)
        if response.status_code == 200:
            print("✅ HTTP API working")
            print(f"   Thread count found: {'Total threads:' in response.text}")
        else:
            print(f"⚠️  HTTP API returned: {response.status_code}")

    except Exception as e:
        print(f"⚠️  HTTP API failed: {e}")
    finally:
        if pf_process:
            try:
                pf_process.terminate()
                pf_process.wait()
            except Exception:
                pass

    success = faulthandler_enabled and signal_result.returncode == 0
    print(
        f"\\n{'✅ DEPLOYMENT TEST PASSED' if success else '❌ DEPLOYMENT TEST FAILED'}"
    )
    return success


def collect_thread_dump():
    """Collect comprehensive thread dump (works even when scheduler is stuck)"""
    print("🚨 COLLECTING THREAD DUMP FROM SCHEDULER")
    print("=" * 60)

    pod_name = find_scheduler_pod()
    if not pod_name:
        print("❌ No scheduler pod found")
        return False

    print(f"📍 Pod: {pod_name}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Send both signals for maximum coverage
    print("📡 Sending signals for thread dumps...")

    # SIGUSR1 (faulthandler)
    result1 = subprocess.run(
        f"kubectl exec -n dev-agpt {pod_name} -- kill -USR1 1".split(),
        capture_output=True,
        text=True,
    )
    print(f"   SIGUSR1: {'✅' if result1.returncode == 0 else '❌'}")

    time.sleep(1)

    # SIGUSR2 (custom handler)
    result2 = subprocess.run(
        f"kubectl exec -n dev-agpt {pod_name} -- kill -USR2 1".split(),
        capture_output=True,
        text=True,
    )
    print(f"   SIGUSR2: {'✅' if result2.returncode == 0 else '❌'}")

    time.sleep(3)  # Give signals time to execute

    # Collect logs with thread dumps
    print("📋 Collecting logs...")
    log_result = subprocess.run(
        f"kubectl logs -n dev-agpt {pod_name} --tail=500".split(),
        capture_output=True,
        text=True,
    )

    # Save everything
    dump_file = f"THREAD_DUMP_{timestamp}.txt"
    with open(dump_file, "w") as f:
        f.write("SCHEDULER THREAD DUMP COLLECTION\\n")
        f.write(f"Timestamp: {datetime.now()}\\n")
        f.write(f"Pod: {pod_name}\\n")
        f.write("=" * 80 + "\\n\\n")
        f.write("FULL LOGS (last 500 lines):\\n")
        f.write("-" * 40 + "\\n")
        f.write(log_result.stdout)

    print(f"💾 Full dump saved: {dump_file}")

    # Extract and show thread dump preview
    lines = log_result.stdout.split("\\n")
    thread_dumps = []
    in_dump = False
    current_dump = []

    for line in lines:
        if any(
            marker in line
            for marker in ["SIGNAL THREAD DUMP", "Fatal Python error", "Thread 0x"]
        ):
            if current_dump:
                thread_dumps.append(current_dump)
            current_dump = [line]
            in_dump = True
        elif in_dump and (
            "END SIGNAL THREAD DUMP" in line or "Current thread 0x" in line
        ):
            current_dump.append(line)
            thread_dumps.append(current_dump)
            current_dump = []
            in_dump = False
        elif in_dump:
            current_dump.append(line)

    if current_dump:
        thread_dumps.append(current_dump)

    if thread_dumps:
        print(f"\\n🔍 FOUND {len(thread_dumps)} THREAD DUMP(S):")
        print("-" * 50)

        # Show the most recent/complete dump
        latest_dump = thread_dumps[-1]
        for i, line in enumerate(latest_dump[:50]):  # First 50 lines
            print(line)

        if len(latest_dump) > 50:
            print("... (truncated, see full dump in file)")

        # Create separate file with just thread dumps
        clean_dump_file = f"CLEAN_THREAD_DUMP_{timestamp}.txt"
        with open(clean_dump_file, "w") as f:
            f.write(f"EXTRACTED THREAD DUMPS - {datetime.now()}\\n")
            f.write("=" * 60 + "\\n\\n")
            for i, dump in enumerate(thread_dumps, 1):
                f.write(f"DUMP #{i}:\\n")
                f.write("-" * 30 + "\\n")
                f.write("\\n".join(dump))
                f.write("\\n\\n")

        print(f"🎯 Clean thread dumps saved: {clean_dump_file}")

    else:
        print("⚠️  No thread dumps found in logs")
        print("Recent log lines:")
        for line in lines[-10:]:
            print(f"   {line}")

    # Try HTTP backup (will fail if scheduler is stuck, but worth trying)
    print("\\n🌐 Attempting HTTP backup...")
    pf_process = None
    try:
        pf_process = subprocess.Popen(
            f"kubectl port-forward -n dev-agpt {pod_name} 8003:8003".split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(2)

        response = requests.get("http://localhost:8003/debug_thread_dump", timeout=5)
        if response.status_code == 200:
            http_file = f"HTTP_THREAD_DUMP_{timestamp}.txt"
            with open(http_file, "w") as f:
                f.write(response.text)
            print(f"✅ HTTP backup saved: {http_file}")
        else:
            print(f"⚠️  HTTP failed: {response.status_code}")

    except Exception as e:
        print(f"⚠️  HTTP failed (expected if stuck): {e}")
    finally:
        if pf_process:
            try:
                pf_process.terminate()
                pf_process.wait()
            except Exception:
                pass

    print("\\n✅ COLLECTION COMPLETE!")
    return len(thread_dumps) > 0


def monitor_periodic_dumps(duration_minutes=10):
    """Monitor periodic thread dumps for a specified duration"""
    print(f"👁️  MONITORING PERIODIC DUMPS FOR {duration_minutes} MINUTES")
    print("=" * 50)

    pod_name = find_scheduler_pod()
    if not pod_name:
        print("❌ No scheduler pod found")
        return

    print(f"📍 Pod: {pod_name}")
    print("⏰ Watching for periodic status messages and thread dumps...")

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

    # Get current log position (for reference, not used currently)
    # Could be used for tracking new vs old logs if needed

    while time.time() < end_time:
        try:
            # Get new logs
            current_logs = subprocess.run(
                f"kubectl logs -n dev-agpt {pod_name} --tail=50".split(),
                capture_output=True,
                text=True,
            )

            for line in current_logs.stdout.split("\\n"):
                if "Periodic check:" in line:
                    print(f"📊 {line}")
                elif "SIGNAL THREAD DUMP" in line:
                    print(f"🚨 Thread dump detected: {line}")
                elif "No health check" in line:
                    print(f"⚠️  Health issue: {line}")

            time.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            print("\\n⏹️  Monitoring stopped by user")
            break
        except Exception as e:
            print(f"Error during monitoring: {e}")
            break

    print("\\n✅ MONITORING COMPLETE")


def main():
    if len(sys.argv) < 2:
        print("🔧 SCHEDULER DEBUG TOOL")
        print("=" * 30)
        print("Usage:")
        print("  python scheduler_debug.py test         - Test deployment")
        print("  python scheduler_debug.py collect      - Collect thread dump")
        print("  python scheduler_debug.py monitor [min] - Monitor periodic dumps")
        print("  python scheduler_debug.py all          - Run test + collect")
        return

    command = sys.argv[1].lower()

    if command == "test":
        test_deployment()
    elif command == "collect":
        collect_thread_dump()
    elif command == "monitor":
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        monitor_periodic_dumps(duration)
    elif command == "all":
        print("Running complete debugging sequence...\\n")
        if test_deployment():
            print("\\n" + "=" * 50)
            collect_thread_dump()
        else:
            print("❌ Test failed, skipping collection")
    else:
        print(f"❌ Unknown command: {command}")


if __name__ == "__main__":
    main()
