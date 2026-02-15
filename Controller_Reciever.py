"""
JetBot UDP motor receiver
Packet format matches sender:
PACK_FMT = "<Idff"  (seq:uint32, t_sent:double, left:float, right:float)
"""

import argparse
import socket
import struct
import time

# JetBot motor control
try:
    from jetbot import Robot
except ImportError:
    Robot = None

PACK_FMT = "<Idff"
PACK_SIZE = struct.calcsize(PACK_FMT)


def is_seq_newer(seq, last_seq):
    """Return True if seq is newer than last_seq in uint32 space (handles wrap-around)."""
    if last_seq is None:
        return True
    diff = (seq - last_seq) & 0xFFFFFFFF
    return 0 < diff < 0x80000000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="0.0.0.0", help="IP to bind on JetBot")
    ap.add_argument("--port", type=int, default=5005, help="UDP port to listen on")
    ap.add_argument("--watchdog", type=float, default=0.25,
                    help="Seconds without valid packet -> stop motors")
    ap.add_argument("--print_hz", type=float, default=0.0,
                    help="Status print rate (Hz). 0 disables printing.")
    args = ap.parse_args()

    if Robot is None:
        raise RuntimeError(
            "Could not import jetbot.Robot. On JetBot, install/enable jetbot package "
            "or adapt this script to your motor driver."
        )

    robot = Robot()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))

    sock.setblocking(False)

    print(f"[START] Listening on {args.bind}:{args.port}")
    print(f"        PACK_FMT={PACK_FMT} PACK_SIZE={PACK_SIZE}")
    print(f"        watchdog={args.watchdog:.3f}s (t_sent ignored)")
    print("Ctrl+C to quit.\n")

    last_rx_time = 0.0
    last_seq = None
    last_addr = None  
    last_left = 0.0
    last_right = 0.0

    # Stats
    pkts_ok = pkts_bad = pkts_ooo = 0
    last_print = time.perf_counter()

    def stop():
        nonlocal last_left, last_right
        robot.left_motor.value = 0.0
        robot.right_motor.value = 0.0
        last_left, last_right = 0.0, 0.0

    # Ensure stopped at start
    stop()

    try:
        while True:
            now = time.time()

            data = None
            addr = None
            while True:
                try:
                    d, a = sock.recvfrom(2048)
                    data, addr = d, a  # overwrite => newest packet wins
                except BlockingIOError:
                    break  # no more packets queued

            if data:
                if len(data) != PACK_SIZE:
                    pkts_bad += 1
                else:
                    try:
                        seq, t_sent, left, right = struct.unpack(PACK_FMT, data)
                    except struct.error:
                        pkts_bad += 1
                    else:
                        # treat as new session and reset sequencing.
                        if last_addr is None or addr != last_addr:
                            print(f"[INFO] New sender {addr}, resetting sequence tracking.")
                            last_addr = addr
                            last_seq = None

                        # sequencing: drop out-of-order / duplicates
                        if not is_seq_newer(seq, last_seq):
                            pkts_ooo += 1
                        else:
                            last_seq = seq
                            last_rx_time = now

                            # NaN check: NaN != NaN
                            if not (left == left and right == right):
                                pkts_bad += 1
                            else:
                                left = float(left)
                                right = float(right)

                                # Clamp to safe range
                                if left > 1.0:
                                    left = 1.0
                                elif left < -1.0:
                                    left = -1.0
                                if right > 1.0:
                                    right = 1.0
                                elif right < -1.0:
                                    right = -1.0

                                # Apply commands directly
                                robot.left_motor.value = left
                                robot.right_motor.value = right
                                last_left, last_right = left, right
                                pkts_ok += 1

            # Watchdog: stop if no valid packet arrived within watchdog seconds
            if last_rx_time > 0 and (now - last_rx_time) > args.watchdog:
                if last_left != 0.0 or last_right != 0.0:
                    stop()
                last_seq = None
                last_addr = None
                last_rx_time = 0.0

            # Status printing (disabled by default)
            if args.print_hz > 0:
                tperf = time.perf_counter()
                if (tperf - last_print) >= (1.0 / args.print_hz):
                    dt = (now - last_rx_time) if last_rx_time > 0 else float("inf")
                    print(
                        f"seq={last_seq} dt={dt:0.3f}s "
                        f"L={last_left:+0.3f} R={last_right:+0.3f} | "
                        f"ok={pkts_ok} bad={pkts_bad} ooo={pkts_ooo}"
                    )
                    last_print = tperf

            # Tiny sleep to avoid maxing CPU while still being responsive
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[STOP] KeyboardInterrupt: stopping motors and closing socket.")
    finally:
        stop()
        sock.close()


if __name__ == "__main__":
    main()
