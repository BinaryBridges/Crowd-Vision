# app/__main__.py
import os
import signal
import threading

from app.controller import start_controller
from app.worker import start_worker

ROLE = os.getenv("ROLE", "worker").lower()
_stop = threading.Event()


def _sigterm(*_):
    _stop.set()


signal.signal(signal.SIGTERM, _sigterm)
signal.signal(signal.SIGINT, _sigterm)


def idle_forever():
    # sleep in long chunks until SIGTERM/SIGINT
    while not _stop.is_set():
        _stop.wait(3600)


def main():
    if ROLE == "controller":
        start_controller()  # currently no-op
        idle_forever()
    elif ROLE == "worker":
        start_worker()  # currently no-op
        idle_forever()
    else:
        idle_forever()


if __name__ == "__main__":
    main()
