import argparse
import os
import signal
import subprocess

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Temporary scaffold to run prediction scripts"
    )

    parser.add_argument("task", choices=["ttc", "nuclei"])
    parser.add_argument("dataset")
    parser.add_argument("--slabs", type=int, required=False)

    args = parser.parse_args()

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if args.slabs:
            if args.slabs > len(device_ids):
                parser.error("More slabs requests than devices in CUDA_VISIBLE_DEVICES")
            slabs = args.slabs
        else:
            slabs = len(device_ids)
    else:
        if args.slabs:
            slabs = args.slabs
            device_ids = list(range(slabs))
        else:
            parser.error("Either CUDA_VISIBLE_DEVICES must be set or --slabs must be specified")

    TASKS = {
        "ttc": "gunpowder_ttc_predict.py",
        "nuclei": "gunpowder_nuclei_predict.py",
    }
    task = TASKS[args.task]
    path = os.path.dirname(os.path.abspath(__file__))

    processes = set(
        subprocess.Popen(f"python {os.path.join(path, task)} {i} {slabs} {args.dataset}",
            shell=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(device_ids[i])}
            )
        for i in range(slabs)
    )

    def handle_signal(signum, _frame):
        for proc in processes:
            if proc.poll() is None:
                proc.send_signal(signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    for p in processes:
        p.wait()
