import os
import json
import psutil

class SystemMetrics:
    def __init__(self):
        pass

    @staticmethod
    def get_cpu_utilization():
        return (psutil.cpu_percent(interval=1),)

    @staticmethod
    def get_ram_utilization():
        mem = psutil.virtual_memory()
        return (mem.percent,)

    @staticmethod
    def get_swap_utilization():
        swap = psutil.swap_memory()
        return (swap.percent,)

    @staticmethod
    def get_process_cpu_utilization(pid):
        process = psutil.Process(pid)
        return (process.cpu_percent(interval=1),)

    @staticmethod
    def get_process_ram_utilization(pid):
        process = psutil.Process(pid)
        mem_info = process.memory_info()
        return (mem_info.rss / (1024 ** 2),)  # Convert to megabytes

    @staticmethod
    def get_process_swap_utilization(pid):
        process = psutil.Process(pid)
        mem_info = process.memory_info()
        swap = psutil.swap_memory()
        return ((swap.used - mem_info.rss) / (1024 ** 2),)  # Convert to megabytes


class MetricsWrapper:
    @staticmethod
    def get_metrics(system=True, process=True):
        system_metrics = {}
        process_metrics = {}

        if system:
            system_metrics["cpu_utilization"] = SystemMetrics.get_cpu_utilization()[0]
            system_metrics["ram_utilization"] = SystemMetrics.get_ram_utilization()[0]
            system_metrics["swap_utilization"] = SystemMetrics.get_swap_utilization()[0]

        if process:
            pid = os.getpid()
            process_metrics["cpu_utilization"] = SystemMetrics.get_process_cpu_utilization(pid)[0]
            process_metrics["ram_utilization"] = SystemMetrics.get_process_ram_utilization(pid)[0]
            process_metrics["swap_utilization"] = SystemMetrics.get_process_swap_utilization(pid)[0]

        metrics = {
            "system_metrics": system_metrics,
            "process_metrics": process_metrics
        }

        return json.dumps(metrics, indent=4)



# Example usage
def get_utilization_json():
    return MetricsWrapper.get_metrics(system=True, process=True)
#print(metrics)

