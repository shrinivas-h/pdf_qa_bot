import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed


class ParallelProcessorThreadPool:
    def __init__(self, function, thread_name_prefix="", num_threads=None):
        self.function = function
        self.num_threads = num_threads or 4
        self.thread_name_prefix = thread_name_prefix

    def run(self, *args):
        results = []
        with ThreadPoolExecutor(max_workers=self.num_threads, thread_name_prefix=self.thread_name_prefix) as executor:
            future_to_args = {executor.submit(self.function, *arg): arg for arg in args}
            for future in as_completed(future_to_args):
                result = future.result()
                results.append(result)
        return results


class ParallelProcessorMultiprocessing:
    def __init__(self, function, num_processes=None):
        self.function = function
        self.num_processes = num_processes or multiprocessing.cpu_count()

    def run(self, *args):
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            results = pool.starmap(self.function, args)
        return results
