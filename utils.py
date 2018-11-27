import time


class Timer:
    def __init__(self):
        self.start_times = []
        self.jobs = []
        self.time_list = []

    def start(self, job, verbal=True):
        self.jobs.append(job)
        self.start_times.append(time.time())
        if verbal:
            print("[I] {job} started.".format(job=job))

    def stop(self, verbal=True):
        if self.jobs:
            elapsed_time = time.time() - self.start_times.pop()
            if verbal:
                print("[I] {job} finished in {elapsed_time:0.3f} s."
                      .format(job=self.jobs.pop(), elapsed_time=elapsed_time))
            return elapsed_time

    def remaining_time(self, elapsed_time, num_jobs):
        self.time_list.append(elapsed_time)
        avg_elapsed_time = sum(self.time_list) / len(self.time_list)
        rt = avg_elapsed_time * (num_jobs - len(self.time_list))
        print('Remaining time: {:0.3f} s'.format(rt), end='\r', flush=True)


class Log:
    verbose = True

    @staticmethod
    def log(text):
        if Log.verbose:
            print(text)
