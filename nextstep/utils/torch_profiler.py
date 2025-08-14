import torch

from nextstep.utils.comm import dist_ctx


class TorchProfiler:
    def __init__(self, dir):
        self.profiler = None
        if dist_ctx.is_main_process:
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1,  # Skip first step
                    warmup=1,  # Warmup for 1 step
                    active=3,  # Profile for 3 steps
                    repeat=2,  # Repeat the schedule 2 times
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )

    def start(self):
        if self.profiler is not None:
            self.profiler.start()

    def step(self):
        if self.profiler is not None:
            self.profiler.step()

    def stop(self):
        if self.profiler is not None:
            self.profiler.stop()
