def debug(rank: int | list[int] = 0, stop: bool = False):
    """Start a debugger, connect to this debugger from VSCode, your launch.json should be like below:

    {
        "version": "0.2.0",
        "inputs": [
            {
                "type": "promptString",
                "id": "WorkerIP",
                "description": "IP of your worker",
                "default": "",
            },
            {
                "type": "promptString",
                "id": "WorkerPort",
                "description": "IP of your worker",
                "default": "5678",
            }
        ],
        "configurations": [
            {
                "name": "Python: Remote Attach",
                "type": "debugpy",
                "request": "attach",
                "connect": {
                    "host": "${input:WorkerIP}",
                    "port": "${input:WorkerPort}"
                },
                "pathMappings": [
                    {
                        "localRoot": "${workspaceFolder}",
                        "remoteRoot": "${workspaceFolder}"
                    }
                ],
                "justMyCode": false,
            }
        ]
    }
    """
    import socket
    try:
        import debugpy
    except ImportError:
        print("debugpy not installed")
        import os
        os.system("sudo pip install debugpy")
        import debugpy
    import torch.distributed as dist

    if not isinstance(rank, list):
        rank = [rank]
    if debugpy.is_client_connected():
        return
    if dist.is_initialized():
        my_rank = dist.get_rank()
    else:
        print("Torch not initialized!")
        my_rank = 0
    if my_rank in rank:
        try:
            host = socket.gethostbyname(socket.getfqdn(socket.gethostname()))
            port = 5678 + rank.index(my_rank)

            debugpy.listen((host, port), in_process_debug_adapter=False)
            print("\033[31m" + "=" * 80 + "\033[0m")
            print("\033[31mWaiting for debugger...\a\033[0m")
            print(f"\033[31mip: {host} port:{port}\033[0m", flush=True)
            print("\033[31m" + "=" * 80 + "\033[0m")
            debugpy.wait_for_client()
            print("Connected.")
            if stop:
                debugpy.breakpoint()
        except Exception as e:
            import traceback

            print("\n".join(traceback.format_exception(e)))