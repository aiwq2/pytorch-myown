from loguru import logger
from utils.rich_tqdm import console

class DDPLogger:
    def __init__(self,datetime,global_device,device):
        logger.remove()  # Remove default 'stderr' handler
        logger.add(
            sink=lambda _: console.print(_,end=''),
            level='TRACE',
            format="[not bold green]{time:YYYY-MM-DD HH:mm:ss.SSS}[/not bold green] | "
            "[red]{extra[device]}[/red] | "
            "[magenta]{level: <7}[/magenta] | "
            "[not bold cyan]{name}[/not bold cyan]:[not bold cyan]{function}[/not bold cyan]:[not bold cyan]{line}[/not bold cyan] - [normal white]{message}[/normal white]",
            filter=lambda record: record['extra'].get('device')==global_device,
            colorize=True
        )
        logger.add(
            "log/log_" + datetime + "/log_" + str(device) + "_" + datetime + ".log",
            format="{time:YYYY-MM-DD HH:mm:ss} | "
            "{extra[device]} | "
            "{level:<7} | "
            "{name}:From {module}.{function}:{line}-{message}",
            encoding='utf-8',
            rotation="500 MB",
            retention='1 month',
        )
        self.log=logger.bind(device=device)

    def get_logger(self):
        return self.log