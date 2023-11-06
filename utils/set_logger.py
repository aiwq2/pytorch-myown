from loguru import logger
from utils.rich_tqdm import console
from datetime import datetime as dt
import os

class DDPLogger:
    def __init__(self,datetime,global_device,device):
        log_day_level=dt.strftime(dt.strptime(datetime,'%Y_%m_%d_%H_%M_%S'),'%Y_%m_%d')
        logger.remove()  # Remove default 'stderr' handler
        logger.add(
            sink=lambda _: console.print(_,end=''),
            level='TRACE',
            format="[not bold green]{time:YYYY-MM-DD HH:mm:ss.SSS}[/not bold green] | "
            "[yellow]{extra[device]}[/yellow] | "
            "[red]{level: <7}[/red] | "
            "[not bold cyan]{name}[/not bold cyan]:[not bold cyan]{function}[/not bold cyan]:[not bold cyan]{line}[/not bold cyan] - [normal white]{message}[/normal white]",
            filter=lambda record: record['extra'].get('device')==global_device,
            colorize=True
        )
        logger.add(
            os.path.join('log',log_day_level,str(device) + "_" + datetime + ".log"),
            format="{time:YYYY-MM-DD HH:mm:ss} | "
            "{extra[device]} | "
            "{level:<7} | "
            "{name}:{module}.{function}:{line}-{message}",
            encoding='utf-8',
            rotation="500 MB",
            retention='1 month',
        )
        self.log=logger.bind(device=device)

    def get_logger(self):
        return self.log