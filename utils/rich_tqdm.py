import time
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    # DownloadColumn,
    FileSizeColumn,
    TransferSpeedColumn,
)
from rich.console import Console

console=Console()

progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    SpinnerColumn(),
    BarColumn(),
    # FileSizeColumn(),
    # DownloadColumn(),
    TextColumn("[progress.elapsed]{task.completed}/{task.total}"),
    TextColumn("[progress.fields]{task.fields}"),
    # TransferSpeedColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeRemainingColumn(),
    TimeElapsedColumn(),
    console=console,
    transient=False,
)

# epoch_tqdm = progress.add_task(description="epoch progress", total=10)
# batch_tqdm = progress.add_task(description="batch progress", total=100)
# progress.start() ## 开启
# for ep in range(10):
#     for batch in range(100):
#         # print("ep: {} batch: {}".format(ep, batch))
#         progress.advance(batch_tqdm, advance=1)
#         time.sleep(0.1)
#     progress.advance(epoch_tqdm, advance=1)
#     progress.reset(batch_tqdm)