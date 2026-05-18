import shutil
import sys
import time
from collections import deque


class TrainingTUI:
    def __init__(self, enabled=True):
        self.enabled = enabled and sys.stdout.isatty()
        self.events = deque(maxlen=8)
        self.phase = "idle"
        self.phase_detail = ""
        self.iteration = 0
        self.total_iterations = 0
        self.best_rating = 0.0
        self.learning_rate = 0.0
        self.last_loss = None
        self.last_eval = None
        self.start_time = time.time()
        self.device = "unknown"
        self.hf_repo = "disabled"
        self.resume_mode = False
        self.args_snapshot = {}

    def configure(self, total_iterations, device, args_snapshot=None, hf_repo="disabled", resume_mode=False):
        self.total_iterations = total_iterations
        self.device = device
        self.hf_repo = hf_repo
        self.resume_mode = resume_mode
        self.args_snapshot = args_snapshot or {}
        self.log("Session initialized")
        self.render()

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        line = f"{timestamp}  {message}"
        self.events.appendleft(line)
        if not self.enabled:
            print(line)
        self.render()

    def set_phase(self, phase, detail=""):
        self.phase = phase
        self.phase_detail = detail
        self.render()

    def update_iteration(self, iteration, learning_rate=None, last_loss=None, best_rating=None, last_eval=None):
        self.iteration = iteration
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if last_loss is not None:
            self.last_loss = last_loss
        if best_rating is not None:
            self.best_rating = best_rating
        if last_eval is not None:
            self.last_eval = last_eval
        self.render()

    def finish(self, message):
        self.set_phase("completed", message)
        self.log(message)

    def render(self):
        if not self.enabled:
            return

        width = max(88, min(shutil.get_terminal_size((120, 40)).columns, 140))
        lines = []
        lines.extend(self._panel("CONNECT4 RL TRAINER", [
            self._kv("phase", f"{self.phase} {self.phase_detail}".strip()),
            self._kv("device", self.device),
            self._kv("hf repo", self.hf_repo),
            self._kv("resume", "yes" if self.resume_mode else "no"),
        ], width))

        progress = self._progress_bar(width - 24)
        lines.extend(self._panel("PROGRESS", [
            self._kv("iteration", f"{self.iteration}/{self.total_iterations}"),
            self._kv("progress", progress),
            self._kv("elapsed", self._format_elapsed()),
        ], width))

        config_rows = [
            self._kv("games", str(self.args_snapshot.get("num_games", "-"))),
            self._kv("sims", str(self.args_snapshot.get("num_simulations", "-"))),
            self._kv("epochs", str(self.args_snapshot.get("num_epochs", "-"))),
            self._kv("batch", str(self.args_snapshot.get("batch_size", "-"))),
            self._kv("lr", self._format_lr(self.learning_rate or self.args_snapshot.get("learning_rate"))),
            self._kv("checkpoint every", str(self.args_snapshot.get("checkpoint_frequency", "-"))),
        ]
        lines.extend(self._panel("CONFIG", config_rows, width))

        metric_rows = [
            self._kv("last loss", self._format_metric(self.last_loss)),
            self._kv("best rating", self._format_metric(self.best_rating)),
            self._kv("last eval", self._format_metric(self.last_eval)),
        ]
        lines.extend(self._panel("METRICS", metric_rows, width))

        lines.extend(self._panel("EVENTS", list(self.events) or ["No events yet"], width))

        sys.stdout.write("\033[2J\033[H")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

    def _panel(self, title, body_lines, width):
        fg = "\033[38;5;255m"
        muted = "\033[38;5;250m"
        border = "\033[38;5;240m"
        reset = "\033[0m"
        top = f"{border}┌{'─' * (width - 2)}┐{reset}"
        label = f" {title} "
        label_line = f"{border}│{reset}{fg}{label:<{width - 2}}{reset}{border}│{reset}"
        rows = [top, label_line]
        for entry in body_lines:
            rows.append(f"{border}│{reset}{muted}{entry:<{width - 2}}{reset}{border}│{reset}")
        rows.append(f"{border}└{'─' * (width - 2)}┘{reset}")
        return rows

    def _progress_bar(self, width):
        if self.total_iterations <= 0:
            return "-" * max(10, width)
        filled = int((self.iteration / self.total_iterations) * width)
        filled = max(0, min(width, filled))
        return f"[{'█' * filled}{'·' * (width - filled)}]"

    def _format_elapsed(self):
        seconds = int(time.time() - self.start_time)
        hours, rem = divmod(seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _format_metric(self, value):
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _format_lr(self, value):
        if value in (None, 0):
            return "n/a"
        return f"{value:.2e}"

    def _kv(self, key, value):
        return f"{key:<18} {value}"
