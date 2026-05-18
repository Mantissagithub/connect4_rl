import argparse
import os
import site
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk


ROWS = 6
COLS = 7
CELL_SIZE = 72
PADDING = 24
BOARD_COLOR = "#1d4ed8"
BOARD_SHADOW = "#0f172a"
EMPTY_COLOR = "#0b1220"
P1_COLOR = "#ff5a5f"
P2_COLOR = "#f7c843"
BG_COLOR = "#0b1020"
PANEL_COLOR = "#121a2b"
PANEL_ALT_COLOR = "#0f1726"
ENTRY_COLOR = "#0d1525"
BORDER_COLOR = "#24324a"
TEXT_COLOR = "#e5edf7"
MUTED_TEXT_COLOR = "#9fb0c8"
ACCENT_COLOR = "#68a3ff"


def _bootstrap_torch_runtime():
    if os.environ.get("CONNECT4_RL_TORCH_BOOTSTRAPPED") == "1":
        return

    nvidia_base = Path(site.getsitepackages()[0]) / "nvidia"
    if not nvidia_base.exists():
        return

    lib_dirs = sorted({str(path.parent) for path in nvidia_base.rglob("lib/*.so*")})
    if not lib_dirs:
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = [part for part in current.split(":") if part]
    missing = [lib_dir for lib_dir in lib_dirs if lib_dir not in current_parts]
    if not missing:
        return

    os.environ["LD_LIBRARY_PATH"] = ":".join(missing + current_parts)
    os.environ["CONNECT4_RL_TORCH_BOOTSTRAPPED"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


_bootstrap_torch_runtime()


class SelfPlayViewer:
    def __init__(self, root: tk.Tk, checkpoint_path: str | None = None, simulations: int = 50, delay_ms: int = 900):
        self.root = root
        self.root.title("Connect4 RL Self-Play Viewer")
        self.root.configure(bg=BG_COLOR)

        self.checkpoint_path = checkpoint_path or self._get_latest_checkpoint()
        self.simulations_var = tk.IntVar(value=simulations)
        self.delay_var = tk.IntVar(value=delay_ms)
        self.status_var = tk.StringVar(value="Loading model...")
        self.value_var = tk.StringVar(value="Value estimate: -")
        self.turn_var = tk.StringVar(value="Current player: 1")
        self.result_var = tk.StringVar(value="Result: in progress")
        self.move_var = tk.StringVar(value="Move count: 0")
        self.banner_var = tk.StringVar(value="Self-play ready")

        self.model = None
        self.board = None
        self.current_player = 1
        self.move_count = 0
        self.autoplay = False
        self.busy = False
        self.last_search = {}

        self._load_dependencies()
        self._build_layout()
        self.reset_game()
        self._load_model()

    def _configure_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError as e:
            print(f"[error]: exception: {e}")

        style.configure(
            "Dark.TButton",
            background=PANEL_ALT_COLOR,
            foreground=TEXT_COLOR,
            bordercolor=BORDER_COLOR,
            focusthickness=1,
            focuscolor=ACCENT_COLOR,
            padding=(10, 6),
            relief="flat",
        )
        style.map(
            "Dark.TButton",
            background=[("active", "#18243a"), ("pressed", "#0d1525")],
            foreground=[("disabled", MUTED_TEXT_COLOR), ("active", TEXT_COLOR)],
        )

    def _load_dependencies(self):
        from game_engine_components.intialize_board import initialize_board
        from game_engine_components.make_move import make_move
        from game_engine_components.check_winner import check_winner
        from game_engine_components.is_draw import is_draw
        from game_engine_components.is_terminal import is_the_end
        from game_engine_components.get_valid_moves import get_valid_moves
        from game_engine_components.copy_game import deep_copy
        from mcts_components.run_simulation import run_simulation
        from mcts_components.get_action_visits import get_action_visits
        from neural_network_components.load_model import load_model_for_inference

        self.initialize_board = initialize_board
        self.make_move = make_move
        self.check_winner = check_winner
        self.is_draw = is_draw
        self.is_the_end = is_the_end
        self.get_valid_moves = get_valid_moves
        self.deep_copy = deep_copy
        self.run_simulation = run_simulation
        self.get_action_visits = get_action_visits
        self.load_model_for_inference = load_model_for_inference

    def _build_layout(self):
        self._configure_styles()
        frame = tk.Frame(self.root, bg=BG_COLOR)
        frame.pack(fill="both", expand=True, padx=18, pady=18)

        left = tk.Frame(frame, bg=BG_COLOR)
        left.pack(side="left", fill="both", expand=False)

        right = tk.Frame(frame, bg=BG_COLOR)
        right.pack(side="left", fill="both", expand=True, padx=(18, 0))

        title = tk.Label(
            left,
            text="Connect4 RL Self-Play",
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            font=("TkDefaultFont", 18, "bold"),
        )
        title.pack(anchor="w", pady=(0, 10))

        self.banner_label = tk.Label(
            left,
            textvariable=self.banner_var,
            bg=PANEL_ALT_COLOR,
            fg=TEXT_COLOR,
            padx=14,
            pady=10,
            anchor="w",
            justify="left",
            font=("TkDefaultFont", 11, "bold"),
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            highlightcolor=BORDER_COLOR,
        )
        self.banner_label.pack(fill="x", pady=(0, 12))

        self.canvas = tk.Canvas(
            left,
            width=COLS * CELL_SIZE + 2 * PADDING,
            height=ROWS * CELL_SIZE + 2 * PADDING,
            bg=BG_COLOR,
            highlightthickness=0,
        )
        self.canvas.pack()

        controls = tk.LabelFrame(
            right,
            text="Controls",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            bd=1,
            highlightbackground=BORDER_COLOR,
            highlightcolor=BORDER_COLOR,
            padx=12,
            pady=12,
        )
        controls.pack(fill="x")

        tk.Label(controls, text="Checkpoint", bg=PANEL_COLOR, fg=MUTED_TEXT_COLOR).grid(row=0, column=0, sticky="w")
        self.checkpoint_entry = tk.Entry(
            controls,
            width=54,
            bg=ENTRY_COLOR,
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            highlightcolor=ACCENT_COLOR,
        )
        self.checkpoint_entry.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(8, 8))
        self.checkpoint_entry.insert(0, self.checkpoint_path or "")

        ttk.Button(controls, text="Reload Model", command=self.reload_model, style="Dark.TButton").grid(row=0, column=4, sticky="ew")

        tk.Label(controls, text="MCTS simulations", bg=PANEL_COLOR, fg=MUTED_TEXT_COLOR).grid(row=1, column=0, sticky="w", pady=(10, 0))
        tk.Spinbox(
            controls,
            from_=1,
            to=500,
            textvariable=self.simulations_var,
            width=8,
            bg=ENTRY_COLOR,
            fg=TEXT_COLOR,
            buttonbackground=PANEL_ALT_COLOR,
            insertbackground=TEXT_COLOR,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            highlightcolor=ACCENT_COLOR,
        ).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(10, 0))

        tk.Label(controls, text="Move delay (ms)", bg=PANEL_COLOR, fg=MUTED_TEXT_COLOR).grid(row=1, column=2, sticky="w", pady=(10, 0))
        tk.Spinbox(
            controls,
            from_=100,
            to=5000,
            increment=100,
            textvariable=self.delay_var,
            width=8,
            bg=ENTRY_COLOR,
            fg=TEXT_COLOR,
            buttonbackground=PANEL_ALT_COLOR,
            insertbackground=TEXT_COLOR,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            highlightcolor=ACCENT_COLOR,
        ).grid(row=1, column=3, sticky="w", padx=(8, 0), pady=(10, 0))

        ttk.Button(controls, text="New Game", command=self.new_game, style="Dark.TButton").grid(row=2, column=0, sticky="ew", pady=(14, 0))
        ttk.Button(controls, text="Step", command=self.step_once, style="Dark.TButton").grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(14, 0))
        self.auto_button = ttk.Button(controls, text="Start Auto", command=self.toggle_autoplay, style="Dark.TButton")
        self.auto_button.grid(row=2, column=2, sticky="ew", padx=(8, 0), pady=(14, 0))
        ttk.Button(controls, text="Reset Board", command=self.reset_game, style="Dark.TButton").grid(row=2, column=3, sticky="ew", padx=(8, 0), pady=(14, 0))

        status = tk.LabelFrame(
            right,
            text="Status",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            bd=1,
            highlightbackground=BORDER_COLOR,
            highlightcolor=BORDER_COLOR,
            padx=12,
            pady=12,
        )
        status.pack(fill="x", pady=(14, 0))
        for idx, variable in enumerate([self.status_var, self.turn_var, self.move_var, self.value_var, self.result_var]):
            tk.Label(status, textvariable=variable, anchor="w", bg=PANEL_COLOR, fg=TEXT_COLOR).grid(row=idx, column=0, sticky="w")

        visits = tk.LabelFrame(
            right,
            text="Last Search Visit Counts",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            bd=1,
            highlightbackground=BORDER_COLOR,
            highlightcolor=BORDER_COLOR,
            padx=12,
            pady=12,
        )
        visits.pack(fill="both", expand=True, pady=(14, 0))
        self.visit_labels = []
        for col in range(COLS):
            var = tk.StringVar(value=f"Column {col}: -")
            label = tk.Label(visits, textvariable=var, anchor="w", bg=PANEL_COLOR, fg=MUTED_TEXT_COLOR)
            label.pack(fill="x")
            self.visit_labels.append(var)

        history = tk.LabelFrame(
            right,
            text="Move History",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            bd=1,
            highlightbackground=BORDER_COLOR,
            highlightcolor=BORDER_COLOR,
            padx=12,
            pady=12,
        )
        history.pack(fill="both", expand=True, pady=(14, 0))
        self.history_box = tk.Text(
            history,
            width=48,
            height=14,
            state="disabled",
            bg=ENTRY_COLOR,
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
            selectbackground="#274061",
            selectforeground=TEXT_COLOR,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            highlightcolor=ACCENT_COLOR,
        )
        self.history_box.pack(fill="both", expand=True)

    def _get_latest_checkpoint(self):
        checkpoint_dir = Path("checkpoints")
        if not checkpoint_dir.exists():
            return ""
        files = sorted(checkpoint_dir.glob("checkpoint_*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
        return str(files[0]) if files else ""

    def _load_model(self):
        checkpoint = self.checkpoint_entry.get().strip() if hasattr(self, "checkpoint_entry") else self.checkpoint_path
        self.checkpoint_path = checkpoint
        self.status_var.set(f"Loading model from {checkpoint or 'N/A'}")

        def worker():
            try:
                model = self.load_model_for_inference(checkpoint) if checkpoint else None
                if model is None:
                    self.root.after(0, lambda: self.status_var.set("Model load failed"))
                    return
                self.model = model
                self.root.after(0, lambda: self.status_var.set(f"Model ready: {checkpoint}"))
            except Exception as exc:
                self.root.after(0, lambda: self.status_var.set(f"Model load failed: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def reload_model(self):
        self._load_model()

    def reset_game(self):
        self.board = self.initialize_board()
        self.current_player = 1
        self.move_count = 0
        self.autoplay = False
        self.busy = False
        self.last_search = {}
        self.auto_button.config(text="Start Auto")
        self.turn_var.set("Current player: 1")
        self.result_var.set("Result: in progress")
        self.move_var.set("Move count: 0")
        self.value_var.set("Value estimate: -")
        self._set_banner("Self-play ready", TEXT_COLOR)
        self._set_history("")
        self._update_visit_labels({})
        self._draw_board()

    def new_game(self):
        self.reset_game()
        self.status_var.set("New self-play game initialized")

    def toggle_autoplay(self):
        self.autoplay = not self.autoplay
        self.auto_button.config(text="Pause Auto" if self.autoplay else "Start Auto")
        if self.autoplay and not self.busy:
            self._schedule_next_step()

    def _schedule_next_step(self):
        if self.autoplay:
            self.root.after(max(100, self.delay_var.get()), self.step_once)

    def step_once(self):
        if self.busy or self.model is None:
            return
        if self.is_the_end(self.board):
            self.autoplay = False
            self.auto_button.config(text="Start Auto")
            return

        self.busy = True
        self.status_var.set(f"Running MCTS for player {self.current_player}")

        def worker():
            try:
                board_copy = self.deep_copy(self.board)
                root = self.run_simulation(
                    board_copy,
                    current_player=self.current_player,
                    neural_net=self.model,
                    num_simulations=self.simulations_var.get(),
                )
                visits = self.get_action_visits(root)
                if not visits:
                    valid_moves = self.get_valid_moves(self.board)
                    move = valid_moves[0]
                else:
                    move = max(visits.items(), key=lambda item: item[1])[0]

                policy, value = self.model.predict(self.board)
                self.root.after(0, lambda: self._apply_move(move, visits, value, policy))
            except Exception as exc:
                self.root.after(0, lambda: self._handle_step_error(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _handle_step_error(self, exc):
        self.busy = False
        self.autoplay = False
        self.auto_button.config(text="Start Auto")
        self.status_var.set(f"Self-play step failed: {exc}")

    def _apply_move(self, move, visits, value, policy):
        success = self.make_move(self.board, move, self.current_player)
        if not success:
            self._handle_step_error(RuntimeError(f"illegal move {move}"))
            return

        self.move_count += 1
        actor = self.current_player
        self.current_player = 3 - self.current_player

        self.turn_var.set(f"Current player: {self.current_player}")
        self.move_var.set(f"Move count: {self.move_count}")
        self.value_var.set(f"Value estimate: {float(value):+.3f}")
        self.status_var.set(f"Player {actor} played column {move}")
        self.last_search = visits
        self._update_visit_labels(visits)
        self._append_history(actor, move, visits, policy, value)
        self._draw_board()
        self._update_result()

        self.busy = False
        if self.autoplay and not self.is_the_end(self.board):
            self._schedule_next_step()
        elif self.is_the_end(self.board):
            self.autoplay = False
            self.auto_button.config(text="Start Auto")

    def _update_result(self):
        winner = self.check_winner(self.board)
        if winner:
            self.result_var.set(f"Result: player {winner} wins")
            self.status_var.set(f"Game over: player {winner} wins after {self.move_count} moves")
            winner_color = P1_COLOR if winner == 1 else P2_COLOR
            self._set_banner(f"Winner: Player {winner}", winner_color)
        elif self.is_draw(self.board):
            self.result_var.set("Result: draw")
            self.status_var.set(f"Game over: draw after {self.move_count} moves")
            self._set_banner("Game drawn", MUTED_TEXT_COLOR)
        else:
            self.result_var.set("Result: in progress")
            self._set_banner(f"Player {self.current_player} to move", TEXT_COLOR)

    def _set_banner(self, message: str, color: str):
        self.banner_var.set(message)
        if hasattr(self, "banner_label"):
            self.banner_label.config(fg=color)

    def _draw_board(self):
        self.canvas.delete("all")
        width = COLS * CELL_SIZE
        height = ROWS * CELL_SIZE
        self.canvas.create_rectangle(
            PADDING - 6,
            PADDING - 6,
            PADDING + width + 6,
            PADDING + height + 6,
            fill=BOARD_SHADOW,
            outline="",
        )
        self.canvas.create_rectangle(PADDING, PADDING, PADDING + width, PADDING + height, fill=BOARD_COLOR, outline="")

        for row in range(ROWS):
            for col in range(COLS):
                x0 = PADDING + col * CELL_SIZE + 6
                y0 = PADDING + row * CELL_SIZE + 6
                x1 = x0 + CELL_SIZE - 12
                y1 = y0 + CELL_SIZE - 12
                cell = self.board[row][col]
                fill = EMPTY_COLOR if cell == 0 else P1_COLOR if cell == 1 else P2_COLOR
                self.canvas.create_oval(x0, y0, x1, y1, fill=fill, outline=BOARD_SHADOW, width=1)

        for col in range(COLS):
            self.canvas.create_text(
                PADDING + col * CELL_SIZE + CELL_SIZE / 2,
                10,
                text=str(col),
                fill=MUTED_TEXT_COLOR,
                font=("TkDefaultFont", 10, "bold"),
            )

    def _update_visit_labels(self, visits):
        total = sum(visits.values()) if visits else 0
        for col in range(COLS):
            value = visits.get(col, 0)
            pct = (value / total * 100.0) if total else 0.0
            self.visit_labels[col].set(f"Column {col}: {value:>3} visits ({pct:5.1f}%)")

    def _append_history(self, player, move, visits, policy, value):
        total_visits = sum(visits.values()) if visits else 0
        top = sorted(visits.items(), key=lambda item: item[1], reverse=True)[:3]
        visit_summary = ", ".join(
            f"{col}:{count}/{(count / total_visits * 100.0):.1f}%" for col, count in top
        ) if top and total_visits else "-"
        policy_summary = ", ".join(f"{idx}:{prob:.2f}" for idx, prob in enumerate(policy))
        line = (
            f"Move {self.move_count:02d} | P{player} -> column {move} | "
            f"value {float(value):+.3f}\n"
            f"  top visits: {visit_summary}\n"
            f"  policy: {policy_summary}\n"
        )
        self.history_box.config(state="normal")
        self.history_box.insert("end", line)
        self.history_box.see("end")
        self.history_box.config(state="disabled")

    def _set_history(self, text: str):
        self.history_box.config(state="normal")
        self.history_box.delete("1.0", "end")
        self.history_box.insert("1.0", text)
        self.history_box.config(state="disabled")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Connect4 model self-play.")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint file to load")
    parser.add_argument("--simulations", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--delay-ms", type=int, default=900, help="Delay between autoplay moves")
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    viewer = SelfPlayViewer(
        root,
        checkpoint_path=args.checkpoint or None,
        simulations=args.simulations,
        delay_ms=args.delay_ms,
    )
    root.resizable(False, False)
    root.mainloop()


if __name__ == "__main__":
    main()
