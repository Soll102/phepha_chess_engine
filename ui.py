import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any
import collections

import chess
from PIL import Image, ImageDraw, ImageFont, ImageTk
import torch
import numpy as np

from main import (
    AZNetwork, move_to_index, ACTION_SIZE,
    SelfPlayConfig, Node, BatchedMCTS,
    ReplayBuffer, load_model, train_on_replay,
    encode_board,
)


LIGHT_SQUARE  = "#f0d9b5"
DARK_SQUARE   = "#b58863"
LAST_MOVE     = "#f6f669"
CHECK_SQUARE  = "#e05a47"
WHITE_PIECE   = "#f8f8f8"
BLACK_PIECE   = "#202020"
PIECE_OUTLINE = "#111111"


class AlphaZeroGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.configure(bg="#e9e9e9")
        self.root.title("Phela AlphaZero Trainer")
        self.root.geometry("1500x940")
        self.root.minsize(1100, 760)

        self.model: AZNetwork | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_training   = False
        self.training_thread: threading.Thread | None = None
        self.training_options: dict[str, Any] = {}

        self.board_labels:     list[tuple[tk.Frame, tk.Label]] = []
        self.board_photos:     list[ImageTk.PhotoImage] = []
        self.board_font_cache: dict[int, ImageFont.ImageFont] = {}
        self.cached_board_size = 320

        self.setup_ui()

    # ── UI setup ──────────────────────────────────────────────────────────
    def setup_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=15)
        left.grid(row=0, column=0, sticky="ns")

        ttk.Label(left, text="Phela AlphaZero", font=("Segoe UI", 18, "bold")).pack(pady=(0, 14))

        ttk.Label(left, text="Model", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.model_var = tk.StringVar(value=self.default_model_path())
        model_frame = ttk.Frame(left)
        model_frame.pack(fill=tk.X, pady=2)
        
        ttk.Entry(model_frame, textvariable=self.model_var, width=36).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(model_frame, text="Locate", command=self.browse_model).pack(
            side=tk.RIGHT, padx=(5, 0)
        )
        ttk.Button(left, text="Reload Model", command=self.reload_model).pack(
            pady=(6, 18), fill=tk.X
        )

        ttk.Label(left, text="Configuration", font=("Segoe UI", 12, "bold")).pack(
            anchor="w", pady=(0, 8)
        )
        self.boards_var         = tk.IntVar(value=8)
        self.games_var          = tk.IntVar(value=8)
        self.sims_var           = tk.IntVar(value=200)
        self.visible_boards_var = tk.IntVar(value=8)
        self.max_moves_var      = tk.IntVar(value=100)
        self.topk_var           = tk.IntVar(value=25)
        self.cpuct_var          = tk.DoubleVar(value=1.5)

        # Integer spinboxes
        for text, var, lo, hi in [
            ("Parallel Boards",   self.boards_var,         1,  64),
            ("Visible Boards",    self.visible_boards_var, 1,  32),
            ("Games to Generate", self.games_var,          1,  100_000),
            ("MCTS Simulations",  self.sims_var,           1,  2_000),
            ("Max Moves / Game",  self.max_moves_var,      20, 1_000),
            ("Policy Top-K",      self.topk_var,           1,  4_672),
        ]:
            ttk.Label(left, text=text).pack(anchor="w")
            ttk.Spinbox(left, from_=lo, to=hi, textvariable=var, width=15).pack(
                pady=(2, 8), anchor="w"
            )

        # PUCT / cpuct — float entry with label
        ttk.Label(left, text="PUCT (cpuct)").pack(anchor="w")
        ttk.Spinbox(
            left, from_=0.1, to=10.0, increment=0.05,
            textvariable=self.cpuct_var, width=15, format="%.2f",
        ).pack(pady=(2, 8), anchor="w")

        ttk.Label(left, text="Save folder (drive)", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", pady=(14, 2)
        )
        self.save_dir_var = tk.StringVar(value=str(Path("checkpoint").absolute()))
        save_frame = ttk.Frame(left)
        save_frame.pack(fill=tk.X, pady=2)
        ttk.Entry(save_frame, textvariable=self.save_dir_var, width=36).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(save_frame, text="Browse", command=self.browse_save_dir).pack(
            side=tk.RIGHT, padx=(5, 0)
        )

        ttk.Button(left, text="Start Self-Play", command=self.start_training).pack(
            pady=(18, 6), fill=tk.X
        )
        ttk.Button(left, text="Stop Training", command=self.stop_training).pack(
            pady=5, fill=tk.X
        )

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(
            left, textvariable=self.status_var, foreground="#1f5fbf", wraplength=260
        ).pack(pady=(14, 6), fill=tk.X)

        self.info_var = tk.StringVar(value="Games 0/0 | Active 0 | Move 0")
        ttk.Label(left, textvariable=self.info_var, font=("Segoe UI", 10)).pack(
            anchor="w", pady=(12, 0)
        )

        self.match_var = tk.StringVar(value="Score W-D-B: 0-0-0")
        ttk.Label(left, textvariable=self.match_var, font=("Segoe UI", 10)).pack(
            anchor="w", pady=(4, 0)
        )

        self.loss_var = tk.StringVar(value="pol_loss=— | val_loss=—")
        ttk.Label(left, textvariable=self.loss_var, font=("Segoe UI", 9),
                  foreground="#555").pack(anchor="w", pady=(4, 0))

        self.replay_var = tk.StringVar(value="Replay: 0 positions")
        ttk.Label(left, textvariable=self.replay_var, font=("Segoe UI", 9),
                  foreground="#555").pack(anchor="w", pady=(2, 0))

        right = ttk.Frame(self.root, padding=(10, 10, 10, 10))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        tk.Label(
            right, text="Live Parallel Boards",
            font=("Segoe UI", 20, "bold"), bg="#e9e9e9", anchor="w",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self.canvas = tk.Canvas(right, bg="#e9e9e9", highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(right, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.grid(row=1, column=1, sticky="ns")
        self.canvas.grid(row=1, column=0, sticky="nsew")

        self.board_frame  = tk.Frame(self.canvas, bg="#e9e9e9")
        self.board_window = self.canvas.create_window(
            (0, 0), window=self.board_frame, anchor="nw"
        )
        self.board_frame.bind("<Configure>", self.on_board_frame_configure)
        self.canvas.bind("<Configure>",      self.on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        self.update_live_boards([(chess.Board().fen(), "Board 1", "Move 0", None)])

    # ── Helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def default_model_path() -> str:
        for p in [
            Path("D:/Downloads/epoch_4.pt"),
            Path("epoch_4.pt"),
        ]:
            if p.exists():
                return str(p)
        return ""

    def on_board_frame_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ── Model loading ─────────────────────────────────────────────────────
    def browse_model(self):
        path = filedialog.askopenfilename(
            filetypes=[("PyTorch Model", "*.pt *.pth"), ("All Files", "*.*")]
        )
        if path:
            self.model_var.set(path)
            self.reload_model()

    def reload_model(self):
        if not self.model_var.get():
            messagebox.showwarning("Warning", "Select a model path first.")
            return
        try:
            self._load_model_from_path(self.model_var.get())
            messagebox.showinfo("Success", "Model loaded successfully.")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def browse_save_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.save_dir_var.set(path)

    def _load_model_from_path(self, path: str):
        self.model = load_model(path, self.device)
        return self.model


    # ── Board rendering ───────────────────────────────────────────────────
    def board_size(self, count: int) -> int:
        width = max(self.canvas.winfo_width(), 1200)
        cols  = 2 if count <= 4 else (3 if count <= 9 else 4)
        size  = max(260, min(420, (width - 40 * cols) // cols))
        if abs(size - self.cached_board_size) > 40:
            self.cached_board_size = size
        return self.cached_board_size

    def get_board_font(self, tile: int):
        size = max(18, int(tile * 0.68))
        if size in self.board_font_cache:
            return self.board_font_cache[size]
        for name in ("seguisym.ttf", "arialbd.ttf", "arial.ttf"):
            try:
                font = ImageFont.truetype(name, size)
                self.board_font_cache[size] = font # pyright: ignore[reportArgumentType]
                return font
            except Exception:
                pass
        font = ImageFont.load_default()
        self.board_font_cache[size] = font # pyright: ignore[reportArgumentType]
        return font

    def render_board(
        self,
        board: chess.Board,
        size: int,
        title: str,
        subtitle: str,
        last_move: chess.Move | None,
    ) -> Image.Image:
        header_h = 34
        footer_h = 24
        tile     = size // 8
        img      = Image.new("RGB", (size, size + header_h + footer_h), "#2b2b2b")
        draw     = ImageDraw.Draw(img)

        draw.rectangle((0, 0, size, header_h), fill="#202020")
        draw.text((10, 7), title,    fill="#ffffff", font=ImageFont.load_default())
        draw.text((size - 96, 7), subtitle, fill="#d7d7d7", font=ImageFont.load_default())

        check_sq     = board.king(board.turn) if board.is_check() else None
        last_squares = {last_move.from_square, last_move.to_square} if last_move else set()
        piece_font   = self.get_board_font(tile)

        for rank in range(8):
            for file in range(8):
                sq    = chess.square(file, 7 - rank)
                x0    = file * tile
                y0    = header_h + rank * tile
                color = LIGHT_SQUARE if (rank + file) % 2 == 0 else DARK_SQUARE
                if sq in last_squares:
                    color = LAST_MOVE
                if sq == check_sq:
                    color = CHECK_SQUARE
                draw.rectangle((x0, y0, x0 + tile, y0 + tile), fill=color)

                piece = board.piece_at(sq)
                if piece:
                    glyph = piece.unicode_symbol()
                    fill  = WHITE_PIECE if piece.color == chess.WHITE else BLACK_PIECE
                    bbox  = draw.textbbox((0, 0), glyph, font=piece_font)
                    tw    = bbox[2] - bbox[0]
                    th    = bbox[3] - bbox[1]
                    tx    = x0 + (tile - tw) / 2
                    ty    = y0 + (tile - th) / 2 - 4
                    if piece.color == chess.WHITE:
                        for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            draw.text((tx + ox, ty + oy), glyph, fill=PIECE_OUTLINE, font=piece_font)
                    draw.text((tx, ty), glyph, fill=fill, font=piece_font)

        footer_y = header_h + size
        draw.rectangle((0, footer_y, size, footer_y + footer_h), fill="#202020")
        turn = "White" if board.turn == chess.WHITE else "Black"
        draw.text((10, footer_y + 6), f"{turn} to move", fill="#d7d7d7", font=ImageFont.load_default())
        return img

    def update_live_boards(self, board_infos):
        count = len(board_infos)
        if count == 0:
            return
        size = self.board_size(count)
        cols = 2 if count <= 4 else (3 if count <= 9 else 4)

        while len(self.board_labels) < count:
            frame = tk.Frame(self.board_frame, bg="#e9e9e9")
            label = tk.Label(frame, bd=0, bg="#e9e9e9")
            label.pack()
            self.board_labels.append((frame, label))

        self.board_photos.clear()
        for i, (fen, title, subtitle, last_uci) in enumerate(board_infos):
            board     = chess.Board(fen)
            last_move = None
            if last_uci:
                try:
                    last_move = chess.Move.from_uci(last_uci)
                except Exception:
                    pass

            image = self.render_board(board, size, title, subtitle, last_move)
            photo = ImageTk.PhotoImage(image)
            self.board_photos.append(photo)

            frame, label = self.board_labels[i]
            frame.grid(row=i // cols, column=i % cols, padx=12, pady=12)
            label.configure(image=photo)

        for i in range(count, len(self.board_labels)):
            self.board_labels[i][0].grid_remove()

        for c in range(cols):
            self.board_frame.grid_columnconfigure(c, weight=0, minsize=size + 24)

        self.board_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        x_padding = max(0, (self.canvas.winfo_width() - cols * (size + 24)) // 2)
        self.canvas.coords(self.board_window, x_padding, 0)

    # ── Training control ──────────────────────────────────────────────────
    def start_training(self):
        if self.is_training:
            return
        if not self.model:
            if self.model_var.get():
                try:
                    self._load_model_from_path(self.model_var.get())
                except Exception as e:
                    messagebox.showerror("Load Error", str(e))
                    return
            else:
                messagebox.showwarning("Warning", "Please load a model first.")
                return

        self.is_training = True
        self.training_options = {
            "model_path":     self.model_var.get(),
            "boards":         max(1, self.boards_var.get()),
            "visible_boards": max(1, self.visible_boards_var.get()),
            "games":          max(1, self.games_var.get()),
            "simulations":    max(1, self.sims_var.get()),
            "max_moves":      max(20, self.max_moves_var.get()),
            "policy_topk":    max(1, self.topk_var.get()),
            "cpuct":          max(0.1, float(self.cpuct_var.get())),
        }
        self.status_var.set("Self-play running...")
        self.training_thread = threading.Thread(target=self.run_selfplay, daemon=True)
        self.training_thread.start()

    def stop_training(self):
        self.is_training = False
        self.status_var.set("Stopping after current step...")

    # ── Move selection ────────────────────────────────────────────────────
    # BUG FIX: was always argmax (greedy).  Now uses temperature sampling
    # for the first temperature_drop_ply moves, then switches to argmax.
    # This provides the exploration diversity required for self-play data
    # quality; pure greedy collapses all games to the same narrow opening.
    def choose_move(
        self,
        mcts: BatchedMCTS,
        root: Node,
        ply: int,
        cfg: SelfPlayConfig,
    ) -> chess.Move | None:
        temp        = cfg.temperature if ply < cfg.temperature_drop_ply else 0.0
        policy      = mcts.visit_policy(root, temp)
        legal_moves = list(root.board.legal_moves)
        if not legal_moves:
            return None

        flip   = (root.board.turn == chess.BLACK)
        scores = []
        moves  = []
        for mv in legal_moves:
            idx = move_to_index(mv, flip)
            p   = float(policy[idx]) if 0 <= idx < ACTION_SIZE else 0.0
            scores.append(p)
            moves.append(mv)

        scores_arr = np.array(scores, dtype=np.float32)

        if temp <= 1e-4:
            # Greedy after temperature_drop_ply
            return moves[int(np.argmax(scores_arr))]

        # Stochastic sampling proportional to visit-count policy
        total = scores_arr.sum()
        if total <= 0:
            # Fallback: uniform over legal moves
            return moves[np.random.randint(len(moves))]

        scores_arr /= total
        return moves[np.random.choice(len(moves), p=scores_arr)]

    # ── Progress reporting ────────────────────────────────────────────────
    def post_progress(
        self,
        games_done: int,
        cfg: SelfPlayConfig,
        active: list[dict],
        score: dict[str, int],
        replay_size: int = 0,
        last_losses: dict | None = None,
    ):
        visible_limit = self.training_options.get("visible_boards", 8)
        visible       = max(1, min(visible_limit, len(active)))
        infos         = []
        for entry in active[:visible]:
            board: chess.Board = entry["board"]
            last_uci = entry["last_move"].uci() if entry["last_move"] else None
            infos.append((
                board.fen(),
                f"Game {entry['game_no']}/{cfg.games}",
                f"Move {entry['ply']}",
                last_uci,
            ))

        max_ply    = max((e["ply"] for e in active), default=0)
        info_text  = f"Games {games_done}/{cfg.games} | Active {len(active)} | Move {max_ply}"
        match_text = f"Score W-D-B: {score['white']}-{score['draw']}-{score['black']}"
        replay_text = f"Replay: {replay_size:,} positions"

        loss_text = "pol_loss=— | val_loss=—"
        if last_losses:
            pl = last_losses.get("pol_loss", float("nan"))
            vl = last_losses.get("val_loss", float("nan"))
            loss_text = f"pol_loss={pl:.4f} | val_loss={vl:.4f}"

        self.root.after(30, lambda: self.update_live_boards(infos))
        self.root.after(0,  lambda: self.info_var.set(info_text))
        self.root.after(0,  lambda: self.match_var.set(match_text))
        self.root.after(0,  lambda: self.replay_var.set(replay_text))
        self.root.after(0,  lambda: self.loss_var.set(loss_text))

    # ── Main self-play loop ───────────────────────────────────────────────
    def run_selfplay(self):
        score      = {"white": 0, "draw": 0, "black": 0}
        games_done = 0
        last_losses: dict = {}

        try:
            cfg = SelfPlayConfig(
                checkpoint_path = str(self.training_options.get("model_path", "")),
                output_path     = f"selfplay_{time.strftime('%Y%m%d_%H%M')}.pt",
                boards          = self.training_options.get("boards", 8),
                games           = self.training_options.get("games", 64),
                simulations     = self.training_options.get("simulations", 200),
                max_moves       = self.training_options.get("max_moves", 512),
                policy_topk     = self.training_options.get("policy_topk", 32),
                cpuct           = self.training_options.get("cpuct", 2.25),
                device          = str(self.device),
                amp             = (self.device.type == "cuda"),
            )

            mcts   = BatchedMCTS(cfg)
            replay = ReplayBuffer(cfg.replay_capacity)

            next_game = 1
            while self.is_training and games_done < cfg.games:

                # ── Fill active batch ──────────────────────────────────────
                active: list[dict] = []
                while next_game <= cfg.games and len(active) < cfg.boards:
                    board = chess.Board()
                    root  = Node(board.copy())

                    # BUG FIX: expand root using the network, not uniform ones().
                    # Uniform priors make the first move's UCB completely random,
                    # causing the engine to miss strong opening moves entirely.
                    if self.model is not None:
                        mcts.init_root(root, self.model, self.device, cfg.amp)
                    mcts._add_noise(root)

                    active.append({
                        "game_no":    next_game,
                        "board":      board,
                        "root":       root,
                        "ply":        0,
                        "last_move":  None,
                        "history":    collections.deque([board.copy()], maxlen=8),
                        "game_data":  [],   # list of (state, pi, z) for this game
                    })
                    next_game += 1

                self.post_progress(games_done, cfg, active, score, len(replay), last_losses)

                # ── Play until batch is empty ──────────────────────────────
                while self.is_training and active:
                    roots = [e["root"] for e in active]
                    if self.model is None:
                        break

                    mcts.run_simulations(roots, self.model, self.device, cfg.amp)

                    still_active: list[dict] = []
                    for entry in active:
                        root:  Node        = entry["root"]
                        board: chess.Board = entry["board"]

                        # Compute π before choosing (temperature still > 0)
                        temp = cfg.temperature if entry["ply"] < cfg.temperature_drop_ply else 0.0
                        pi   = mcts.visit_policy(root, cfg.temperature)  # always temp=1 for target

                        # Record training sample (state before move, π, value TBD)
                        boards_list = list(entry["history"])
                        from main import encode_board_history
                        state = encode_board_history(boards_list, board.turn)
                        entry["game_data"].append([state, pi, None])   # z filled at game end

                        move = self.choose_move(mcts, root, entry["ply"], cfg)
                        if move is None:
                            games_done = self._finish_game(board, entry, score, games_done, replay)
                            continue

                        board.push(move)
                        entry["history"].appendleft(board.copy())
                        entry["ply"]      += 1
                        entry["last_move"] = move

                        # Tree reuse
                        if move in root.children:
                            new_root        = root.children[move]
                            new_root.parent = None
                            entry["root"]   = new_root
                        else:
                            new_root = Node(board.copy())
                            if self.model is not None:
                                mcts.init_root(new_root, self.model, self.device, cfg.amp)
                            entry["root"] = new_root

                        if board.is_game_over() or entry["ply"] >= cfg.max_moves:
                            games_done = self._finish_game(board, entry, score, games_done, replay)
                        else:
                            still_active.append(entry)

                    active = still_active
                    self.post_progress(games_done, cfg, active, score, len(replay), last_losses)

                # ── Periodic training pass ─────────────────────────────────
                if (
                    games_done > 0
                    and games_done % cfg.train_every_games == 0
                    and len(replay) >= cfg.min_replay_size
                    and self.model is not None
                ):
                    self.root.after(0, lambda: self.status_var.set("Training on replay buffer..."))
                    last_losses = train_on_replay(self.model, replay, cfg, self.device)

                    # Save checkpoint
                    out = Path(cfg.output_path)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    _model = self.model.module if hasattr(self.model, "module") else self.model
                    torch.save(
                        {
                            "model":    _model.state_dict(),
                            "games":    games_done,
                            "metrics":  last_losses,
                        },
                        out,
                    )
                    self.root.after(0, lambda: self.status_var.set("Self-play running..."))

            final_status = "Training finished" if games_done >= cfg.games else "Training stopped"
            self.root.after(0, lambda: self.status_var.set(final_status))

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", f"{e}\n\n{tb}"))
            self.root.after(0, lambda: self.status_var.set("Error"))
        finally:
            self.is_training = False

    # ── Game-end handler: fill value targets and push to replay ───────────
    def _finish_game(
        self,
        board: chess.Board,
        entry: dict,
        score: dict[str, int],
        games_done: int,
        replay: ReplayBuffer,
    ) -> int:
        outcome = board.outcome(claim_draw=True)

        if outcome is None or outcome.winner is None:
            score["draw"] += 1
            result_value = 0.0
        elif outcome.winner == chess.WHITE:
            score["white"] += 1
            result_value = 1.0
        else:
            score["black"] += 1
            result_value = -1.0

        # Fill value targets (retroactively, from each player's perspective).
        # We recorded side-to-move at each step; flip sign every half-move.
        game_data = entry["game_data"]
        n         = len(game_data)
        # result_value is from White's perspective (+1 White wins, -1 Black wins)
        # Convert to side-to-move perspective for each ply.
        # At ply 0 it's White's turn → +result_value; ply 1 Black → -result_value; etc.
        for ply_idx, sample in enumerate(game_data):
            z = result_value * (1 if ply_idx % 2 == 0 else -1)
            sample[2] = np.float32(z)
            replay.push(sample[0], sample[1], sample[2])

        return games_done + 1

    def run(self):
        self.root.mainloop()


