from __future__ import annotations
import collections
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


# ============================== MOVE ENCODING ==============================
_QUEEN_DIRS    = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
_KNIGHT_DELTAS = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]
_UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
_UNDERPROMO_DIRS   = [-1, 0, 1]


def _build_move_index():
    move_to_idx = {}

    for dir_i, (dr, df) in enumerate(_QUEEN_DIRS):
        for dist in range(1, 8):
            move_type = dir_i * 7 + (dist - 1)
            for from_sq in chess.SQUARES:
                r0 = chess.square_rank(from_sq)
                f0 = chess.square_file(from_sq)
                r1, f1 = r0 + dr * dist, f0 + df * dist
                if 0 <= r1 < 8 and 0 <= f1 < 8:
                    to_sq = chess.square(f1, r1)
                    move_to_idx[(from_sq, to_sq, None)] = from_sq * 73 + move_type

    for k_i, (dr, df) in enumerate(_KNIGHT_DELTAS):
        move_type = 56 + k_i
        for from_sq in chess.SQUARES:
            r0 = chess.square_rank(from_sq)
            f0 = chess.square_file(from_sq)
            r1, f1 = r0 + dr, f0 + df
            if 0 <= r1 < 8 and 0 <= f1 < 8:
                to_sq = chess.square(f1, r1)
                move_to_idx[(from_sq, to_sq, None)] = from_sq * 73 + move_type

    for p_i, piece in enumerate(_UNDERPROMO_PIECES):
        for d_i, df in enumerate(_UNDERPROMO_DIRS):
            move_type = 64 + p_i * 3 + d_i
            for from_sq in chess.SQUARES:
                if chess.square_rank(from_sq) != 6:
                    continue
                f0 = chess.square_file(from_sq)
                f1 = f0 + df
                if 0 <= f1 < 8:
                    to_sq = chess.square(f1, 7)
                    move_to_idx[(from_sq, to_sq, piece)] = from_sq * 73 + move_type

    return move_to_idx


MOVE_TO_IDX = _build_move_index()
ACTION_SIZE  = 4672   # 64 * 73


def move_to_index(move: chess.Move, flip: bool = False) -> int:
    promo = move.promotion
    if promo == chess.QUEEN:
        promo = None
    fs = move.from_square
    ts = move.to_square
    if flip:
        fs = chess.square_mirror(fs)
        ts = chess.square_mirror(ts)
    return MOVE_TO_IDX.get((fs, ts, promo), -1)


# ============================== BOARD ENCODER ==============================
# BUG FIX: dùng chess.square_rank / chess.square_file thay vì divmod(sq, 8).
# divmod cho kết quả sai orientation — rank = sq // 8 không map đúng với
# chess.square_rank vì python-chess dùng little-endian rank mapping.
# Notebook SL pipeline đã dùng cách đúng; sp.py phải nhất quán.

PIECE_TO_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}


def _encode_single_board(board: chess.Board, flip: bool) -> np.ndarray:
    """Encode one position into 14 planes (12 piece + 2 repetition placeholders)."""
    planes = np.zeros((8, 8, 14), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        rank = chess.square_rank(sq)   # FIX: square_rank, not divmod
        file = chess.square_file(sq)   # FIX: square_file, not divmod
        if flip:
            rank  = 7 - rank
            color = not piece.color
        else:
            color = piece.color
        plane = PIECE_TO_PLANE[(piece.piece_type, color)]
        planes[rank, file, plane] = 1.0
    return planes


def encode_board_history(
    boards: list,           # [current, t-1, t-2, ...] newest first, up to 8
    current_turn: chess.Color,
) -> np.ndarray:
    """
    Encode board history → (119, 8, 8) float32, channel-first for PyTorch.
    Identical to board_history_to_planes in the SL notebook.
    """
    flip    = (current_turn == chess.BLACK)
    history = np.zeros((8, 8, 112), dtype=np.float32)
    for i, b in enumerate(boards[:8]):
        enc = _encode_single_board(b, flip)       # (8, 8, 14)
        history[:, :, i * 14:(i + 1) * 14] = enc

    board = boards[0]
    meta  = np.zeros((8, 8, 7), dtype=np.float32)
    meta[:, :, 0] = 1.0   # colour plane (always "my side" after flip)
    if not flip:
        meta[:, :, 1] = float(board.has_kingside_castling_rights(chess.WHITE))
        meta[:, :, 2] = float(board.has_queenside_castling_rights(chess.WHITE))
        meta[:, :, 3] = float(board.has_kingside_castling_rights(chess.BLACK))
        meta[:, :, 4] = float(board.has_queenside_castling_rights(chess.BLACK))
    else:
        meta[:, :, 1] = float(board.has_kingside_castling_rights(chess.BLACK))
        meta[:, :, 2] = float(board.has_queenside_castling_rights(chess.BLACK))
        meta[:, :, 3] = float(board.has_kingside_castling_rights(chess.WHITE))
        meta[:, :, 4] = float(board.has_queenside_castling_rights(chess.WHITE))
    meta[:, :, 5] = board.halfmove_clock / 100.0
    meta[:, :, 6] = board.fullmove_number / 100.0

    planes = np.concatenate([history, meta], axis=-1)   # (8, 8, 119)
    return planes.transpose(2, 0, 1).astype(np.float32)  # (119, 8, 8)


def encode_board(board: chess.Board) -> np.ndarray:
    """Single-board encode (no history). Used in MCTS leaf evaluation."""
    return encode_board_history([board], board.turn)


# ============================== NETWORK ==============================
class SEBlock(nn.Module):
    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Linear(channels, channels // ratio)
        self.fc2  = nn.Linear(channels // ratio, channels)

    def forward(self, x):
        se = self.pool(x).flatten(1)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se.view(-1, x.size(1), 1, 1)


class ResBlock(nn.Module):
    def __init__(self, filters: int, se_ratio: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters)
        self.se    = SEBlock(filters, se_ratio)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + x)


class AZNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        filters = 256
        n_res   = 15
        self.stem_conv = nn.Conv2d(119, filters, 3, padding=1, bias=False)
        self.stem_bn   = nn.BatchNorm2d(filters)
        self.tower     = nn.Sequential(*[ResBlock(filters) for _ in range(n_res)])

        self.pol_conv = nn.Conv2d(filters, 2, 1, bias=False)
        self.pol_bn   = nn.BatchNorm2d(2)
        self.pol_fc   = nn.Linear(2 * 8 * 8, ACTION_SIZE)

        self.val_conv = nn.Conv2d(filters, 1, 1, bias=False)
        self.val_bn   = nn.BatchNorm2d(1)
        self.val_fc1  = nn.Linear(8 * 8, 256)
        self.val_fc2  = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        x = self.tower(x)

        p = F.relu(self.pol_bn(self.pol_conv(x)))
        p = self.pol_fc(p.flatten(1))

        v = F.relu(self.val_bn(self.val_conv(x)))
        v = F.relu(self.val_fc1(v.flatten(1)))
        v = torch.tanh(self.val_fc2(v))
        return p, v


# ============================== CONFIG ==============================
@dataclass
class SelfPlayConfig:
    checkpoint_path: str
    output_path:     str

    device: str  = "cuda"
    amp:    bool = True

    boards: int = 16
    games:  int = 128

    simulations: int = 800

    cpuct:         float = 2.25
    fpu_reduction: float = 0.25

    temperature:          float = 1.0
    temperature_drop_ply: int   = 24

    max_moves: int = 512   

    dirichlet_alpha:    float = 0.30
    dirichlet_fraction: float = 0.25

    policy_topk: int = 32

    virtual_loss: int = 3

    # Replay buffer + training
    replay_capacity:   int   = 200_000
    min_replay_size:   int   = 2_048
    train_batch_size:  int   = 512
    train_epochs:      int   = 3
    lr:                float = 3e-4
    weight_decay:      float = 1e-4
    grad_clip:         float = 1.0
    train_every_games: int   = 8    

    log_every: int = 10


# ============================== REPLAY BUFFER ==============================
class ReplayBuffer:
    """
    Circular buffer storing (state, policy_target, value_target) tuples.
    All tensors kept on CPU; moved to device in training batch.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._states:  list = []
        self._pols:    list = []
        self._vals:    list = []
        self._ptr      = 0

    def __len__(self):
        return len(self._states)

    def push(self, state: np.ndarray, pi: np.ndarray, z: float):
        """state: (119,8,8) float32 | pi: (4672,) float32 | z: scalar"""
        if len(self._states) < self.capacity:
            self._states.append(None)
            self._pols.append(None)
            self._vals.append(None)
        self._states[self._ptr] = state
        self._pols[self._ptr]   = pi
        self._vals[self._ptr]   = z
        self._ptr = (self._ptr + 1) % self.capacity

    def push_game(self, game_data: list):
        """game_data: list of (state, pi, z) from one complete game."""
        for state, pi, z in game_data:
            self.push(state, pi, z)

    def sample(self, batch_size: int):
        indices = random.sample(range(len(self._states)), batch_size)
        states = np.stack([self._states[i] for i in indices])
        pols   = np.stack([self._pols[i]   for i in indices])
        vals   = np.array([self._vals[i]   for i in indices], dtype=np.float32)
        return (
            torch.from_numpy(states),
            torch.from_numpy(pols),
            torch.from_numpy(vals),
        )


# ============================== MCTS ==============================
class Node:
    __slots__ = (
        "board", "parent", "move", "prior",
        "children", "visits", "value_sum",
        "virtual_loss", "expanded",
    )

    def __init__(self, board: chess.Board, parent=None, move=None, prior: float = 0.0):
        self.board       = board
        self.parent      = parent
        self.move        = move
        self.prior       = prior
        self.children    = {}
        self.visits      = 0
        self.value_sum   = 0.0
        self.virtual_loss = 0
        self.expanded    = False

    @property
    def value(self):
        n = self.visits + self.virtual_loss
        return self.value_sum / n if n > 0 else 0.0

    def is_leaf(self):
        return not self.expanded


class BatchedMCTS:

    def __init__(self, cfg: SelfPlayConfig):
        self.cfg = cfg

    # ── UCB selection ──────────────────────────────────────────────────────
    def _best_child(self, node: Node) -> Node:
        sqrt_n     = math.sqrt(max(1, node.visits))
        best_score = -1e9
        best_child = next(iter(node.children.values()))
        parent_q   = node.value

        for child in node.children.values():
            q = -child.value if child.visits > 0 else parent_q - self.cfg.fpu_reduction
            u = (
                self.cfg.cpuct
                * child.prior
                * sqrt_n
                / (1 + child.visits + child.virtual_loss)
            )
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _select(self, root: Node):
        node = root
        path = [node]
        while node.expanded and node.children:
            node = self._best_child(node)
            node.virtual_loss += self.cfg.virtual_loss
            path.append(node)
        return path, node

    # ── Expand ─────────────────────────────────────────────────────────────
    def _expand(self, node: Node, policy: np.ndarray):
        legal_moves = list(node.board.legal_moves)
        if not legal_moves:
            node.expanded = True
            return

        flip   = (node.board.turn == chess.BLACK)
        scored = []
        for mv in legal_moves:
            idx = move_to_index(mv, flip)
            p   = float(policy[idx]) if 0 <= idx < ACTION_SIZE else 1e-8
            scored.append((p, mv))

        scored.sort(reverse=True, key=lambda x: x[0])
        scored = scored[: self.cfg.policy_topk]

        total = sum(p for p, _ in scored) or 1.0
        for p, mv in scored:
            b = node.board.copy()
            b.push(mv)
            node.children[mv] = Node(
                board=b, parent=node, move=mv, prior=p / total
            )

        node.expanded = True

    def _terminal_value(self, board: chess.Board) -> float:
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            return 0.0
        # From the perspective of the player whose turn it is *at this node*
        # (backprop will flip the sign for the parent).
        return 1.0 if outcome.winner == board.turn else -1.0

    def _backprop(self, path, value: float):
        for node in reversed(path):
            node.virtual_loss = max(0, node.virtual_loss - self.cfg.virtual_loss)
            node.visits      += 1
            node.value_sum   += value
            value             = -value

    # ── Dirichlet noise on root ────────────────────────────────────────────
    def _add_noise(self, root: Node):
        if not root.children:
            return
        noise = np.random.dirichlet(
            [self.cfg.dirichlet_alpha] * len(root.children)
        )
        frac = self.cfg.dirichlet_fraction
        for child, eta in zip(root.children.values(), noise):
            child.prior = child.prior * (1 - frac) + eta * frac

    # ── Root initialisation using the network ─────────────────────────────
    # BUG FIX: root was previously expanded with uniform ones(), ignoring the
    # network entirely.  Now we query the model to get real prior probabilities
    # before the first simulation, exactly as every other leaf node does.
    def init_root(self, root: Node, model, device, amp: bool):
        """Expand root using the network (not uniform noise)."""
        if root.expanded:
            return

        autocast_device = device.type if device.type in ("cuda", "cpu") else "cuda"
        state  = encode_board(root.board)
        tensor = torch.from_numpy(state[np.newaxis]).to(device)

        with torch.no_grad(), torch.autocast(device_type=autocast_device, enabled=amp):
            logits, _ = model(tensor)
            policy    = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        self._expand(root, policy)

    # ── Batched simulation loop ────────────────────────────────────────────
    def run_simulations(self, roots, model, device, amp: bool):
        autocast_device = device.type if device.type in ("cuda", "cpu") else "cuda"

        for _ in range(self.cfg.simulations):
            leaves = []
            paths  = []

            for root in roots:
                path, leaf = self._select(root)
                paths.append(path)
                leaves.append(leaf)

            batch_states  = []
            batch_indices = []
            values        = [0.0] * len(leaves)

            for i, leaf in enumerate(leaves):
                if leaf.board.is_game_over():
                    values[i] = self._terminal_value(leaf.board)
                else:
                    batch_states.append(encode_board(leaf.board))
                    batch_indices.append(i)

            if batch_states:
                tensor = torch.from_numpy(np.stack(batch_states)).to(device)
                with torch.no_grad(), torch.autocast(device_type=autocast_device, enabled=amp):
                    logits, vals = model(tensor)
                    probs = F.softmax(logits, dim=-1).cpu().numpy()
                    vals  = vals.squeeze(1).cpu().numpy()

                for j, idx in enumerate(batch_indices):
                    self._expand(leaves[idx], probs[j])
                    values[idx] = float(vals[j])

            for path, value in zip(paths, values):
                self._backprop(path, value)

    # ── Visit-count policy π ──────────────────────────────────────────────
    def visit_policy(self, root: Node, temperature: float) -> np.ndarray:
        pi = np.zeros(ACTION_SIZE, dtype=np.float32)
        if not root.children:
            return pi

        moves  = list(root.children.keys())
        visits = np.array([c.visits for c in root.children.values()], dtype=np.float32)
        flip   = (root.board.turn == chess.BLACK)

        if temperature <= 1e-4:
            best = int(np.argmax(visits))
            idx  = move_to_index(moves[best], flip)
            if idx >= 0:
                pi[idx] = 1.0
            return pi

        visits = visits ** (1.0 / temperature)
        visits /= visits.sum()

        for mv, p in zip(moves, visits):
            idx = move_to_index(mv, flip)
            if idx >= 0:
                pi[idx] = p

        return pi


# ============================== TRAINING ==============================
def train_on_replay(
    model: AZNetwork,
    replay: ReplayBuffer,
    cfg: SelfPlayConfig,
    device: torch.device,
) -> dict:
    """
    Run cfg.train_epochs passes over randomly sampled batches from the replay
    buffer.  Returns a dict with averaged loss metrics.
    """
    if len(replay) < cfg.min_replay_size:
        return {}

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    autocast_device = device.type if device.type in ("cuda", "cpu") else "cuda"

    steps_per_epoch = max(1, len(replay) // cfg.train_batch_size)
    total_pol = total_val = total_n = 0.0

    for _ in range(cfg.train_epochs):
        for _ in range(steps_per_epoch):
            states, pi_targets, z_targets = replay.sample(cfg.train_batch_size)
            states     = states.to(device, non_blocking=True)
            pi_targets = pi_targets.to(device, non_blocking=True)
            z_targets  = z_targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=autocast_device, enabled=(device.type == "cuda")):
                logits, values = model(states)

                # Policy loss — cross-entropy with visit-count soft target
                fill_val = torch.finfo(logits.dtype).min
                # No legal-mask during self-play training (pi already over legal moves)
                log_prob = F.log_softmax(logits, dim=-1)
                p_loss   = -(pi_targets * log_prob).sum(dim=-1).mean()

                # Value loss — MSE
                v_loss = F.mse_loss(values.squeeze(1), z_targets)

                loss = p_loss + v_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_pol += p_loss.item()
            total_val += v_loss.item()
            total_n   += 1

    model.eval()
    if total_n == 0:
        return {}
    return {
        "pol_loss": total_pol / total_n,
        "val_loss": total_val / total_n,
    }


# ============================== MODEL LOADING ==============================
def load_model(checkpoint_path: str, device: torch.device) -> AZNetwork:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        ckpt.get("model")
        or ckpt.get("state_dict")
        or ckpt.get("model_state_dict")
        or ckpt
    )
    new_sd = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }
    model = AZNetwork().to(device)
    model.load_state_dict(new_sd, strict=False)
    model.eval()
    return model


# ============================== INTERACTIVE MENU ==============================
def interactive_menu():
    print("\n" + "=" * 70)
    print("   PHELA ALPHAZERO — PARALLEL SELF-PLAY TRAINER")
    print("=" * 70)

    model_path = input("\nModel path: ").strip()
    while not Path(model_path).exists():
        print("File not found!")
        model_path = input("Enter model path again: ").strip()

    output_dir  = input("\nOutput directory (default: checkpoints): ").strip() or "checkpoints"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    default_name = f"selfplay_{time.strftime('%Y%m%d_%H%M')}.pt"
    name = input(f"Output filename (default: {default_name}): ").strip() or default_name
    if not name.endswith(".pt"):
        name += ".pt"
    output_path = str(Path(output_dir) / name)

    boards = int(input("\nParallel boards (default 16): ") or 16)
    games  = int(input("Games to generate (default 128): ") or 128)
    sims   = int(input("MCTS simulations/move (default 300): ") or 300)

    print(f"\n✅ Ready!")
    print(f"   Model : {model_path}")
    print(f"   Save  : {output_path}")
    print(f"   Boards: {boards} | Games: {games} | Sims: {sims}")

    return model_path, output_path, boards, games, sims


# ============================== MAIN ==============================
def main():
    from ui import AlphaZeroGUI
    app = AlphaZeroGUI()
    app.run()


if __name__ == "__main__":
    main()