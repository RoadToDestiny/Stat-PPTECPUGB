"""
Generates synthetic chess games dataset adhering to the Data Contract schema.
Used for pipeline verification and smoke testing before real PGN parsing is ready.
"""

import os
import numpy as np
import pandas as pd


def generate_synthetic_chess_data(
    num_games: int = 50,
    moves_per_game_range: tuple = (25, 45),
    error_threshold_cp: float = 200.0,
    random_seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    records = []

    for game_idx in range(1, num_games + 1):
        game_id = f"lichess_game_{game_idx:04d}"
        num_moves = rng.integers(moves_per_game_range[0], moves_per_game_range[1] + 1)
        total_plies = num_moves * 2

        white_rating = float(rng.integers(1400, 2200))
        black_rating = float(rng.integers(1400, 2200))
        base_time = 600.0  # 10 min
        inc_time = 0.0

        current_eval_white = 0.0
        prev_eval_white = 0.0

        for ply in range(1, total_plies + 1):
            move_number = (ply + 1) // 2
            is_white = (ply % 2 != 0)
            color = "w" if is_white else "b"

            player_rating = white_rating if is_white else black_rating
            opponent_rating = black_rating if is_white else white_rating
            rating_diff = player_rating - opponent_rating

            # Stage of game
            if move_number <= 10:
                game_phase = 1
            elif move_number <= 30:
                game_phase = 2
            else:
                game_phase = 3

            # Board counts
            pawn_count = max(0, 16 - int((ply / total_plies) * 10) + rng.integers(-1, 2))
            pawn_count = min(16, max(0, pawn_count))
            piece_count = max(2, 32 - int((ply / total_plies) * 20) + rng.integers(-2, 2))
            piece_count = min(32, max(2, piece_count))

            material_balance = float(rng.integers(-5, 6))
            legal_checks_count = int(rng.choice([0, 1, 2], p=[0.75, 0.20, 0.05]))
            legal_captures_count = int(rng.integers(0, 7))
            attacked_pieces_count = int(rng.integers(0, 5))

            # Evaluations
            # eval_before with respect to player who moves
            eval_before_player = current_eval_white if is_white else -current_eval_white
            eval_diff_prev_move = current_eval_white - prev_eval_white

            # Error simulation (probability depends on rating and tactical complexity)
            error_prob = 0.08 + (0.04 if attacked_pieces_count > 2 else 0.0) - (player_rating - 1500) * 0.00005
            error_prob = max(0.02, min(0.30, error_prob))

            is_error = 1 if rng.random() < error_prob else 0

            if is_error:
                eval_loss = float(rng.uniform(error_threshold_cp, 600.0))
            else:
                eval_loss = float(rng.uniform(0.0, error_threshold_cp - 10.0))

            # Compute eval_after from white perspective
            if is_white:
                next_eval_white = current_eval_white - eval_loss
                eval_after_player = eval_before_player - eval_loss
            else:
                next_eval_white = current_eval_white + eval_loss
                eval_after_player = eval_before_player - eval_loss

            prev_eval_white = current_eval_white
            current_eval_white = np.clip(next_eval_white, -2500.0, 2500.0)

            records.append({
                "game_id": game_id,
                "ply": ply,
                "move_number": move_number,
                "color": color,
                "eval_before": round(eval_before_player, 1),
                "eval_diff_prev_move": round(eval_diff_prev_move, 1),
                "material_balance": material_balance,
                "piece_count": piece_count,
                "pawn_count": pawn_count,
                "legal_checks_count": legal_checks_count,
                "legal_captures_count": legal_captures_count,
                "attacked_pieces_count": attacked_pieces_count,
                "game_phase": game_phase,
                "player_rating": player_rating,
                "opponent_rating": opponent_rating,
                "rating_diff": rating_diff,
                "time_control_base": base_time,
                "time_control_inc": inc_time,
                "eval_after": round(eval_after_player, 1),
                "eval_loss": round(eval_loss, 1),
                "is_error": is_error,
                "is_mate_eval": False,
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df_dummy = generate_synthetic_chess_data(num_games=50)
    out_file = os.path.join(out_dir, "sample_dataset.csv")
    df_dummy.to_csv(out_file, index=False)
    print(f"Generated sample dataset with {len(df_dummy)} rows to {out_file}")
