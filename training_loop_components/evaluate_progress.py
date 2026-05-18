import random
import time

from game_engine_components.connect4_env import Connect4Env
from mcts_components.get_action_visits import get_action_visits
from mcts_components.run_simulation import run_simulation
from training_data_components.generate_self_play_game import play_self_play_game


def _choose_mcts_action(board, current_player, model, num_simulations):
    root_node = run_simulation(board, current_player, model, num_simulations=num_simulations)
    action_visits = get_action_visits(root_node)
    if not action_visits:
        return None
    return max(action_visits.items(), key=lambda item: item[1])[0]


def evaluate_progress(neural_net, baseline_models=None, num_evaluation_games=50, num_simulations=100, verbose=True):
    evaluation_start_time = time.time()
    results = {
        "evaluation_time": 0.0,
        "self_play_results": evaluate_self_play(neural_net, num_evaluation_games, num_simulations, verbose),
        "baseline_results": {},
        "overall_rating": 0.0,
    }

    if baseline_models:
        results["baseline_results"] = evaluate_against_baselines(
            neural_net, baseline_models, num_evaluation_games, num_simulations, verbose
        )
    else:
        results["baseline_results"]["random"] = evaluate_against_random(
            neural_net, num_evaluation_games, num_simulations, verbose
        )

    results["overall_rating"] = calculate_overall_rating(results)
    results["evaluation_time"] = time.time() - evaluation_start_time

    if verbose:
        print(f"Evaluation completed in {results['evaluation_time']:.2f} seconds")
        print(f"Overall rating: {results['overall_rating']:.2f}")
        print_evaluation_summary(results)

    return results


def evaluate_self_play(neural_net, num_games, num_simulations, verbose=True):
    outcomes = {"player1_wins": 0, "player2_wins": 0, "draws": 0}
    total_moves = 0

    for game_idx in range(num_games):
        try:
            _, metadata = play_self_play_game(neural_net, num_simulations=num_simulations, verbose=False)
            total_moves += metadata["num_moves"]
            if metadata["winner"] == 1:
                outcomes["player1_wins"] += 1
            elif metadata["winner"] == 2:
                outcomes["player2_wins"] += 1
            else:
                outcomes["draws"] += 1
        except Exception as exc:
            if verbose:
                print(f"Error in self-play evaluation game {game_idx + 1}: {exc}")

    games_played = sum(outcomes.values())
    return {
        "games_played": games_played,
        "avg_game_length": (total_moves / games_played) if games_played else 0.0,
        "total_moves": total_moves,
        "outcomes": outcomes,
        "win_rate_balance": (
            abs(outcomes["player1_wins"] - outcomes["player2_wins"]) / games_played if games_played else 1.0
        ),
    }


def evaluate_against_random(neural_net, num_games, num_simulations=100, verbose=True):
    wins = 0
    losses = 0
    draws = 0
    total_moves = 0

    for game_idx in range(num_games):
        try:
            env = Connect4Env()
            neural_net_player = random.choice([1, 2])
            move_count = 0

            while not env.is_terminal():
                legal_actions = env.legal_actions()
                if env.current_player == neural_net_player:
                    move = _choose_mcts_action(env.board, env.current_player, neural_net, num_simulations) 
                    if move not in legal_actions:
                        move = random.choice(legal_actions)
                else:
                    move = random.choice(legal_actions)
                _, _, _, legal = env.step(move)
                if not legal:
                    raise ValueError(f"Illegal move during random evaluation: {move}")
                move_count += 1

            winner = env.winner()
            if winner == neural_net_player:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1
            total_moves += move_count
        except Exception as exc:
            if verbose:
                print(f"Error in game {game_idx + 1} against random: {exc}")

    games_played = wins + losses + draws
    return {
        "games_played": games_played,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games_played if games_played else 0.0,
        "avg_moves_per_game": total_moves / games_played if games_played else 0.0,
    }


def evaluate_against_baselines(neural_net, baseline_models, num_games, num_simulations, verbose=True):
    results = {}
    for name, model in baseline_models.items():
        try:
            results[name] = evaluate_head_to_head(neural_net, model, num_games, num_simulations)
        except Exception as exc:
            if verbose:
                print(f"Error evaluating against {name}: {exc}")
            results[name] = {"error": str(exc)}
    return results


def evaluate_head_to_head(model1, model2, num_games, num_simulations):
    model1_wins = 0
    model2_wins = 0
    draws = 0

    for game_idx in range(num_games):
        env = Connect4Env()
        player_models = {1: model1, 2: model2} if game_idx % 2 == 0 else {1: model2, 2: model1}

        while not env.is_terminal():
            legal_actions = env.legal_actions()
            move = _choose_mcts_action(env.board, env.current_player, player_models[env.current_player], num_simulations)
            if move not in legal_actions:
                move = random.choice(legal_actions)
            _, _, _, legal = env.step(move)
            if not legal:
                raise ValueError(f"Illegal move during head-to-head evaluation: {move}")

        winner = env.winner()
        if winner == 0:
            draws += 1
        elif player_models[winner] is model1:
            model1_wins += 1
        else:
            model2_wins += 1

    games_played = model1_wins + model2_wins + draws
    return {
        "games_played": games_played,
        "model1_wins": model1_wins,
        "model2_wins": model2_wins,
        "draws": draws,
        "model1_win_rate": model1_wins / games_played if games_played else 0.0,
    }


def calculate_overall_rating(results):
    score = 0.0
    random_results = results.get("baseline_results", {}).get("random")
    if random_results:
        score += 100.0 * random_results.get("win_rate", 0.0)

    self_play = results.get("self_play_results", {})
    score += 20.0 * (1.0 - self_play.get("win_rate_balance", 1.0))
    return score


def print_evaluation_summary(results):
    self_play = results.get("self_play_results", {})
    print(f"Self-play games: {self_play.get('games_played', 0)}")
    print(f"Average game length: {self_play.get('avg_game_length', 0):.2f}")
    random_results = results.get("baseline_results", {}).get("random")
    if random_results:
        print(f"Random baseline win rate: {random_results.get('win_rate', 0.0):.2%}")
