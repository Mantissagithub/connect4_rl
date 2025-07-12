import time
import random
from typing import Dict, List, Tuple, Any

def evaluate_progress(neural_net, baseline_models=None, num_evaluation_games=50, num_simulations=100, verbose=True):
    from training_data_components.generate_self_play_game import generate_self_play_game
    from game_engine_components.check_winner import check_winner
    from game_engine_components.is_draw import is_draw
    
    evaluation_start_time = time.time()
    results = {
        'evaluation_time': 0.0,
        'self_play_results': {},
        'baseline_results': {},
        'overall_rating': 0.0
    }
    
    if verbose:
        print(f"Starting evaluation with {num_evaluation_games} games per opponent")
    
    #evaluting the self play perf
    self_play_results = evaluate_self_play(neural_net, num_evaluation_games, num_simulations, verbose)
    results['self_play_results'] = self_play_results
    
    #if baseline models are present, for my case its not there, maybe let this be here, if something comes in the future let me add it here for comparing thr perf
    if baseline_models:
        baseline_results = evaluate_against_baselines(
            neural_net, baseline_models, num_evaluation_games, num_simulations, verbose
        )
        results['baseline_results'] = baseline_results
    else:
        random_results = evaluate_against_random(neural_net, num_evaluation_games, verbose)
        results['baseline_results']['random'] = random_results
    
    overall_rating = calculate_overall_rating(results)
    results['overall_rating'] = overall_rating
    
    evaluation_time = time.time() - evaluation_start_time
    results['evaluation_time'] = evaluation_time
    
    if verbose:
        print(f"Evaluation completed in {evaluation_time:.2f} seconds")
        print(f"Overall rating: {overall_rating:.2f}")
        print_evaluation_summary(results)
    
    return results

def evaluate_self_play(neural_net, num_games, num_simulations, verbose=True):
    from training_data_components.generate_self_play_game import generate_self_play_game
    
    games_played = 0
    total_moves = 0
    total_game_length = 0
    game_outcomes = {'player1_wins': 0, 'player2_wins': 0, 'draws': 0}
    
    for game_idx in range(num_games):
        try:
            game_data = generate_self_play_game(
                neural_net=neural_net, 
                num_simulations=num_simulations, 
                verbose=False
            )
            
            if game_data:
                games_played += 1
                game_length = len(game_data)
                total_game_length += game_length
                total_moves += game_length
                
                if game_data:
                    final_state = game_data[-1][0] 
                    final_value = game_data[-1][2] 
                    
                    if abs(final_value) > 0.8: 
                        if final_value > 0:
                            game_outcomes['player1_wins'] += 1
                        else:
                            game_outcomes['player2_wins'] += 1
                    else:
                        game_outcomes['draws'] += 1
                        
        except Exception as e:
            if verbose:
                print(f"Error in self-play game {game_idx + 1}: {e}")
            continue
    
    if games_played > 0:
        avg_game_length = total_game_length / games_played
        win_rate_balance = abs(game_outcomes['player1_wins'] - game_outcomes['player2_wins']) / games_played
    else:
        avg_game_length = 0
        win_rate_balance = 1.0
    
    return {
        'games_played': games_played,
        'avg_game_length': avg_game_length,
        'total_moves': total_moves,
        'outcomes': game_outcomes,
        'win_rate_balance': win_rate_balance
    }

def evaluate_against_random(neural_net, num_games, verbose=True):
    """
    Evaluate neural network against random player
    """
    import random
    from mcts_components.run_simulation import run_simulation
    from mcts_components.get_action_visits import get_action_visits, get_action_probabilities
    from game_engine_components.intialize_board import initialize_board
    from game_engine_components.get_valid_moves import get_valid_moves
    from game_engine_components.make_move import make_move
    from game_engine_components.check_winner import check_winner
    from game_engine_components.is_draw import is_draw
    from game_engine_components.is_terminal import is_the_end
    from game_engine_components.copy_game import deep_copy
    
    wins = 0
    losses = 0
    draws = 0
    total_moves = 0
    
    for game_idx in range(num_games):
        try:
            board = initialize_board()
            current_player = 1
            moves_in_game = 0
            
            # Randomly decide if neural net goes first or second
            neural_net_player = random.choice([1, 2])
            
            while not is_the_end(board):
                valid_moves = get_valid_moves(board)
                if not valid_moves:
                    break
                
                if current_player == neural_net_player:
                    # Neural network move using MCTS
                    try:
                        # Make a copy of the board for MCTS
                        board_copy = deep_copy(board)
                        root_node = run_simulation(board_copy, current_player, neural_net, num_simulations=50)
                        
                        # Get action probabilities from MCTS
                        action_visits = get_action_visits(root_node)
                        if action_visits:
                            # Select move with highest visit count
                            move = max(action_visits.items(), key=lambda x: x[1])[0]
                            if move not in valid_moves:
                                move = random.choice(valid_moves)
                        else:
                            move = random.choice(valid_moves)
                    except Exception as e:
                        if verbose:
                            print(f"MCTS error in game {game_idx + 1}: {e}")
                        move = random.choice(valid_moves)
                else:
                    # Random move
                    move = random.choice(valid_moves)
                
                # Make the move (modifies board in place)
                success = make_move(board, move, current_player)
                if not success:
                    # If move failed, break the game
                    break
                    
                moves_in_game += 1
                current_player = 3 - current_player  # Switch between 1 and 2
            
            # Determine winner
            winner = check_winner(board)
            if winner == neural_net_player:
                wins += 1
            elif winner == 0 or is_draw(board):
                draws += 1
            else:
                losses += 1
            
            total_moves += moves_in_game
            
        except Exception as e:
            if verbose:
                print(f"Error in game {game_idx + 1} against random: {e}")
            continue
    
    games_played = wins + losses + draws
    win_rate = wins / games_played if games_played > 0 else 0.0
    
    return {
        'games_played': games_played,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'avg_moves_per_game': total_moves / games_played if games_played > 0 else 0
    }

def evaluate_against_baselines(neural_net, baseline_models, num_games, num_simulations, verbose=True):
    baseline_results = {}
    
    for baseline_name, baseline_model in baseline_models.items():
        if verbose:
            print(f"Evaluating against {baseline_name}...")
        
        try:
            results = evaluate_head_to_head(neural_net, baseline_model, num_games, num_simulations)
            baseline_results[baseline_name] = results
        except Exception as e:
            if verbose:
                print(f"Error evaluating against {baseline_name}: {e}")
            baseline_results[baseline_name] = {'error': str(e)}
    
    return baseline_results

def evaluate_head_to_head(model1, model2, num_games, num_simulations):
    """
    Play two models against each other
    """
    import random
    from mcts_components.run_simulation import run_simulation
    from mcts_components.get_action_visits import get_action_visits
    from game_engine_components.intialize_board import initialize_board
    from game_engine_components.get_valid_moves import get_valid_moves
    from game_engine_components.make_move import make_move
    from game_engine_components.check_winner import check_winner
    from game_engine_components.is_draw import is_draw
    from game_engine_components.is_terminal import is_the_end
    from game_engine_components.copy_game import deep_copy
    
    model1_wins = 0
    model2_wins = 0
    draws = 0
    
    for game_idx in range(num_games):
        try:
            board = initialize_board()
            current_player = 1
            
            # Alternate who goes first
            if game_idx % 2 == 0:
                player_models = {1: model1, 2: model2}
            else:
                player_models = {1: model2, 2: model1}
            
            while not is_the_end(board):
                valid_moves = get_valid_moves(board)
                if not valid_moves:
                    break
                
                try:
                    current_model = player_models[current_player]
                    board_copy = deep_copy(board)
                    root_node = run_simulation(board_copy, current_player, current_model, num_simulations=num_simulations//2)
                    
                    action_visits = get_action_visits(root_node)
                    if action_visits:
                        move = max(action_visits.items(), key=lambda x: x[1])[0]
                        if move not in valid_moves:
                            move = random.choice(valid_moves)
                    else:
                        move = random.choice(valid_moves)
                except:
                    move = random.choice(valid_moves)
                
                success = make_move(board, move, current_player)
                if not success:
                    break
                    
                current_player = 3 - current_player
            
            # Determine winner
            winner = check_winner(board)
            if winner == 0 or is_draw(board):
                draws += 1
            elif (game_idx % 2 == 0 and winner == 1) or (game_idx % 2 == 1 and winner == 2):
                model1_wins += 1
            else:
                model2_wins += 1
                
        except Exception as e:
            print(f"Error in head-to-head game {game_idx + 1}: {e}")
            continue
    
    games_played = model1_wins + model2_wins + draws
    win_rate = model1_wins / games_played if games_played > 0 else 0.0
    
    return {
        'games_played': games_played,
        'model1_wins': model1_wins,
        'model2_wins': model2_wins,
        'draws': draws,
        'win_rate': win_rate
    }

def calculate_overall_rating(results):
    rating = 0.0
    
    self_play = results.get('self_play_results', {})
    if 'win_rate_balance' in self_play:
        balance_score = 1.0 - self_play['win_rate_balance']  # Lower balance is better
        rating += balance_score * 30  # 30 points for balance
    
    baseline_results = results.get('baseline_results', {})
    
    if 'random' in baseline_results:
        random_win_rate = baseline_results['random'].get('win_rate', 0.0)
        rating += random_win_rate * 40  # 40 points for beating random
    
    baseline_wins = 0
    baseline_count = 0
    for baseline_name, baseline_result in baseline_results.items():
        if baseline_name != 'random' and 'win_rate' in baseline_result:
            baseline_wins += baseline_result['win_rate']
            baseline_count += 1
    
    if baseline_count > 0:
        avg_baseline_performance = baseline_wins / baseline_count
        rating += avg_baseline_performance * 30  # 30 points for baseline performance
    
    return min(rating, 100.0)  # Cap at 100

def print_evaluation_summary(results):
    """
    Print a summary of evaluation results
    """
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    self_play = results.get('self_play_results', {})
    if self_play:
        print(f"\nSelf-Play Performance:")
        print(f"  Games played: {self_play.get('games_played', 0)}")
        print(f"  Average game length: {self_play.get('avg_game_length', 0):.1f} moves")
        outcomes = self_play.get('outcomes', {})
        print(f"  Outcomes: P1: {outcomes.get('player1_wins', 0)}, "
              f"P2: {outcomes.get('player2_wins', 0)}, "
              f"Draws: {outcomes.get('draws', 0)}")
        print(f"  Win rate balance: {self_play.get('win_rate_balance', 0):.3f} (lower is better)")
    
    baseline_results = results.get('baseline_results', {})
    if baseline_results:
        print(f"\nBaseline Performance:")
        for baseline_name, baseline_result in baseline_results.items():
            if 'error' in baseline_result:
                print(f"  vs {baseline_name}: Error - {baseline_result['error']}")
            else:
                win_rate = baseline_result.get('win_rate', 0.0)
                games = baseline_result.get('games_played', 0)
                print(f"  vs {baseline_name}: {win_rate:.1%} win rate ({games} games)")
    
    print(f"\nOverall Rating: {results.get('overall_rating', 0):.1f}/100")
    print(f"Evaluation Time: {results.get('evaluation_time', 0):.2f} seconds")
    print("="*50 + "\n")
