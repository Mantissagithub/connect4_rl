#in this file we are gonna make the game self playable

import numpy as np
import random
import torch

#getting te game engine components -> initialize board, make_move, chec_winner, is_terminal, is_draw, get_state_tensor, get_valid_moves
from game_engine_components.intialize_board import initialize_board
from game_engine_components.make_move import make_move
from game_engine_components.check_winner import check_winner
from game_engine_components.is_terminal import is_the_end
from game_engine_components.is_draw import is_draw
from game_engine_components.get_state_tensor import convert_into_tensor
from game_engine_components.get_valid_moves import get_valid_moves
from game_engine_components.copy_game import deep_copy

#neural network components -> forward pass includes everything, calculate loss, update weights, save model, load model
from neural_network_components.forward_pass import forward_pass
from neural_network_components.calculate_loss import calculate_loss
from neural_network_components.update_weights import update_weights
from neural_network_components.save_model import save_model
from neural_network_components.load_model import load_model

#MCTS components -> create_node, expand_node, select_child, backpropagate, run_mcts, get_action_visits
from mcts_components.create_node import create_root_node
from mcts_components.expand_node import expand_node
from mcts_components.select_child import select_child
from mcts_components.backpropogate_value import backpropagate
from mcts_components.run_simulation import run_simulation
from mcts_components.get_action_visits import get_action_visits


