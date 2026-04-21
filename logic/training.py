import random

"""Unified training loop for all TD agents."""
def train(model, reset, update, num_games=1000, opponent=None, board_size=None):
    """Train via self-play or against a given opponent."""
    from game import Game

    bs = board_size or model.board_size
    model.training = True
    self_play = opponent is None

    for i in range(num_games):
        game = Game(size=bs)
        reset()  # reset prev_cache and prev_value (and traces for TD(lambda))

        # Randomly assign sides
        if self_play:
            td_player = 0  # plays both sides
        else:
            td_player = random.choice([1, 2])

        while not game.is_over():
            current = game.current_player
            if self_play or current == td_player:
                move = model.choose_move(game)
            else:
                move = opponent.choose_move(game)
            if move is None:
                break
            game.make_move(move[0], move[1])

        # Final update
        if model._prev_cache is not None:
            if game.winner == 0:
                target = 0.5
            elif self_play:
                target = 0.0  # last prev was the loser's perspective
            else:
                # If TD agent won
                target = 1.0 if game.winner == td_player else 0.0
            td_error = target - model._prev_value
            update(td_error)  # update model weights

        reset()  # reset prev_cache and prev_value (and traces for TD(lambda))

        if (i + 1) % 100 == 0:
            print(f"  Training game {i + 1}/{num_games}")
    
    model.training = False
