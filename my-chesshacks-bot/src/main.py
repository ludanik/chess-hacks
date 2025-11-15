from .utils import chess_manager, GameContext
from chess import Move
import random
import time

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("Maxlegrec/ChessBot", trust_remote_code=True)
device = "cpu"
model = model.to(device)

def get_move(pgn: str) -> str:
    # Your bot logic here
    # Receives board state as PGN
    # Returns move in UCI format
    pass

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Cooking move...")
    print(ctx.board.move_stack)
    time.sleep(0.1)

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")


    # my code
    # need to get fen from pgn
    fen = ctx.board.fen()
    probs = model.get_move_from_fen_no_thinking(fen, T=1, device=device, return_probs=True)
    top_moves = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    '''
    
    '''
    move_probs = top_moves
    print("my model")
    print(move_probs)
    print(probs)
    print(top_moves)
    ctx.logProbabilities(move_probs)

    move_weights = [random.random() for _ in legal_moves]
    total_weight = sum(move_weights)
    # Normalize so probabilities sum to 1
    move_probs = {
        Move.from_uci(move): weight
        for move, weight in top_moves
    }
    ctx.logProbabilities(move_probs)
    print("random weights")
    
    print(move_probs)
    print(move_weights)
    return Move.from_uci(top_moves[0][0])
    #return random.choices(legal_moves, weights=move_weights, k=1)[0]



@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
