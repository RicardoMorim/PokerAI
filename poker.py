from pypokerengine.players import BasePokerPlayer
import numpy as np
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.engine.card import Card
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.engine.hand_evaluator import HandEvaluator
import random
import json

class PokerConstants:
    suits = ['H', 'D', 'C', 'S']  # Hearts, Diamonds, Clubs, Spades
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']  # 2-9, Jack, Queen, King, Ace
    deck = []
    for suit in suits:
        for rank in ranks:
            deck.append(suit + rank)

class ReinforcementLearningAgent(BasePokerPlayer):
    def __init__(self):
        self.q_table = {}  # Initialize Q-table as an empty dictionary
        self.alpha = 0.5  # Learning rate
        self.gamma = 0.6  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.nb_player = 2
        self.previous_action = None
        self.previous_state = None
        self.state = None
        self.valid_actions = None
        
    def receive_game_start_message(self, game_info):
        pass
    def receive_round_start_message(self, round_count, hole_card, seats):
        pass
    def receive_street_start_message(self, street, round_state):
        pass
    def receive_game_update_message(self, action, round_state):
        pass
    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    def update_q_table(self, reward):
        # Update Q-table based on the game result
        if self.previous_action:
            self.learn(self.previous_state, self.previous_action, reward, self.state, self.valid_actions)

    def draw_random_hand(self):
        deck = list(PokerConstants.deck) * 4
        random.shuffle(deck)
        return [Card.from_str(c) for c in deck[:2]]

    def declare_action(self, valid_actions, hole_card, round_state):
        state = self.get_state(hole_card, round_state)
        action = self.choose_action(state, valid_actions)
        return action
    
    def estimate_hole_card_win_rate(self, nb_simulation, nb_player, hole_card, community_card=None):
        if not community_card: community_card = []

        if not isinstance(hole_card[0], Card):
            hole_card = [Card.from_str(c) for c in hole_card]
        if community_card and not isinstance(community_card[0], Card):
            community_card = [Card.from_str(c) for c in community_card]

        # Calculate the win count by comparing your hole cards with the others
        win_count = sum([HandEvaluator.eval_hand(hole_card, community_card) >= HandEvaluator.eval_hand(self.draw_random_hand(), community_card) for _ in range(nb_simulation)])

        return 1.0 * win_count / nb_simulation

    def get_state(self, hole_card, round_state):
        # Calculate hand strength
        hand_strength = self.estimate_hole_card_win_rate(
            nb_simulation=1000,
            nb_player=self.nb_player,
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(round_state["community_card"]),
        )

        # Get number of players still in the game
        nb_players = len(
            [
                player
                for player in round_state["seats"]
                if player["state"] == "participating"
            ]
        )

        # Get current stage of the game
        game_stage = round_state["street"]

        return (hand_strength, nb_players, game_stage)

    def choose_action(self, state, valid_actions):
        state_str = json.dumps(state, sort_keys=True)
        valid_actions_str = [json.dumps(action, sort_keys=True) for action in valid_actions]
        q_values = {action_str: self.q_table.get((state_str, action_str), 0) for action_str in valid_actions_str}
        max_q_value = max(q_values.values())
        actions_with_max_q_value = [action_str for action_str, q_value in q_values.items() if q_value == max_q_value]
        r = json.loads(random.choice(actions_with_max_q_value))
        action = r["action"]
        amount = r["amount"]
        if action == "raise":
            # Choose a random amount between 'min' and 'max'
            amount = random.randint(amount["min"], amount["max"])
            
        self.previous_action = self.previous_action = (action, amount)
        self.previous_state = self.state
        self.state = state
        self.valid_actions = valid_actions

        return (action, amount)

    def learn(self, state, action, reward, next_state, valid_actions):
        old_value = self.q_table.get((state, action), 0)
        next_max = max([self.q_table.get((next_state, json.dumps(a, sort_keys=True)), 0) for a in valid_actions])

        new_value = (1 - self.alpha) * old_value + self.alpha * (
            reward + self.gamma * next_max
        )
        self.q_table[(state, action)] = new_value
