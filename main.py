from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.game import setup_config, start_poker
from poker import ReinforcementLearningAgent
import pypokerengine.utils.visualize_utils as U
import random
import pickle

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __str__(self):
        return f'''
        _______
       |{self.rank:<2}    |
       |      |
       |  {self.suit}   |
       |      |
       |____{self.rank:>2}|
        '''

class Board:
    def __init__(self, cards):
        self.cards = cards

    def __str__(self):
        if not self.cards:
            return ''

        # Split each card's string representation into lines
        card_lines = [str(card).split('\n') for card in self.cards]

        # Get the maximum number of lines
        max_lines = max(len(lines) for lines in card_lines)

        # Pad the lists of lines with empty lines if necessary
        for lines in card_lines:
            lines.extend([''] * (max_lines - len(lines)))

        # Concatenate the corresponding lines of each card
        lines = ['   '.join(card_lines[j][i] for j in range(len(self.cards))) for i in range(max_lines)]

        # Join the lines into a single string
        return '\n'.join(lines)
    
class ConsolePlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        print(
            U.visualize_declare_action(valid_actions, hole_card, round_state, self.uuid)
        )
        print(f"\n\n\n\n SUMMARY \n\n\n\n")

        # Print the current stack of each player
        print('Player stacks:')
        for seat in round_state['seats']:
            print(f"{seat['name']}: {seat['stack']}")


        # Create Card objects for the community cards and print them
        community_cards = [Card(card[1:], card[0]) for card in round_state['community_card']]
        print('Community cards:')
        print(Board(community_cards))

        hole_cards = [Card(card[1:], card[0]) for card in hole_card]
        
        print('Your cards:')
        for card in hole_cards:
            print(card,)
        
        print(f"\nAvailable actions: ")
        for action in valid_actions:
            print(action)
        try:
            action, amount = self._receive_action_from_console(valid_actions)
        except:
            return self.declare_action(valid_actions, hole_card, round_state)
        return action, amount

    def receive_game_start_message(self, game_info):
        print(U.visualize_game_start(game_info, self.uuid))
        self._wait_until_input()

    def receive_round_start_message(self, round_count, hole_card, seats):
        print(U.visualize_round_start(round_count, hole_card, seats, self.uuid))
        self._wait_until_input()

    def receive_street_start_message(self, street, round_state):
        print(U.visualize_street_start(street, round_state, self.uuid))
        self._wait_until_input()

    def receive_game_update_message(self, new_action, round_state):
        print(U.visualize_game_update(new_action, round_state, self.uuid))
        self._wait_until_input()

    def receive_round_result_message(self, winners, hand_info, round_state):
        print(U.visualize_round_result(winners, hand_info, round_state, self.uuid))
        self._wait_until_input()

    def _wait_until_input(self):
        input("Enter some key to continue ...")

    def _receive_action_from_console(self, valid_actions):
        action = input("Enter action to declare >> ")
        amount = 0
        if action.lower() == "fold":
            amount = 0
        elif action.lower() == "call":
            # Find the dictionary in valid_actions where the action is "call"
            call_action = next((a for a in valid_actions if a["action"] == "call"), None)
            if call_action:
                amount = call_action["amount"]
        if action.lower() == "raise":
            amount = int(input("Enter raise amount >> "))
        return action.lower(), amount

def load_agents():
    try:
        with open('rl_agents.pkl', 'rb') as f:
            rl_agents = pickle.load(f)
    except FileNotFoundError:
        rl_agents = [ReinforcementLearningAgent() for _ in range(random.randint(3, 5))]
    return rl_agents

rl_agents = load_agents()

def train():
    # Set up the game configuration
    config = setup_config(max_round=10, initial_stack=10000, small_blind_amount=20)

    # Register the AI players
    for i, rl_agent in enumerate(rl_agents):
        config.register_player(name=f"RLAgent{i+1}", algorithm=rl_agent)

    # Start the poker game
    for i in range(100):
        print(f"Game {i}")
        game_result = start_poker(config, verbose=0)

        # Update each agent's Q-table
        for j, rl_agent in enumerate(rl_agents):
            reward = game_result["players"][j]["stack"]
            rl_agent.update_q_table(reward)
        
        # Determine the winner and print it
        stacks = [player["stack"] for player in game_result["players"]]
        winner = stacks.index(max(stacks))
        print(f"Winner of game {i}: RLAgent{winner+1}")

    # Save the trained agents
    with open('rl_agents.pkl', 'wb') as f:
        pickle.dump(rl_agents, f)

def game():
    # Set up the game configuration
    config = setup_config(max_round=10, initial_stack=10000, small_blind_amount=20)
    

    # Register the AI players
    for i, rl_agent in enumerate(rl_agents):
        config.register_player(name=f"RLAgent{i+1}", algorithm=rl_agent)

    # Register the console player
    config.register_player(name="Human", algorithm=ConsolePlayer())

    # Start the poker game
    game_result = start_poker(config, verbose=1)
    
    print(game_result)

if __name__ == "__main__":
    # train()
    game()