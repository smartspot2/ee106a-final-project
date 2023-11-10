import rospy
from set_msgs.msg import Card
from set_msgs.srv import CardData, Position, SolveSet

SHAPES = ['Ovals', 'Squiggles', 'Diamonds']
COLORS = ['Red', 'Purple', 'Green']
NUMBERS = ['One', 'Two', 'Three']
SHADING = ['Solid', 'Striped', 'Outlined']

def play_set():
    rospy.init_node("set_robot")
    
    print("Let's play Set!")

    round_number = 1
    while True:
        print(f"Round {round_number}")

        print("Getting cards...")
        cards = get_card_data()
        print(f"Got {len(cards)} cards!")

        print("Finding set...")
        card_set_indices = get_set(cards)
        if len(card_set_indices) == 0:
            print("No sets found!")
            continue_game = input("If the human agrees, place three cards and hit ENTER. If the human finds a set, type L. If the human agrees and there are no more cards in the deck, type anything else. ")
            if continue_game == "":
                continue
            elif continue_game == 'L' or continue_game == 'l':
                print("Damn, you beat me!")
                continue
            else:
                break
        else:
            assert len(card_set_indices) == 3, f"The algorithm didn't return a valid set, {len(card_set_indices)} card(s) were returned"
            print("---------------------------------- SET! ----------------------------------")
            
            card_set = [cards[i] for i in card_set_indices]
            
            print([get_card_name(card) for card in card_set].join(' | '))

            for card in card_set:
                continue_pick_up = input("Hit ENTER to pick up the first card, anything else to stop everything. ")
                if continue_pick_up == "":
                    move_card(card.position)
                else:
                    break
            
            continue_game = input("Done! Place three new cards if there are less than 12 cards on the board. Hit ENTER to play another round, or anything else to finish. ")
            if continue_game == "":
                round_number += 1
            else:
                break

@rospy_error_wrapper
def get_card_data():
    vision_proxy = rospy.ServiceProxy("/vision", CardData)
    rospy.loginfo("Get card data")
    return vision_proxy().cards

@rospy_error_wrapper
def get_set(cards):
    set_solver_proxy = rospy.ServiceProxy("/set_solver", SolveSet)
    rospy.loginfo("Get set")
    return set_solver_proxy(cards).set

@rospy_error_wrapper
def move_card(position):
    sawyer_full_stack_proxy = rospy.ServiceProxy("/sawyer_full_stack", Position)
    rospy.loginfo("Pick up card")
    sawyer_full_stack_proxy(position)

def rospy_error_wrapper(fn):
    try:
        return fn()
    except rospy.ServiceException as e:
        rospy.loginfo(e)

def get_card_name(card):
    return [SHAPES[card.shape], COLORS[card.color], NUMBERS[card.number], SHADINGS[card.shading]].join('-')

if __name__ == "__main__":
    play_set()