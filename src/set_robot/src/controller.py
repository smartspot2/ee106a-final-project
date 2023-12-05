import rospy
import intera_interface
import intera_external_devices
from intera_interface import CHECK_VERSION

from geometry_msgs.msg import Point

from set_msgs.msg import Card
from set_msgs.srv import CardData, TargetPosition, SolveSet

SHAPES = ['Ovals', 'Squiggles', 'Diamonds']
COLORS = ['Red', 'Purple', 'Green']
NUMBERS = ['One', 'Two', 'Three']
SHADINGS = ['Solid', 'Striped', 'Outlined']

DEFAULT_STATE = Point(0, 0, 0.5) # DANGER -- NEED TO SET
TABLE_OFFSET = 0.1 # DANGER -- NEED TO SET
DROPOFF_POINT = Point(2, 2, 0.5) # DANGER -- NEED TO SET

def play_set(gripper):
    print("Let's play Set!")

    round_number = 1
    while True:
        print(f"Round {round_number}")

        print("Getting cards and finding set...")
        cards, card_set_indices = get_card_data()
        print(f"Got {len(cards)} cards!")

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
            
            print(' | '.join([get_card_name(card) for card in card_set]))
            print("Hit ENTER to execute the queued command, anything else to stop everything:")

            continue_movement = input("Move to default state -- DANGER: THIS WAS RANDOMLY ASSIGNED TEMPORARILY ")
            if continue_movement == "":
                move_to(DEFAULT_STATE)

            for card in card_set:
                continue_movement = input("Move to card ")
                if continue_movement == "":
                    card.position.z += TABLE_OFFSET
                    print(f"Moving to {card}")
                    move_to(card.position)
                else:
                    return
                
                continue_movement = input("Pick up card ")
                if continue_movement == "":
                    gripper.open()
                else:
                    return
                
                continue_movement = input("Move to drop off point -- DANGER: THIS WAS RANDOMLY ASSIGNED TEMPORARILY ")
                if continue_movement == "":
                    print(f"Moving to {DROPOFF_POINT}")
                    move_to(DROPOFF_POINT)
                else:
                    return
                
                continue_movement = input("Drop off card ")
                if continue_movement == "":
                    gripper.close()
                else:
                    return
                
                continue_movement = input("Move to default state -- DANGER: THIS WAS RANDOMLY ASSIGNED TEMPORARILY ")
                if continue_movement == "":
                    print(f"Moving to {DEFAULT_STATE}")
                    move_to(DEFAULT_STATE)
            
            continue_game = input("Done! Place three new cards if there are less than 12 cards on the board. Hit ENTER to play another round, or anything else to finish. ")
            if continue_game == "":
                round_number += 1
            else:
                break

def rospy_error_wrapper(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except rospy.ServiceException as e:
            rospy.loginfo(e)
    return wrapper

@rospy_error_wrapper
def get_card_data():
    vision_proxy = rospy.ServiceProxy("/vision", CardData)
    rospy.loginfo("Get card data and set")
    response = vision_proxy()
    assert response is not None

    return response.cards, response.set

@rospy_error_wrapper
def move_to(position):
    sawyer_full_stack_proxy = rospy.ServiceProxy("/sawyer_target_card", TargetPosition)
    rospy.loginfo("Moving")
    sawyer_full_stack_proxy(position)

def get_card_name(card):
    return '-'.join([SHAPES[card.shape], COLORS[card.color], NUMBERS[card.number], SHADINGS[card.shading]])

def calibrate_robot():
    rp = intera_interface.RobotParams()
    valid_limbs = rp.get_limb_names()

    if not valid_limbs:
        rp.log_message(("Cannot detect any limb parameters on this robot. "
                        "Exiting."), "ERROR")
        return
    
    print("Initializing node...")
    rospy.init_node("set_robot")

    print("Getting robot state...")
    rs = intera_interface.RobotEnable(CHECK_VERSION)
    init_state = rs.state().enabled

    def clean_shutdown():
        print("\nStopping game...")

    rospy.on_shutdown(clean_shutdown)

    rospy.loginfo("Enabling robot...")
    rs.enable()



    try:
        gripper = intera_interface.Gripper(valid_limbs[0] + '_gripper')
    except:
        gripper = None
        rospy.loginfo("The electric gripper is not detected on the robot.")
    
    return gripper

if __name__ == "__main__":
    gripper = calibrate_robot()
    if gripper != None:
        play_set(gripper)