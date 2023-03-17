import keyboard
import time
def k_down(e):
    print(e.name, e.event_type)

def k_up(e):
    print(e.name, e.event_type)

# keyboard.on_press_key('k', k_down, suppress=False)
# keyboard.on_release_key('k', k_up, suppress=False)


def print_pressed_keys(e):
    #for code in keyboard._pressed_events:
    # '\r' and end='' overwrites the previous line.
    # ' '*40 prints 40 spaces at the end to ensure the previous line is cleared.
        print(e.time, e.name, e.event_type)

keyboard.hook(print_pressed_keys)
keyboard.wait()

