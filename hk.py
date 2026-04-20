from functools import partial
import pathlib
import turtle

from shapes import *
from turtle_capture import TurtleCapture


(OUTPUT_DIR := pathlib.Path('output')).mkdir(exist_ok=True)


SPOKES = 8
RINGS = 5
RING_GAP = 72

DOT_RADIUS = 12


def sr_to_xy(spoke, ring):
    r = (ring + 1) * RING_GAP
    theta_degrees = spoke * 360.0/SPOKES
    theta = theta_degrees * math.pi/180.0
    x = r * math.cos(theta)
    y = r * math.sin(theta)

    return x, y



def get_next_ring_sharp(ring):
    return (ring + 2)%RINGS


def get_next_spoke_clockwise(spoke):
    return (spoke + SPOKES - 1) % SPOKES

def get_next_spoke_anticlockwise(spoke):
    return (spoke + 1) % SPOKES

get_previous_spoke_clockwise = get_next_spoke_anticlockwise
get_previous_spoke_anticlockwise = get_next_spoke_clockwise

def draw_dots(tt):
    tt.teleport(0, 0)

    tt.dot(DOT_RADIUS*2, "grey")

    for ring in range(RINGS):
        old_col = tt.pencolor()
        print(old_col)
        #print(t.colormode())
        tt.pencolor("lightgrey")
        Circle((0, 0), RING_GAP*(ring+1)).turtle_draw(tt)
        tt.pencolor(old_col)
        for spoke in range(SPOKES):
            xy = sr_to_xy(spoke, ring)
            # Circle(xy, DOT_RADIUS).turtle_draw(tt, fill="grey")
            tt.teleport(xy[0], xy[1])
            tt.dot(DOT_RADIUS*2-2, "grey")
            Text(str(ring + 1), xy, h_align="center", v_align="bottom", color="lightgrey", font=('Arial', DOT_RADIUS-2, 'bold')).turtle_draw(tt)
    
    for spoke in range(SPOKES):
        x1, y1 = sr_to_xy(spoke, 0)
        tt.teleport(x1, y1)
        old_col = tt.pencolor()
        print(old_col)
        #print(t.colormode())
        tt.pencolor("lightgrey")
        tt.goto(sr_to_xy(spoke, RINGS-1))
        tt.pencolor(old_col)


def draw_lotus(tt, next_ring_func, next_spoke_func, prev_spoke_func):

    tt.pencolor("black")
    tt.pensize(4)

    cur_spoke = 0
    cur_ring = 0

    tt.penup()
    cur_point = sr_to_xy(cur_spoke, cur_ring)
    tt.goto(cur_point)
    tt.pendown()

    next_spoke = next_ring = -1

    while next_spoke != 0 or next_ring != 0:
        next_spoke = next_spoke_func(cur_spoke)
        next_ring = next_ring_func(cur_ring)
        
        next_point = sr_to_xy(next_spoke, next_ring)
        
        if cur_ring == 2:
            prev_ip_x, prev_ip_y = sr_to_xy(prev_spoke_func(cur_spoke), cur_ring-1)
            intermediate_point = (cur_point[0]*2 - prev_ip_x, cur_point[1]*2 - prev_ip_y)
        elif cur_ring < next_ring:
            intermediate_point = sr_to_xy(cur_spoke, cur_ring+1)
        else:
            intermediate_point = sr_to_xy(next_spoke, cur_ring-1)
        
        BezierCurve([cur_point, intermediate_point, next_point], resolution=80).turtle_draw(tt)

        # tt.goto()

        cur_spoke = next_spoke
        cur_ring = next_ring
        cur_point = next_point

    
    tt.penup()


def _show_count_down(t, seconds_left):
    t.clear()
    t.write(seconds_left)
    if seconds_left > 1:
        t.screen.ontimer(partial(_show_count_down, t, seconds_left-1), t=1000)


def draw_all(t, tt_cap=None):
    draw_dots(t)

    t.screen.delay(1000)

    t.hideturtle()
    t.teleport(0, 0)
    t.showturtle()
    
    t.screen.delay(0)
    t.speed(2)

    draw_lotus(t, get_next_ring_sharp, get_next_spoke_clockwise, get_previous_spoke_clockwise)

    t.hideturtle()
    if tt_cap:
        t.screen.ontimer(tt_cap.stop_capture, t=3000)


def do_turtle_main_loop(t, tt_cap=None):
    t.speed(0)
    
    wait_time = 5
    t.screen.ontimer(partial(draw_all, t, tt_cap), t=wait_time*1000)

    t.screen.ontimer(partial(_show_count_down, t, wait_time-1))

    t.screen.mainloop()


def init_from_args():
    import sys
    args = sys.argv[1:]

    capture = ('--capture' in args)

    return { 'capture': capture }

    
def main():
    params = init_from_args()
    capture = params['capture']

    t = turtle.Turtle()

    if capture:
        xy_val = RING_GAP*RINGS + 10
        wh_val = 2*RING_GAP*RINGS + 20

        capture_x, capture_y, capture_width, capture_height = (-xy_val, -xy_val, wh_val, wh_val)
        print(capture_x, capture_y, capture_width, capture_height)

        with TurtleCapture(
            t, capture_interval=50, 
            capture_x=capture_x, capture_y=capture_y,
            capture_width=capture_width, capture_height=capture_height,
            capture_filename_prefix='hk',
            output_dir=OUTPUT_DIR
        ) as tt_cap:
        
            do_turtle_main_loop(t, tt_cap)
    
    else:
        do_turtle_main_loop(t)


if __name__ == "__main__":
    main()

