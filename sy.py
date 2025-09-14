from turtle import *
import math
import pathlib

import PIL
from PIL import Image, ImageDraw, ImageChops

# Mathematical co-ord system:
#  - right is +ve x
#  - up is +ve y
# (x, y) is a point
# (point1, point2) is a line
# (point1, point2, point3) is a triangle


def mirror_left_right(point):
    """
    Get the reflection point when Y-Axis is the mirror; Left <=> Right
    """
    x, y = point
    return -x, y

def mirror_up_down(point):
    """
    Get the reflection point when X-Axis is the mirror; Up <=> Down
    """
    x, y = point
    return x, -y

def intersection_point(line1, line2):
    print('Computing intersection for:', line1, line2)
    ((ax1, ay1), (ax2, ay2)) = line1
    ((bx1, by1), (bx2, by2)) = line2
    
    # Line 1: y=m1.x+c1
    m1 = (ay2 - ay1)/(ax2 - ax1)
    c1 = ay1 - ax1*m1

    # Line 2: y=m2.x+c2
    m2 = (by2 - by1)/(bx2 - bx1)
    c2 = by1 - bx1*m2

    # Intersection
    x = (c2-c1)/(m1-m2)
    y = m1*x + c1

    return x, y


def build_sy_triangles(radius, show_turtle=False, show_turtle_intermediate_steps=False):
    t = Turtle() if show_turtle else None

    #_ = input('Hit <Enter> to start')

    def draw_circle(centre, radius):
        if not t:
            return
        cx, cy = centre
        t.teleport(cx+radius, cy)
        t.setheading(90)
        t.circle(radius)


    def draw_line(line):
        if not t:
            return
        ((x1, y1), (x2, y2)) = line
        t.teleport(x1, y1)
        t.goto(x2, y2)


    def draw_polygon(points):
        if not t:
            return
        ox, oy = points[0]
        t.teleport(ox, oy)
        for x, y in points[1:]:
            t.goto(x, y)
        t.goto(ox, oy)

    origin_point = (0, 0)

    if t:
        t.home()

    circle_points_24 = [(round(radius * math.cos(i*math.pi/12), 6), round(radius * math.sin(i*math.pi/12), 6)) for i in range(24)]

    if show_turtle_intermediate_steps:
        draw_circle(origin_point, radius)

    # for point in circle_points_24:
    #    print(point)

    right_point = circle_points_24[0]
    top_point = circle_points_24[6]
    left_point = circle_points_24[12]
    bottom_point = circle_points_24[18]

    print(
        "Right, Top, Left, Bottom:",
        right_point, top_point, left_point, bottom_point)

    y_axis_line = (top_point, bottom_point)
    x_axis_line = (left_point, right_point)

    print("Y-Axis:", y_axis_line)
    print("X-Axis:", x_axis_line)

    if show_turtle_intermediate_steps:
        draw_line(y_axis_line)
        draw_line(x_axis_line)

    latitude_n_1 = (circle_points_24[11], circle_points_24[1], )
    latitude_s_1 = (circle_points_24[13], circle_points_24[23], )

    print("Lat N1:", latitude_n_1)
    print("Lat S1:", latitude_s_1)

    if show_turtle_intermediate_steps:
        draw_line(latitude_n_1)
        draw_line(latitude_s_1)

    up_triangle1 = (top_point,) + latitude_s_1
    down_triangle1 = (bottom_point,) + latitude_n_1

    print("Top-S1 triangle:", up_triangle1)
    print("Bottom-N1 triangle:", down_triangle1)

    if show_turtle_intermediate_steps:
        draw_polygon(up_triangle1)
        draw_polygon(down_triangle1)

    lat_n1_midpoint = (0.0, circle_points_24[1][1])
    lat_s1_midpoint = (0.0, circle_points_24[13][1])
    right_half_lat_s1 = (lat_s1_midpoint, latitude_s_1[1])
    length_half_lat_s1 = latitude_s_1[1][0]

    top_point_2 = (lat_s1_midpoint[0], lat_s1_midpoint[1] + length_half_lat_s1)
    bottom_point_2 = mirror_up_down(top_point_2)

    print('Top point 2:', top_point_2)
    print('Bottom point 2:', bottom_point_2)

    # up_tri1_base = up_triangle1[1], up_triangle1[2]
    up_tri1_base = latitude_s_1

    down_tri1_left_line = (down_triangle1[0], down_triangle1[1])

    up_tri2_part_left_intx_point = intersection_point(up_tri1_base, down_tri1_left_line)
    up_tri2_part_right_intx_point = mirror_left_right(up_tri2_part_left_intx_point)

    print('Upward triangle-2 Part Left Intersection point:', up_tri2_part_left_intx_point)
    print('Upward triangle-2 Part Right Intersection point2:', up_tri2_part_right_intx_point)

    up_triangle2_left_line = (top_point_2, up_tri2_part_left_intx_point)
    up_triangle2_right_line = (top_point_2, up_tri2_part_right_intx_point)

    print('Upward triangle-2 partial left line:', up_triangle2_left_line)
    print('Upward triangle-2 partial right line:', up_triangle2_right_line)

    if show_turtle_intermediate_steps:
        draw_line(up_triangle2_left_line)
        draw_line(up_triangle2_right_line)

    bottom_tri2_left_intx_point = mirror_up_down(up_tri2_part_left_intx_point)

    bottom_tri2_right_intx_point = mirror_left_right(bottom_tri2_left_intx_point)

    print('Bottom triangle-2 Left Intersection point:', bottom_tri2_left_intx_point)
    print('Bottom triangle-2 Right Intersection point2:', bottom_tri2_right_intx_point)

    down_triangle2_left_line = (bottom_point_2, bottom_tri2_left_intx_point)
    down_triangle2_right_line = (bottom_point_2, bottom_tri2_right_intx_point)

    print('Bottom triangle-2 left line:', down_triangle2_left_line)
    print('Bottom triangle-2 right line:', down_triangle2_right_line)

    if show_turtle_intermediate_steps:
        draw_line(down_triangle2_left_line)
        draw_line(down_triangle2_right_line)

    top_point_4 = up_triangle4_top_point = lat_n1_midpoint
    up_triangle4_part_base_left_point = intersection_point(down_triangle2_left_line, latitude_s_1)
    up_triangle4_part_base_right_point = (
        -up_triangle4_part_base_left_point[0],
        up_triangle4_part_base_left_point[1])

    up_triangle4_part = (up_triangle4_top_point, up_triangle4_part_base_left_point, up_triangle4_part_base_right_point)

    print('Upward triangle-4 part:', up_triangle4_part)

    if show_turtle_intermediate_steps:
        draw_polygon(up_triangle4_part)

    bottom_point_2_latitude = (
    (-1000.0, bottom_point_2[1]),
    (1000.0, bottom_point_2[1])
    )

    up_triangle4_base_left_point = intersection_point(
        bottom_point_2_latitude,
        (up_triangle4_part[0], up_triangle4_part[1])
    )
    up_triangle4_base_right_point = mirror_left_right(up_triangle4_base_left_point)
    up_triangle4 = (
        up_triangle4_top_point,
        up_triangle4_base_left_point,
        up_triangle4_base_right_point
    )

    print('Upward triangle-4:', up_triangle4)
    if show_turtle_intermediate_steps:
            draw_polygon(up_triangle4)

    up_triangle4_left_line = (up_triangle4_top_point, up_triangle4_base_left_point)

    up_triangle2_base_left_intx_pt1 = intersection_point(
        down_tri1_left_line,
        up_triangle4_left_line
    )

    up_triangle2_base_right_intx_pt1 = mirror_left_right(up_triangle2_base_left_intx_pt1)

    up_triangle2_base_line_part = (
        up_triangle2_base_left_intx_pt1,
        up_triangle2_base_right_intx_pt1    
    )

    up_triangle2_base_left_point = intersection_point(up_triangle2_left_line, up_triangle2_base_line_part)
    up_triangle2_base_right_point = mirror_left_right(up_triangle2_base_left_point)

    up_triangle2 = (top_point_2, up_triangle2_base_left_point, up_triangle2_base_right_point)

    print('Upward triangle-2:', up_triangle2)
    if show_turtle_intermediate_steps:
        draw_polygon(up_triangle2)

    trapezium_top_left_pt = intersection_point(up_triangle2_left_line, latitude_n_1)
    trapezium_top_right_pt = mirror_left_right(trapezium_top_left_pt)
    trapezium_bottom_left_pt = up_tri2_part_left_intx_point
    trapezium_bottom_right_pt = up_tri2_part_right_intx_point

    trapezium_diag_tlbr = (trapezium_top_left_pt, trapezium_bottom_right_pt)
    trapezium_diag_bltr = (trapezium_bottom_left_pt, trapezium_top_right_pt)
    print('Trapezium Diagonal 1:', trapezium_diag_tlbr)
    print('Trapezium Diagonal 2:', trapezium_diag_bltr)

    if show_turtle_intermediate_steps:
        draw_line(trapezium_diag_tlbr)
        draw_line(trapezium_diag_bltr)

    up_tri3_base_left_pt = intersection_point(trapezium_diag_bltr, down_triangle2_left_line)
    up_tri3_base_right_pt = mirror_left_right(up_tri3_base_left_pt)

    if show_turtle_intermediate_steps:
        draw_line((up_tri3_base_left_pt, up_tri3_base_right_pt))

    bottom_point_5 = 0, up_tri3_base_left_pt[1]
    down_triangle5_left_line = bottom_point_5, trapezium_top_left_pt
    down_triangle5_right_line = bottom_point_5, trapezium_top_right_pt

    top_point_2_latitude = (-1000.0, top_point_2[1]), (1000.0, top_point_2[1])
    down_triangle5_base_left_point = intersection_point(down_triangle5_left_line, top_point_2_latitude)
    down_triangle5_base_right_point = mirror_left_right(down_triangle5_base_left_point)

    down_triangle5 = bottom_point_5, down_triangle5_base_left_point, down_triangle5_base_right_point
    print('Bottom Triangle-5:', down_triangle5)
    if show_turtle_intermediate_steps:
        draw_polygon(down_triangle5)

    up_tri1_left_line = up_triangle1[0], up_triangle1[1]
    top_pt3_lat_left_pt = intersection_point(down_triangle5_left_line, up_tri1_left_line)
    top_pt3_lat_right_pt = mirror_left_right(top_pt3_lat_left_pt)
    top_pt3_lat = top_pt3_lat_left_pt, top_pt3_lat_right_pt

    top_point3 = 0, top_pt3_lat_left_pt[1]

    down_triangle2_base_left_pt =  intersection_point(down_triangle2_left_line, top_pt3_lat)
    down_triangle2_base_right_pt = mirror_left_right(down_triangle2_base_left_pt)
    down_triangle2 = (bottom_point_2, down_triangle2_base_left_pt, down_triangle2_base_right_pt)

    print('Downward triangle-2:', down_triangle2)
    if show_turtle_intermediate_steps:
        draw_polygon(down_triangle2)

    up_triangle3 = (top_point3, up_tri3_base_left_pt, up_tri3_base_right_pt)
    print('Up triangle-3:', up_triangle3)
    if show_turtle_intermediate_steps:
        draw_polygon(up_triangle3)

    up_tri3_left_line = (top_point3, up_tri3_base_left_pt)

    down_tri3_base_line_part_left_pt =  intersection_point(down_triangle5_left_line, up_tri3_left_line)
    down_tri3_base_line_part_right_pt = mirror_left_right(down_tri3_base_line_part_left_pt)
    down_tri3_base_line_part = (down_tri3_base_line_part_left_pt, down_tri3_base_line_part_right_pt)

    print('Down triangle-4 base line part:', down_tri3_base_line_part)
    if show_turtle_intermediate_steps:
        draw_line(down_tri3_base_line_part)

    down_tri3_base_line_left_pt = intersection_point(down_tri3_base_line_part, up_triangle2_left_line)
    down_tri3_base_line_right_pt = mirror_left_right(down_tri3_base_line_left_pt)

    bottom_point_3 = 0, up_triangle2_base_left_point[1]
    down_triangle3 = (bottom_point_3, down_tri3_base_line_left_pt, down_tri3_base_line_right_pt)

    print('Down triangle-3:', down_triangle3)
    if show_turtle_intermediate_steps:
        draw_polygon(down_triangle3)

    down_tri4_base_line_part_left_pt = intersection_point(up_triangle4_left_line, down_triangle5_left_line)
    down_tri4_base_line_part_right_pt = mirror_left_right(down_tri4_base_line_part_left_pt)
    down_tri4_base_line_part = (down_tri4_base_line_part_left_pt, down_tri4_base_line_part_right_pt)

    print('Down triangle-4 base line part:', down_tri4_base_line_part)
    if show_turtle_intermediate_steps:
        draw_line(down_tri4_base_line_part)

    bottom_point_4 = lat_s1_midpoint
    down_tri4_base_line_left_pt = intersection_point(up_tri3_left_line, down_tri4_base_line_part)
    down_tri4_base_line_right_pt = mirror_left_right(down_tri4_base_line_left_pt)

    down_triangle4 = (bottom_point_4, down_tri4_base_line_left_pt, down_tri4_base_line_right_pt)

    print('Down triangle-4:', down_triangle4)
    if show_turtle_intermediate_steps:
        draw_polygon(down_triangle4)

    if t:
        if show_turtle_intermediate_steps:
            _ = input('Hit <ENTER> for final result')
            t.clear()

        draw_circle(origin_point, radius)
        draw_polygon(up_triangle1)
        draw_polygon(down_triangle1)
        draw_polygon(up_triangle4)
        draw_polygon(up_triangle2)
        draw_polygon(down_triangle5)
        draw_polygon(down_triangle2)
        draw_polygon(up_triangle3)
        draw_polygon(down_triangle3)
        draw_polygon(down_triangle4)

        t.teleport(0, 0)

        t.screen.mainloop()

    return up_triangle1, down_triangle1, up_triangle4, up_triangle2, down_triangle5, down_triangle2, up_triangle3, down_triangle3, down_triangle4


def make_sy_outer_circle_image(outer_circle_radius):
    img = Image.new('RGB', (2*outer_circle_radius+1, 2*outer_circle_radius+1), color = 'black')

    draw = ImageDraw.Draw(img)

    # Drawing a green circle on the image
    draw.circle(xy = (outer_circle_radius, outer_circle_radius), radius=outer_circle_radius,
                fill = (255, 255, 255),
                outline = (255, 255, 255),
                width = 1)
    return img

def make_sy_triangle_image(sy_triangle, outer_circle_radius):
    t_img = Image.new('RGB', (2*outer_circle_radius+1, 2*outer_circle_radius+1), color = 'black')
    t_draw = ImageDraw.Draw(t_img)
    t_draw.polygon(sy_triangle, fill = (255, 255, 255), outline = (255, 255, 255))
    return t_img

def to_image_coords(point, tx=0, ty=0):
    x, y = point
    return x + tx, ty - y

def generate_sy_png_gif(radius, sy_triangles):
    blank_image = Image.new('RGB', (2*radius+1, 2*radius+1), color = 'black')
    circle_image = make_sy_outer_circle_image(radius)
    triangle_images = [make_sy_triangle_image([to_image_coords(syt_pt, radius, radius) for syt_pt in syt], radius) for syt in sy_triangles]

    xor_image_list = [blank_image, circle_image] + triangle_images
    xor_image_list_converted = [ximg.convert("1") for ximg in xor_image_list]

    xor_result_image = None
    gif_frames = xor_image_list_converted[:]
    # xor_image.show()
    for image_to_xor in xor_image_list_converted:
        xor_result_image = ImageChops.logical_xor(xor_result_image, image_to_xor) if xor_result_image else image_to_xor
        # xor_image.show()
        gif_frames.append(xor_result_image)

    # xor_result_image.show()
    gif_frames.append(xor_result_image)

    xor_result_image.save(f'sy-{radius}.png')

    gif_frames[0].save(f'sy-{radius}.gif', save_all = True, append_images = gif_frames[1:],
                        optimize = False, duration = 1000, loop=0)


def generate_sy_svg(radius, sy_triangles):
    format_params = {
        'outer_width': 2*radius+10, 
        'outer_height': 2*radius+10,
        'inner_width': 2*radius,
        'inner_height': 2*radius,
        'out_in_padding': 5,
        'circle_centre_x': radius,
        'circle_centre_y': radius,
        'circle_radius': radius,
    }

    image_triangles = [[to_image_coords(syt_pt, radius, radius) for syt_pt in syt] for syt in sy_triangles]
    for i, tri in enumerate(image_triangles):
        key = f't{i+1}_points'
        val = ' '.join([','.join([str(v) for v in pt]) for pt in tri])
        format_params[key] = val
    
    template_svg_str = pathlib.Path('sy-svg-template.svg').read_text()
    svg_str = template_svg_str.format_map(format_params)
    pathlib.Path(f'sy-{radius}.svg').write_text(svg_str)


def main():
    radius = 300
    sy_triangles = build_sy_triangles(radius, show_turtle=False, show_turtle_intermediate_steps=True)
    generate_sy_png_gif(radius, sy_triangles)
    generate_sy_svg(radius, sy_triangles)


if __name__ == '__main__':
    main()

