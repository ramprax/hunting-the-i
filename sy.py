from turtle import *
import math
import pathlib
from typing import Iterable

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

def mirror_xy(point):
    """
    Get the reflection point when Y=X line is the mirror; Diagonal
    """
    x, y = point
    return y, x


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

import abc

class Shape(abc.ABC):
    @staticmethod
    def to_image_coords(xy, image_centre_xy):
        tx, ty = image_centre_xy
        x, y = xy
        return x + tx, ty - y

    @abc.abstractmethod
    def turtle_draw(self, tt: Turtle):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def pil_draw(self, img_draw: ImageDraw, image_centre_xy, fill=None, outline=(255, 255, 255), width=1):
        raise NotImplementedError()

    @abc.abstractmethod
    def svg_draw(self, element_id, image_centre_xy, fill, stroke):
        raise NotImplementedError()

class Circle(Shape):
    def __init__(self, centre_xy, radius):
        cx, cy = centre_xy
        self._centre_xy = cx, cy
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    def turtle_draw(self, tt: Turtle):
        if not tt:
            return
        cx, cy = self._centre_xy
        tt.teleport(cx+self._radius, cy)
        tt.setheading(90)
        tt.circle(self._radius)
    
    def pil_draw(self, img_draw: ImageDraw, image_centre_xy, fill=None, outline=(255, 255, 255), width=1):
        c_xy = super().to_image_coords(self._centre_xy, image_centre_xy)
        img_draw.circle(xy = c_xy, radius=self._radius,
                fill = fill,
                outline = outline,
                width = width)
    
    def svg_draw(self, element_id, image_centre_xy, fill, stroke):
        c_xy = super().to_image_coords(self._centre_xy, image_centre_xy)
        cx, cy = c_xy
        return f'''<circle id="{element_id}" cx="{cx}" cy="{cy}" r="{math.ceil(self._radius)}" fill="{fill}" stroke="{stroke}"/>'''


class Polygon(Shape):
    def __init__(self, points):
        self._points = [(x, y) for x, y in points]

    def __len__(self):
        return len(self._points)

    def __getitem__(self, idx):
        return self._points[idx]    

    def turtle_draw(self, tt: Turtle):
        if not tt:
            return
        ox, oy = self._points[0]
        tt.teleport(ox, oy)
        for x, y in self._points[1:]:
            tt.goto(x, y)
        tt.goto(ox, oy)

    def pil_draw(self, img_draw: ImageDraw, image_centre_xy, fill=None, outline=(255, 255, 255), width=1):
        if image_centre_xy:
            image_coord_points = [super().to_image_coords(xy, image_centre_xy) for xy in self._points] 
        else:
            image_coord_points = self._points[:]
        img_draw.polygon(image_coord_points, fill = fill, outline = outline, width=width)

    def svg_draw(self, element_id, image_centre_xy, fill, stroke):
        image_coord_points = [super().to_image_coords(xy, image_centre_xy) for xy in self._points] 
        image_coord_points_str = ' '.join([','.join([str(v) for v in pt]) for pt in image_coord_points])
        return f'''<polygon id="{element_id}" points="{image_coord_points_str}" fill="{fill}" stroke="{stroke}"/>'''


class BezierCurve(Shape):
    def __init__(self, guide_points):
        self._guide_points = [(x, y) for x, y in guide_points]

    @staticmethod
    def calculate_bezier_curve_points(guide_points):
        n = len(guide_points) - 1
        coeffs = [math.comb(n, i) for i in range(n+1)]
        curve_points = []
        for tn in range(101):
            t = tn/100
            cx = 0
            cy = 0
            for i, (x, y) in enumerate(guide_points):
                coeff = coeffs[i] * (t**i) * ((1-t)**(n-i))
                cx += coeff * x
                cy += coeff * y
            curve_points.append((cx, cy))
        return curve_points
    
    @property
    def guide_points(self):
        return self._guide_points[:]
    
    def curve_points(self, to_image_coords_ox_oy = None):
        if to_image_coords_ox_oy is not None:
            guide_points = [self.to_image_coords(pt, to_image_coords_ox_oy) for pt in self._guide_points]
        else:
            guide_points = self._guide_points
        return self.calculate_bezier_curve_points(guide_points)

    def turtle_draw(self, tt: Turtle):
        if not tt:
            return
        curve_pts = self.curve_points()
        ox, oy = curve_pts[0]
        tt.teleport(ox, oy)
        for x, y in curve_pts[1:]:
            tt.goto(x, y)

    def pil_draw(self, img_draw: ImageDraw, image_centre_xy, fill=None, outline=(255, 255, 255), width=1):
        curve_points = self.curve_points(to_image_coords_ox_oy=image_centre_xy)
        img_draw.line(curve_points, fill=fill, width=width)
    
    def svg_draw(self, element_id, image_centre_xy, fill, stroke):
        if image_centre_xy:
            guide_points = [to_image_coords(pt, image_centre_xy) for pt in self._guide_points]
        else:
            guide_points = self._guide_points
        start_x, start_y = guide_points[0]
        guide_points_str = ', '.join([' '.join(xy) for xy in guide_points[1:]])
        svg_bezier_path_str = f'''<path id="{element_id}" d="M {start_x} {start_y} C {guide_points_str}" fill="{fill}" stroke="{stroke}"/>'''
        return svg_bezier_path_str

class Dalam(Shape):
    def __init__(self, n, base_radius, tip_radius):
        self._n = n
        self._base_radius = base_radius
        self._tip_radius = tip_radius

    def _make_dalam_guide_points(self):
        angle_incr = 2*math.pi/self._n
        half_angle_incr = math.pi/self._n
        dalam_tips_x = [
            self._tip_radius*math.cos(i*angle_incr) for i in range(self._n)
        ]
        dalam_tips_y = [
            self._tip_radius*math.sin(i*angle_incr) for i in range(self._n)
        ]
        dalam_tips = list(zip(dalam_tips_x, dalam_tips_y))

        dalam_bases_x = [
            self._base_radius*math.cos(half_angle_incr + i*angle_incr) for i in range(self._n)
        ]
        dalam_bases_y = [
            self._base_radius*math.sin(half_angle_incr + i*angle_incr) for i in range(self._n)
        ]
        dalam_bases = list(zip(dalam_bases_x, dalam_bases_y))

        dalam_intp1_x = [
            self._base_radius*math.cos(i*angle_incr) for i in range(self._n)
        ]
        dalam_intp1_y = [
            self._base_radius*math.sin(i*angle_incr) for i in range(self._n)
        ]
        dalam_intp1 = list(zip(dalam_intp1_x, dalam_intp1_y))

        dalam_intp2_x = [
            self._tip_radius*math.cos(half_angle_incr + i*angle_incr) for i in range(self._n)
        ]
        dalam_intp2_y = [
            self._tip_radius*math.sin(half_angle_incr + i*angle_incr) for i in range(self._n)
        ]
        dalam_intp2 = list(zip(dalam_intp2_x, dalam_intp2_y))

        return dalam_tips, dalam_intp1, dalam_intp2, dalam_bases

    def make_dalam_beziers(self, to_image_coords_ox_oy = None):
        dalam_guide_points = self._make_dalam_guide_points()
        if to_image_coords_ox_oy:
            dalam_guide_points = tuple([tuple([self.to_image_coords(xy, to_image_coords_ox_oy)
                                  for xy in dalam_tiib]) for dalam_tiib in dalam_guide_points])
        
        dalam_tips, dalam_intp1, dalam_intp2, dalam_bases = dalam_guide_points

        n = len(dalam_tips)
        dalam_beziers = []
        for i, tip_pt in enumerate(dalam_tips):
            base_pt = dalam_bases[(i+n-1)%n]
            int_pt_1 = dalam_intp1[i]
            int_pt_2 = dalam_intp2[(i+n-1)%n]
            bz1 = BezierCurve([base_pt, int_pt_2, int_pt_1, tip_pt])
            dalam_beziers.append(bz1)

            base_pt = dalam_bases[i]
            int_pt_1 = dalam_intp1[i]
            int_pt_2 = dalam_intp2[i]
            bz2 = BezierCurve([tip_pt, int_pt_1, int_pt_2, base_pt])
            dalam_beziers.append(bz2)

        return dalam_beziers

    def curve_points(self, to_image_coords_ox_oy = None):
        dalam_beziers = self.make_dalam_beziers(to_image_coords_ox_oy)
        dalam_curve_points = []
        for bz in dalam_beziers:
            dalam_curve_points += bz.curve_points()
        return dalam_curve_points

    def turtle_draw(self, tt: Turtle):
        if not tt:
            return
        Polygon(self.curve_points()).turtle_draw(tt)
    
    def pil_draw(self, img_draw: ImageDraw, image_centre_xy, fill=None, outline=(255, 255, 255), width=1):
        c_pts = self.curve_points(image_centre_xy)
        Polygon(c_pts).pil_draw(img_draw, None, fill=fill, outline=outline, width=width)
    
    def svg_draw(self, element_id, image_centre_xy, fill, stroke):
        bzs = self.make_dalam_beziers(image_centre_xy)
        path_instructions = []
        last_end_pt = None
        for bz in bzs:
            g_pts = bz.guide_points
            cur_start_pt = g_pts[0]
            if last_end_pt != cur_start_pt:
                sx, sy = cur_start_pt
                path_instructions.append('M')
                path_instructions.append(str(sx))
                path_instructions.append(str(sy))
            path_instructions.append('C')
            path_instructions.append(', '.join([' '.join([str(v) for v in g_pt]) for g_pt in g_pts[1:]]))
            last_end_pt = g_pts[-1]

        path_instruction_str = ' '.join(path_instructions)

        svg_dalam_path_str = f'''<path id="{element_id}" d="{path_instruction_str}" fill="{fill}" stroke="{stroke}"/>'''
        return svg_dalam_path_str


def sy_bhupura_shapes(radius, turtle, show_turtle=True, show_intermediate_steps=True):

    c = Circle((0, 0), radius=radius)
    if show_intermediate_steps:
        c.turtle_draw(turtle)
    
    # square_pts = []
    # angle = math.pi/4
    sin_cos_45 = 1/(2**.5)
    square_pts = (
        (radius*sin_cos_45, radius*sin_cos_45),
        (-radius*sin_cos_45, radius*sin_cos_45),
        (-radius*sin_cos_45, -radius*sin_cos_45),
        (radius*sin_cos_45, -radius*sin_cos_45),
    )
    
    sq = Polygon(square_pts)
    if show_intermediate_steps:
        sq.turtle_draw(turtle)

    sq_side = 2*radius*sin_cos_45
    sin_t = 1/2**1.5 # (sq_side/4)/radius
    cos_t = (7**.5)/2**1.5 # sqrt(1-(sin_t)^2)

    right_rectangle_points = (
        (radius*cos_t, radius*sin_t),
        (sq_side/2,    radius*sin_t),
        (sq_side/2,    -radius*sin_t),
        (radius*cos_t, -radius*sin_t),
    )
    right_rect = Polygon(right_rectangle_points)
    if show_intermediate_steps:
        right_rect.turtle_draw(turtle)
    
    top_rectangle_points = tuple(
        mirror_xy(xy) for xy in right_rectangle_points
    )
    top_rect = Polygon(top_rectangle_points)
    if show_intermediate_steps:
        top_rect.turtle_draw(turtle)
    
    left_rectangle_points = tuple(
        mirror_left_right(xy) for xy in right_rectangle_points
    )
    left_rect = Polygon(left_rectangle_points)
    if show_intermediate_steps:
        left_rect.turtle_draw(turtle)

    bottom_rectangle_points = tuple(
        mirror_up_down(xy) for xy in top_rectangle_points
    )
    bottom_rect = Polygon(bottom_rectangle_points)
    if show_intermediate_steps:
        bottom_rect.turtle_draw(turtle)

    right_neck_x = radius*cos_t - sq_side/8
    right_neck_y = sq_side/8

    one_eigth_polygon = (
        (radius*cos_t, 0),
        (radius*cos_t, sq_side/4),
        (right_neck_x, sq_side/4),
        (right_neck_x, right_neck_y),
        (sq_side/2, right_neck_y),
        (sq_side/2, sq_side/2),
    )
    boundary_polygon = full_polygon_from_1_8th(one_eigth_polygon)
    boundary_polygon.turtle_draw(turtle)

    # boundary thickness
    bt = right_neck_x - sq_side/2
    bt_half = bt/2
    middle_boundary_1_8th = (
        (radius*cos_t - bt_half, 0),
        (radius*cos_t - bt_half, sq_side/4 - bt_half),
        (right_neck_x + bt_half, sq_side/4 - bt_half),
        (right_neck_x + bt_half, right_neck_y - bt_half),
        (sq_side/2 - bt_half, right_neck_y - bt_half),
        (sq_side/2 - bt_half, sq_side/2 - bt_half),
    )
    middle_boundary_polygon = full_polygon_from_1_8th(middle_boundary_1_8th)
    middle_boundary_polygon.turtle_draw(turtle)

    inner_boundary_1_8th = (
        (radius*cos_t - bt, 0),
        (radius*cos_t - bt, sq_side/4 - bt),
        (right_neck_x + bt, sq_side/4 - bt),
        (right_neck_x + bt, right_neck_y - bt),
        (sq_side/2 - bt, right_neck_y - bt),
        (sq_side/2 - bt, sq_side/2 - bt),
    )
    inner_boundary_polygon = full_polygon_from_1_8th(inner_boundary_1_8th)
    inner_boundary_polygon.turtle_draw(turtle)

    inner_b_corner = (sq_side/2 - bt, right_neck_y - bt)
    corner_dist = ((sq_side/2 - bt)**2 + (right_neck_y - bt)**2)**0.5
    circle_corner_touch = Circle((0, 0), corner_dist)
    circle_corner_touch.turtle_draw(turtle)

    circle_1_radius = sq_side/2 - bt
    circle_1 = Circle((0, 0), circle_1_radius)
    circle_1.turtle_draw(turtle)

    circle_gap = (radius*cos_t - bt) - (sq_side/2)
    circle_2 = Circle((0, 0), circle_1_radius - circle_gap/2)
    circle_2.turtle_draw(turtle)

    circle_3 = Circle((0, 0), circle_1_radius - circle_gap)
    circle_3.turtle_draw(turtle)

    return (
        boundary_polygon, middle_boundary_polygon, inner_boundary_polygon,
        circle_corner_touch, circle_1, circle_2, circle_3,
    )


def full_polygon_from_1_8th(poly_points_1_8th):
    poly_points_1_8th_2 = tuple(reversed(tuple(mirror_xy(xy) for xy in poly_points_1_8th)))
    poly_points_1_4th = poly_points_1_8th + poly_points_1_8th_2
    
    poly_points_1_4th_2 = tuple(reversed(tuple(mirror_left_right(xy) for xy in poly_points_1_4th)))
    poly_points_half = poly_points_1_4th + poly_points_1_4th_2

    poly_points_half_2 = tuple(reversed(tuple(mirror_up_down(xy) for xy in poly_points_half)))
    poly_points = poly_points_half + poly_points_half_2

    return Polygon(poly_points)


def build_sy_shapes(outer_radius, show_turtle=False, show_turtle_intermediate_steps=False) -> Iterable[Shape]:
    turtle = Turtle() if show_turtle else None
    
    bhupura_shapes = sy_bhupura_shapes(outer_radius, turtle, show_turtle=show_turtle, show_intermediate_steps=show_turtle_intermediate_steps)
    
    dalam_circle1, dalam_circle2, dalam_circle3 = bhupura_shapes[-3:]
    
    shodasha_dalam_radius = dalam_circle1.radius
    shodasha_dalam_width = dalam_circle1.radius - dalam_circle2.radius
    ashta_dalam_width = dalam_circle2.radius - dalam_circle3.radius

    radius = dalam_circle3.radius

    #_ = input('Hit <Enter> to start')

    def draw_circle(centre, radius):
        if not turtle:
            return
        cx, cy = centre
        turtle.teleport(cx+radius, cy)
        turtle.setheading(90)
        turtle.circle(radius)

    def draw_line(line):
        if not turtle:
            return
        ((x1, y1), (x2, y2)) = line
        turtle.teleport(x1, y1)
        turtle.goto(x2, y2)

    def draw_polygon(points):
        if not turtle:
            return
        if points[0] != points[-1]:
            points = list(points[:]) + [points[0]]
        draw_polyline(points)
    
    def draw_polyline(points):
        if not turtle:
            return
        ox, oy = points[0]
        turtle.teleport(ox, oy)
        for x, y in points[1:]:
            turtle.goto(x, y)


    origin_point = (0, 0)

    if turtle:
        turtle.home()

    circle_points_24 = [(round(radius * math.cos(i*math.pi/12), 6), round(radius * math.sin(i*math.pi/12), 6)) for i in range(24)]

    # outer_circle = Circle(origin_point, radius)
    # if show_turtle_intermediate_steps:
    #     # draw_circle(origin_point, radius)
    #     outer_circle.turtle_draw(turtle)

    # for point in circle_points_24:
    #    print(point)

    # Ashta dalam
    ashta_dalam = Dalam(8, radius, radius + ashta_dalam_width)

    if show_turtle_intermediate_steps:
        ashta_dalam.turtle_draw(turtle)

    # Shodasha dalam
    shodasha_dalam = Dalam(16, shodasha_dalam_radius - shodasha_dalam_width, shodasha_dalam_radius)

    if show_turtle_intermediate_steps:
        shodasha_dalam.turtle_draw(turtle)

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

    up_triangle1 = Polygon((top_point,) + latitude_s_1)
    down_triangle1 = Polygon((bottom_point,) + latitude_n_1)

    print("Top-S1 triangle:", up_triangle1)
    print("Bottom-N1 triangle:", down_triangle1)

    if show_turtle_intermediate_steps:
        up_triangle1.turtle_draw(turtle)
        down_triangle1.turtle_draw(turtle)

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
    up_triangle4 = Polygon((
        up_triangle4_top_point,
        up_triangle4_base_left_point,
        up_triangle4_base_right_point
    ))

    print('Upward triangle-4:', up_triangle4)
    if show_turtle_intermediate_steps:
        up_triangle4.turtle_draw(turtle)
        # draw_polygon(up_triangle4)

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

    up_triangle2 = Polygon((top_point_2, up_triangle2_base_left_point, up_triangle2_base_right_point))

    print('Upward triangle-2:', up_triangle2)
    if show_turtle_intermediate_steps:
        up_triangle2.turtle_draw(turtle)
        # draw_polygon(up_triangle2)

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

    down_triangle5 = Polygon((bottom_point_5, down_triangle5_base_left_point, down_triangle5_base_right_point))
    print('Bottom Triangle-5:', down_triangle5)
    if show_turtle_intermediate_steps:
        down_triangle5.turtle_draw(turtle)
        # draw_polygon(down_triangle5)

    up_tri1_left_line = up_triangle1[0], up_triangle1[1]
    top_pt3_lat_left_pt = intersection_point(down_triangle5_left_line, up_tri1_left_line)
    top_pt3_lat_right_pt = mirror_left_right(top_pt3_lat_left_pt)
    top_pt3_lat = top_pt3_lat_left_pt, top_pt3_lat_right_pt

    top_point3 = 0, top_pt3_lat_left_pt[1]

    down_triangle2_base_left_pt =  intersection_point(down_triangle2_left_line, top_pt3_lat)
    down_triangle2_base_right_pt = mirror_left_right(down_triangle2_base_left_pt)
    down_triangle2 = Polygon((bottom_point_2, down_triangle2_base_left_pt, down_triangle2_base_right_pt))

    print('Downward triangle-2:', down_triangle2)
    if show_turtle_intermediate_steps:
        down_triangle2.turtle_draw(turtle)
        # draw_polygon(down_triangle2)

    up_triangle3 = Polygon((top_point3, up_tri3_base_left_pt, up_tri3_base_right_pt))
    print('Up triangle-3:', up_triangle3)
    if show_turtle_intermediate_steps:
        up_triangle3.turtle_draw(turtle)
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
    down_triangle3 = Polygon((bottom_point_3, down_tri3_base_line_left_pt, down_tri3_base_line_right_pt))

    print('Down triangle-3:', down_triangle3)
    if show_turtle_intermediate_steps:
        down_triangle3.turtle_draw(turtle)
        # draw_polygon(down_triangle3)

    down_tri4_base_line_part_left_pt = intersection_point(up_triangle4_left_line, down_triangle5_left_line)
    down_tri4_base_line_part_right_pt = mirror_left_right(down_tri4_base_line_part_left_pt)
    down_tri4_base_line_part = (down_tri4_base_line_part_left_pt, down_tri4_base_line_part_right_pt)

    print('Down triangle-4 base line part:', down_tri4_base_line_part)
    if show_turtle_intermediate_steps:
        draw_line(down_tri4_base_line_part)

    bottom_point_4 = lat_s1_midpoint
    down_tri4_base_line_left_pt = intersection_point(up_tri3_left_line, down_tri4_base_line_part)
    down_tri4_base_line_right_pt = mirror_left_right(down_tri4_base_line_left_pt)

    down_triangle4 = Polygon((bottom_point_4, down_tri4_base_line_left_pt, down_tri4_base_line_right_pt))

    print('Down triangle-4:', down_triangle4)
    if show_turtle_intermediate_steps:
        down_triangle4.turtle_draw(turtle)
        # draw_polygon(down_triangle4)

    centre_circle = Circle(origin_point, radius/100)
    if show_turtle_intermediate_steps:
        centre_circle.turtle_draw(turtle)
        # draw_circle(origin_point, radius/100)
    
    final_shapes = bhupura_shapes + (
        shodasha_dalam, ashta_dalam,
        up_triangle1, down_triangle1, up_triangle4,
        up_triangle2, down_triangle5, down_triangle2,
        up_triangle3, down_triangle3, down_triangle4,
        centre_circle
    )

    if turtle:
        if show_turtle_intermediate_steps:
            _ = input('Hit <ENTER> for final result')
            turtle.clear()

        # draw_circle(origin_point, radius)
        # draw_polygon(up_triangle1)
        # draw_polygon(down_triangle1)
        # draw_polygon(up_triangle4)
        # draw_polygon(up_triangle2)
        # draw_polygon(down_triangle5)
        # draw_polygon(down_triangle2)
        # draw_polygon(up_triangle3)
        # draw_polygon(down_triangle3)
        # draw_polygon(down_triangle4)
        # draw_circle(origin_point, radius/100)

        for s in final_shapes:
            s.turtle_draw(turtle)

        turtle.teleport(0, 0)

        turtle.screen.mainloop()

    return final_shapes


def make_sy_outer_circle_image(radius):
    img = Image.new('RGB', (2*radius+1, 2*radius+1), color = 'black')

    draw = ImageDraw.Draw(img)

    img_centre = (radius, radius)
    outer_circle_radius = int(radius*.75)
    # Drawing a green circle on the image
    draw.circle(xy = img_centre, radius=outer_circle_radius,
                fill = (255, 255, 255),
                outline = (255, 255, 255),
                width = 1)
    return img

def make_sy_centre_circle_image(radius):
    img = Image.new('RGB', (2*radius+1, 2*radius+1), color = 'black')

    draw = ImageDraw.Draw(img)

    img_centre = (radius, radius)
    # Drawing a green circle on the image
    centre_circle_radius = radius/100
    draw.circle(xy = img_centre, radius=centre_circle_radius,
                fill = (255, 255, 255),
                outline = (255, 255, 255),
                width = 1)
    return img


def make_sy_shape_image(sy_shape, radius, image_centre_xy, fill=(255,255,255), outline=(255,255,255), width=1):
    t_img = Image.new('RGB', (2*radius+1, 2*radius+1), color = 'black')
    t_draw = ImageDraw.Draw(t_img)
    sy_shape.pil_draw(t_draw, image_centre_xy, fill=fill, outline=outline, width=width)
    return t_img

def to_image_coords(point, tx=0, ty=0):
    x, y = point
    return x + tx, ty - y


def generate_sy_png_outline(radius, sy_shapes: Iterable[Shape]):
    img = Image.new('RGB', (int(2*radius+1), int(2*radius+1)), color = 'white')
    draw = ImageDraw.Draw(img)
    fill = None
    outline = (0, 0, 0)
    width = 1
    image_centre_xy = (radius, radius)

    for s in sy_shapes:
        s.pil_draw(draw, image_centre_xy, fill=fill, outline=outline, width=width)

    img.save(f'sy-{radius}.png')
    

def generate_sy_png_gif(radius, sy_shapes):
    blank_image = Image.new('RGB', (int(2*radius+1), int(2*radius+1)), color = 'black')
    image_centre_xy = (radius, radius)
    fill = (255, 255, 255)
    outline = (255, 255, 255)
    width = 1
    shape_images = [
        make_sy_shape_image(sh, radius, image_centre_xy,
                            fill=fill, outline=outline, width=width)
        for sh in sy_shapes
    ]

    sub_image_list = [blank_image] + shape_images
    sub_image_list_converted = [ximg.convert("1") for ximg in sub_image_list]

    combined_image = None
    gif_frames = sub_image_list_converted[:]
    # xor_image.show()
    combine_func = ImageChops.logical_xor
    for image_to_xor in sub_image_list_converted:
        combined_image = combine_func(combined_image, image_to_xor) if combined_image else image_to_xor
        # xor_image.show()
        gif_frames.append(combined_image)

    # xor_result_image.show()
    gif_frames.append(combined_image)

    combined_image.save(f'sy-{radius}-filled.png')

    gif_frames[0].save(f'sy-{radius}-filled.gif', save_all = True, append_images = gif_frames[1:],
                        optimize = False, duration = 1000, loop=0)


def generate_sy_svg(radius, sy_shapes):
    image_centre_xy = (radius, radius)
    shape_tags = [sh.svg_draw(f'shape{i+1}', image_centre_xy, fill="blue", stroke="blue") for i, sh in enumerate(sy_shapes)]

    format_params = {
        'outer_width': 2*radius+10, 
        'outer_height': 2*radius+10,
        'inner_width': 2*radius,
        'inner_height': 2*radius,
        'out_in_padding': 5,
        'shape_tags': '\n'.join(shape_tags),
    }
    
    template_svg_str = pathlib.Path('sy-svg-template.svg').read_text()
    svg_str = template_svg_str.format_map(format_params)
    pathlib.Path(f'sy-{radius}.svg').write_text(svg_str)

    shape_outline_tags = [sh.svg_draw(f'shape{i+1}', image_centre_xy, fill="transparent", stroke="blue") for i, sh in enumerate(sy_shapes)]
    format_params = {
        'outer_width': 2*radius+10, 
        'outer_height': 2*radius+10,
        'inner_width': 2*radius,
        'inner_height': 2*radius,
        'out_in_padding': 5,
        'shape_tags': '\n'.join(shape_outline_tags),
    }
    template_svg_str = pathlib.Path('sy-svg-outline-template.svg').read_text()
    svg_str = template_svg_str.format_map(format_params)
    pathlib.Path(f'sy-{radius}-outline.svg').write_text(svg_str)



def main():
    radius = 512
    sy_shapes = build_sy_shapes(radius, show_turtle=False, show_turtle_intermediate_steps=False)
    generate_sy_png_outline(radius, sy_shapes)
    generate_sy_png_gif(radius, sy_shapes)
    generate_sy_svg(radius, sy_shapes)


if __name__ == '__main__':
    main()

