import time
from turtle import *
import math
import pathlib
from functools import partial
from typing import Iterable

from concurrent.futures import ThreadPoolExecutor, wait

from PIL import Image, ImageDraw, ImageChops

from gltf_helper import make_gltf, save_gltf_to_file
from shapes import *
from turtle_capture import TurtleCapture

(OUTPUT_DIR := pathlib.Path('output')).mkdir(exist_ok=True)


class Dalam(Polygon, Triangulate3D):
    def __init__(self, n, base_radius, tip_radius,centre_xy=(0, 0)):
        self._n = n
        self._base_radius = base_radius
        self._tip_radius = tip_radius
        self._centre_xy = centre_xy
        super().__init__(self.curve_points())

    def _make_dalam_guide_points(self):
        cx, cy = self._centre_xy
        angle_incr = 2*math.pi/self._n
        half_angle_incr = math.pi/self._n
        dalam_tips_x = [
            (cx + self._tip_radius*math.cos(i*angle_incr)) for i in range(self._n)
        ]
        dalam_tips_y = [
            (cy + self._tip_radius*math.sin(i*angle_incr)) for i in range(self._n)
        ]
        dalam_tips = list(zip(dalam_tips_x, dalam_tips_y))

        dalam_bases_x = [
            (cx + self._base_radius*math.cos(half_angle_incr + i*angle_incr)) for i in range(self._n)
        ]
        dalam_bases_y = [
            (cy + self._base_radius*math.sin(half_angle_incr + i*angle_incr)) for i in range(self._n)
        ]
        dalam_bases = list(zip(dalam_bases_x, dalam_bases_y))

        dalam_intp1_x = [
            (cx + self._base_radius*math.cos(i*angle_incr)) for i in range(self._n)
        ]
        dalam_intp1_y = [
            (cy + self._base_radius*math.sin(i*angle_incr)) for i in range(self._n)
        ]
        dalam_intp1 = list(zip(dalam_intp1_x, dalam_intp1_y))

        dalam_intp2_x = [
            (cx + self._tip_radius*math.cos(half_angle_incr + i*angle_incr)) for i in range(self._n)
        ]
        dalam_intp2_y = [
            (cy + self._tip_radius*math.sin(half_angle_incr + i*angle_incr)) for i in range(self._n)
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
    
    def get_2D_points(self):
        return tuple(self.curve_points())

    def get_2D_triangles(self):
        dalam_tips, dalam_intp1, dalam_intp2, dalam_bases = self._make_dalam_guide_points()

        base_2D_triangles = ConvexPolygon(dalam_bases).get_2D_triangles()
        dalam_beziers = self.make_dalam_beziers()

        dalam_2D_triangles = []
        for i, pt in enumerate(dalam_intp1):
            dalam_points = dalam_beziers[2*i].curve_points() + dalam_beziers[2*i+1].curve_points()
            dalam_2D_triangles.extend(CenteredPolygon(dalam_points, pt).get_2D_triangles())

        return base_2D_triangles + dalam_2D_triangles


SIN_COS_45 = 1/(2**.5)


class BhupuraBoundaryPolygon(Polygon, Triangulate3D):

    def __init__(self, containing_circle_radius, boundary_level=0):
        radius = float(containing_circle_radius)
        assert 0 <= boundary_level <= 2
        boundary_level = int(boundary_level)

        square_pts = (
            (radius * SIN_COS_45, radius * SIN_COS_45),
            (-radius * SIN_COS_45, radius * SIN_COS_45),
            (-radius * SIN_COS_45, -radius * SIN_COS_45),
            (radius * SIN_COS_45, -radius * SIN_COS_45),
        )

        sq_side = 2 * radius * SIN_COS_45
        sin_t = SIN_COS_45 / 2  # (sq_side/4)/radius = 1/2**1.5
        cos_t = (7 ** .5) * SIN_COS_45/ 2  # sqrt(1-(sin_t)^2) = (7**.5)/2**1.5

        right_rectangle_points = (
            (radius * cos_t, radius * sin_t),
            (sq_side / 2, radius * sin_t),
            (sq_side / 2, -radius * sin_t),
            (radius * cos_t, -radius * sin_t),
        )

        top_rectangle_points = tuple(
            mirror_xy(xy) for xy in right_rectangle_points
        )

        left_rectangle_points = tuple(
            mirror_left_right(xy) for xy in right_rectangle_points
        )

        bottom_rectangle_points = tuple(
            mirror_up_down(xy) for xy in top_rectangle_points
        )

        right_neck_x = radius * cos_t - sq_side / 8
        right_neck_y = sq_side / 8

        # boundary thickness
        bt_full = right_neck_x - sq_side / 2
        bt_half = bt_full / 2

        bt = bt_half * boundary_level

        boundary_1_8th = (
            (radius * cos_t - bt, 0),
            (radius * cos_t - bt, sq_side / 4 - bt),
            (right_neck_x + bt, sq_side / 4 - bt),
            (right_neck_x + bt, right_neck_y - bt),
            (sq_side / 2 - bt, right_neck_y - bt),
            (sq_side / 2 - bt, sq_side / 2 - bt),
        )

        polygon = full_polygon_from_1_8th(boundary_1_8th)

        # Remove consecutive repeats
        poly_points = polygon._points[:]
        points = [pt for i, pt in enumerate(poly_points) if ((i == 0) or (poly_points[i-1] != pt))]

        super().__init__(points)

    def get_2D_triangles(self):
        print(f'{self.__class__}.get_2D_triangles()')
        points_all = self.get_2D_points()
        triangles_2D = []

        if points_all[0] == points_all[-1]:
            points_all = points_all[:-1]
        print('Num bhupura polygon points:', len(points_all))

        # For outer-rectangles and necks on 4 sides
        for i in range(4):
            points = points_all[(i*10):]+points_all[((i-1)*10):(i*10)]

            outer_rect_points = points[0:4] + points[-3:]
            outer_rect_minimal = outer_rect_points[1:3] + outer_rect_points[-2:]
            assert len(outer_rect_minimal) == 4

            triangles_2D += [
                outer_rect_minimal[:3],
                outer_rect_minimal[2:4] + outer_rect_minimal[:1]
            ]

            neck_rect_points = points[3:5] + points[-4:-2]
            assert len(neck_rect_points) == 4

            triangles_2D += [
                neck_rect_points[:3],
                neck_rect_points[2:4] + neck_rect_points[:1]
            ]

        # Big inner rectangle
        big_inner_rect = [points_all[5], points_all[15], points_all[25], points_all[35]]
        triangles_2D += [
            big_inner_rect[:3],
            big_inner_rect[2:4] + big_inner_rect[:1]
        ]

        return triangles_2D


class StarryPolygon(Polygon, Triangulate3D):
    def __init__(self, points, outward_vertex_parity=0):
        points = sort_points_by_direction(points, (0, 0))
        super().__init__(points)
        self._outward_vertex_parity = outward_vertex_parity%2

    def get_2D_triangles(self):
        points = self.get_2D_points()

        # Remove consecutuve duplicates
        points = [pt for i, pt in enumerate(points) if ((i==0) or pt != points[i-1])]

        # If first == last, remove last
        if points[0] == points[-1]:
            points = points[:-1]

        num_points = len(points)
        if num_points < 3:
            return
        if num_points == 3:
            return [points[:]]

        triangles_2d = []
        for i, pt in enumerate(points):
            if i%2 == self._outward_vertex_parity:
                continue
            next_pt = points[(i+1)%num_points]
            next_next_pt = points[(i+2)%num_points]
            triangles_2d.append([pt, next_pt, next_next_pt])

        # print("StarryPolygon.Triangles_2D:", triangles_2d)
        return triangles_2d


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
    # boundary_polygon = full_polygon_from_1_8th(one_eigth_polygon)
    boundary_polygon = BhupuraBoundaryPolygon(radius, 0)
    if show_intermediate_steps:
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
    # middle_boundary_polygon = full_polygon_from_1_8th(middle_boundary_1_8th)
    middle_boundary_polygon = BhupuraBoundaryPolygon(radius, 1)
    if show_intermediate_steps:
        middle_boundary_polygon.turtle_draw(turtle)

    inner_boundary_1_8th = (
        (radius*cos_t - bt, 0),
        (radius*cos_t - bt, sq_side/4 - bt),
        (right_neck_x + bt, sq_side/4 - bt),
        (right_neck_x + bt, right_neck_y - bt),
        (sq_side/2 - bt, right_neck_y - bt),
        (sq_side/2 - bt, sq_side/2 - bt),
    )
    # inner_boundary_polygon = full_polygon_from_1_8th(inner_boundary_1_8th)
    inner_boundary_polygon = BhupuraBoundaryPolygon(radius, 2)
    if show_intermediate_steps:
        inner_boundary_polygon.turtle_draw(turtle)

    inner_b_corner = (sq_side/2 - bt, right_neck_y - bt)
    corner_dist = ((sq_side/2 - bt)**2 + (right_neck_y - bt)**2)**0.5
    circle_corner_touch = Circle((0, 0), corner_dist)
    if show_intermediate_steps:
        circle_corner_touch.turtle_draw(turtle)

    # circle_1_radius = sq_side/2 - bt
    circle_1_radius = corner_dist - bt_half
    circle_1 = Circle((0, 0), circle_1_radius)
    if show_intermediate_steps:
        circle_1.turtle_draw(turtle)

    circle_gap = (radius*cos_t - bt) - (sq_side/2)
    circle_2 = Circle((0, 0), circle_1_radius - circle_gap/2)
    if show_intermediate_steps:
        circle_2.turtle_draw(turtle)

    circle_3 = Circle((0, 0), circle_1_radius - circle_gap)
    if show_intermediate_steps:
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


def build_sy_shapes(outer_radius, show_turtle=False, show_turtle_intermediate_steps=False, enable_screen_capture=False) -> Iterable[Shape]:
    turtle = Turtle() if show_turtle else None
    if turtle:
        turtle.home()

    bhupura_shapes = sy_bhupura_shapes(outer_radius, turtle, show_turtle=show_turtle, show_intermediate_steps=show_turtle_intermediate_steps)
    
    bhupura_no_dalams = bhupura_shapes[:-3]
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
    
    circle_points_24 = [(round(radius * math.cos(i*math.pi/12), 6), round(radius * math.sin(i*math.pi/12), 6)) for i in range(24)]

    sy_all_points = list()  # OrderedDict()
    sy_level_points = {
        'level_0': [],  # tuple(circle_points_24),
        'level_1': [],  # tuple(),
        'level_2': [],  # tuple(),
        'level_3': [],  # tuple(),
        'level_4': [],  # tuple(),
        'level_5': [],  # tuple(),
        'level_6': [],  # tuple(),
        'level_7': [],  # tuple(),
        'level_8': [],  # tuple(),
        'level_9': [],  # tuple(),
    }
    
    def _add_sy_point(point, level):
        x, y = point
        _x, _y = float(x), float(y)
        if point not in sy_all_points:
            sy_all_points.append(point)
        pt_idx = sy_all_points.index(point)
        lps = sy_level_points[level]
        if pt_idx not in lps:
            lps.append(pt_idx)
    
    def _add_all_sy_points(points, level):
        for pt in points:
            _add_sy_point(pt, level)
    

    def _sort_point_indices_by_direction(point_idx_list):
        point_idx_list.sort(key=lambda pt_idx: get_point_angle(sy_all_points[pt_idx]))
        
    _add_all_sy_points(circle_points_24[:], 'level_0')

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
    
    # sy_points['level_1'] = (top_point,) + latitude_s_1 + (bottom_point,) + latitude_n_1
    _add_all_sy_points(((top_point,) + latitude_s_1 + (bottom_point,) + latitude_n_1), 'level_1')

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

    # sy_points['level_1'] += (up_tri2_part_left_intx_point, up_tri2_part_right_intx_point)
    _add_all_sy_points((up_tri2_part_left_intx_point, up_tri2_part_right_intx_point), 'level_1')
    
    # sy_points['level_3'] += (up_tri2_part_left_intx_point, up_tri2_part_right_intx_point)
    _add_all_sy_points((up_tri2_part_left_intx_point, up_tri2_part_right_intx_point), 'level_3')

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
    
    # sy_points['level_1'] += (bottom_tri2_left_intx_point, bottom_tri2_right_intx_point)
    _add_all_sy_points((bottom_tri2_left_intx_point, bottom_tri2_right_intx_point), 'level_1')
    
    # sy_points['level_3'] += (bottom_tri2_left_intx_point, bottom_tri2_right_intx_point)
    _add_all_sy_points((bottom_tri2_left_intx_point, bottom_tri2_right_intx_point), 'level_3')

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
    
    # sy_points['level_1'] += (up_triangle4_base_left_point, up_triangle4_base_right_point)
    _add_all_sy_points((up_triangle4_base_left_point, up_triangle4_base_right_point), 'level_1')
    
    # sy_points['level_3'] += (up_triangle4_part[1], up_triangle4_part[2])
    _add_all_sy_points((up_triangle4_part[1], up_triangle4_part[2]), 'level_3')
    
    # sy_points['level_5'] += (up_triangle4_part[1], up_triangle4_part[2])
    _add_all_sy_points((up_triangle4_part[1], up_triangle4_part[2]), 'level_5')

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
    
    # sy_points['level_1'] += (up_triangle2_base_left_point, up_triangle2_base_right_point)
    _add_all_sy_points((up_triangle2_base_left_point, up_triangle2_base_right_point), 'level_1')

    # sy_points['level_1'] += (up_triangle2_base_left_intx_pt1, up_triangle2_base_right_intx_pt1)
    _add_all_sy_points((up_triangle2_base_left_intx_pt1, up_triangle2_base_right_intx_pt1), 'level_1')

    # sy_points['level_3'] += (up_triangle2_base_left_intx_pt1, up_triangle2_base_right_intx_pt1)
    _add_all_sy_points((up_triangle2_base_left_intx_pt1, up_triangle2_base_right_intx_pt1), 'level_3')


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
    
    # sy_points['level_1'] += (down_triangle5_base_left_point, down_triangle5_base_right_point)
    _add_all_sy_points((down_triangle5_base_left_point, down_triangle5_base_right_point), 'level_1')

    # sy_points['level_3'] += (trapezium_top_left_pt, trapezium_top_right_pt)
    _add_all_sy_points((trapezium_top_left_pt, trapezium_top_right_pt), 'level_3')
    # sy_points['level_5'] += (trapezium_top_left_pt, trapezium_top_right_pt)
    _add_all_sy_points((trapezium_top_left_pt, trapezium_top_right_pt), 'level_5')

    up_tri1_left_line = up_triangle1[0], up_triangle1[1]
    top_pt3_lat_left_pt = intersection_point(down_triangle5_left_line, up_tri1_left_line)
    top_pt3_lat_right_pt = mirror_left_right(top_pt3_lat_left_pt)
    top_pt3_lat = top_pt3_lat_left_pt, top_pt3_lat_right_pt

    top_point3 = 0, top_pt3_lat_left_pt[1]

    down_triangle2_base_left_pt =  intersection_point(down_triangle2_left_line, top_pt3_lat)
    down_triangle2_base_right_pt = mirror_left_right(down_triangle2_base_left_pt)
    down_triangle2 = Polygon((bottom_point_2, down_triangle2_base_left_pt, down_triangle2_base_right_pt))
    
    # sy_points['level_1'] += (down_triangle2_base_left_pt, down_triangle2_base_right_pt)
    _add_all_sy_points((down_triangle2_base_left_pt, down_triangle2_base_right_pt), 'level_1')
    
    # sy_points['level_1'] += (top_pt3_lat_left_pt, top_pt3_lat_right_pt)
    _add_all_sy_points((top_pt3_lat_left_pt, top_pt3_lat_right_pt), 'level_1')
    # sy_points['level_3'] += (top_pt3_lat_left_pt, top_pt3_lat_right_pt)
    _add_all_sy_points((top_pt3_lat_left_pt, top_pt3_lat_right_pt), 'level_3')

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
    
    equator_left_intx_pt = intersection_point(up_tri1_left_line, down_tri1_left_line)
    equator_left_intx_pt = (equator_left_intx_pt[0], 0)
    equator_right_intx_pt = mirror_left_right(equator_left_intx_pt)
    
    top_2_lat_left_intx_pt = intersection_point(top_point_2_latitude, up_tri1_left_line)
    top_2_lat_right_intx_pt = mirror_left_right(top_2_lat_left_intx_pt)
    
    bottom_2_lat_left_intx_pt = intersection_point(bottom_point_2_latitude, down_tri1_left_line)
    bottom_2_lat_right_intx_pt = mirror_left_right(bottom_2_lat_left_intx_pt)
    
    _add_all_sy_points(
        [
            equator_left_intx_pt,
            equator_right_intx_pt,
            top_2_lat_left_intx_pt,
            top_2_lat_right_intx_pt,
            bottom_2_lat_left_intx_pt,
            bottom_2_lat_right_intx_pt,
        ],
        'level_1'
    )

    _add_all_sy_points(
        [
            equator_left_intx_pt,
            equator_right_intx_pt,
            top_2_lat_left_intx_pt,
            top_2_lat_right_intx_pt,
            bottom_2_lat_left_intx_pt,
            bottom_2_lat_right_intx_pt,
        ],
        'level_2'
    )
    
    equator_left_2_intx_pt = intersection_point(
        up_triangle2_left_line, down_triangle2_left_line)
    equator_left_2_intx_pt = (equator_left_2_intx_pt[0], 0)
    equator_right_2_intx_pt = mirror_left_right(equator_left_2_intx_pt)
    
    _add_all_sy_points(
        [equator_left_2_intx_pt, equator_right_2_intx_pt],
        'level_3'
    )
    _add_all_sy_points(
        [equator_left_2_intx_pt, equator_right_2_intx_pt],
        'level_4'
    )
    
    _add_all_sy_points(
        (top_point_2, bottom_point_2),
        'level_3'
    )
    
    _add_all_sy_points(
        (top_point3, bottom_point_3, up_tri3_base_left_pt, up_tri3_base_right_pt, down_tri3_base_line_left_pt,
        down_tri3_base_line_right_pt),
        'level_5'
    )
    
    _add_all_sy_points(
        (top_point_4, bottom_point_4),
        'level_7'
    )
    
    _add_sy_point(bottom_point_5, 'level_9')

    top_lat_3_left_3_intx_pt = intersection_point(top_pt3_lat, up_triangle2_left_line)
    top_lat_3_left_3_intx_pt = (top_lat_3_left_3_intx_pt[0], top_point3[1])
    top_lat_3_right_3_intx_pt = mirror_left_right(top_lat_3_left_3_intx_pt)
    
    _add_all_sy_points(
        [top_lat_3_left_3_intx_pt, top_lat_3_right_3_intx_pt],
        'level_3'
    )
    _add_all_sy_points(
        [top_lat_3_left_3_intx_pt, top_lat_3_right_3_intx_pt],
        'level_4'
    )
    
    bot_lat_3_left_3_intx_pt = intersection_point(up_triangle2_base_line_part, down_triangle2_left_line)
    bot_lat_3_left_3_intx_pt = (bot_lat_3_left_3_intx_pt[0], bottom_point_3[1])
    bot_lat_3_right_3_intx_pt = mirror_left_right(bot_lat_3_left_3_intx_pt)
    
    _add_all_sy_points(
        [bot_lat_3_left_3_intx_pt, bot_lat_3_right_3_intx_pt],
        'level_3'
    )
    _add_all_sy_points(
        [bot_lat_3_left_3_intx_pt, bot_lat_3_right_3_intx_pt],
        'level_4'
    )
    
    left_intx = intersection_point(up_tri3_left_line, latitude_n_1)
    left_intx = (left_intx[0], lat_n1_midpoint[1])
    right_intx = mirror_left_right(left_intx)
    _add_all_sy_points([left_intx, right_intx], 'level_5')
    _add_all_sy_points([left_intx, right_intx], 'level_6')
    
    down_tri3_left_line = (down_tri3_base_line_left_pt, bottom_point_3)
    left_intx = intersection_point(down_tri3_left_line, latitude_s_1)
    left_intx = left_intx[0], lat_s1_midpoint[1]
    right_intx = mirror_left_right(left_intx)
    _add_all_sy_points([left_intx, right_intx], 'level_5')
    _add_all_sy_points([left_intx, right_intx], 'level_6')
    
    left_intx = intersection_point(up_tri3_left_line, down_tri3_left_line)
    right_intx = mirror_left_right(left_intx)
    _add_all_sy_points([left_intx, right_intx], 'level_5')
    _add_all_sy_points([left_intx, right_intx], 'level_6')
    
    _add_all_sy_points(
        [down_tri3_base_line_part_left_pt, down_tri3_base_line_part_right_pt],
        'level_5'
    )
    _add_all_sy_points(
        [down_tri3_base_line_part_left_pt, down_tri3_base_line_part_right_pt],
        'level_7'
    )
    
    up_tri3_base_line = (up_tri3_base_left_pt, up_tri3_base_right_pt)
    left_intx = intersection_point(up_tri3_base_line, up_triangle4_left_line)
    right_intx = mirror_left_right(left_intx)
    _add_all_sy_points([left_intx, right_intx], 'level_5')
    _add_all_sy_points([left_intx, right_intx], 'level_7')
    
    
    _add_all_sy_points((down_tri4_base_line_left_pt, down_tri4_base_line_right_pt), 'level_7')
    
    _add_all_sy_points(
        (down_tri4_base_line_part_left_pt, down_tri4_base_line_part_right_pt),
        'level_7')
    _add_all_sy_points(
        (down_tri4_base_line_part_left_pt, down_tri4_base_line_part_right_pt),
        'level_9')
        
    left_intx = intersection_point(down_tri3_base_line_part, up_triangle4_left_line)
    right_intx = mirror_left_right(left_intx)
    _add_all_sy_points([left_intx, right_intx], 'level_7')
    _add_all_sy_points([left_intx, right_intx], 'level_8')
    
    down_tri4_left_line = (down_tri4_base_line_left_pt, bottom_point_4)
    left_intx = intersection_point(up_triangle4_left_line, down_tri4_left_line)
    right_intx = mirror_left_right(left_intx)
    _add_all_sy_points([left_intx, right_intx], 'level_7')
    _add_all_sy_points([left_intx, right_intx], 'level_8')

    left_intx = intersection_point(up_tri3_base_line, down_tri4_left_line)
    right_intx = mirror_left_right(left_intx)
    _add_all_sy_points([left_intx, right_intx], 'level_7')
    _add_all_sy_points([left_intx, right_intx], 'level_8')
    
    final_level_wise_polygons = []
    for level_name, level_point_indices in sy_level_points.items():
        _sort_point_indices_by_direction(level_point_indices)
        if level_name != 'level_0':
            level_num = int(level_name[len('level_'):])
            if level_num % 2 == 1:
                out_vertext_parity = 0 if level_num == 7 else 1
                p = StarryPolygon([sy_all_points[lpidx] for lpidx in level_point_indices],
                                  outward_vertex_parity=out_vertext_parity)
            else:
                p = ConvexPolygon([sy_all_points[lpidx] for lpidx in level_point_indices])
            final_level_wise_polygons.append(p)
    
    final_level_wise_polygons = tuple(final_level_wise_polygons)
    
    final_intersecting_triangles = (
        up_triangle1, down_triangle1, up_triangle4,
        up_triangle2, down_triangle5, down_triangle2,
        up_triangle3, down_triangle3, down_triangle4,
    )
    
    # final_shapes_all = bhupura_no_dalams + (
    #     dalam_circle1, shodasha_dalam,
    #     dalam_circle2, ashta_dalam,
    #     dalam_circle3,) + final_intersecting_triangles + (centre_circle,)
    
    final_shapes_all = bhupura_no_dalams + (
        dalam_circle1, shodasha_dalam,
        dalam_circle2, ashta_dalam,
        dalam_circle3,) + final_level_wise_polygons + (centre_circle,)

    final_shapes_quick = final_level_wise_polygons  # final_intersecting_triangles
    
    final_shapes = final_shapes_all  # final_shapes_quick


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

        xy_val = outer_radius+10
        wh_val = 2*outer_radius + 20
        
        tt_cap = None
        if enable_screen_capture:
            capture_x, capture_y, capture_width, capture_height = -xy_val, -xy_val, wh_val, wh_val
            print(capture_x, capture_y, capture_width, capture_height)
            
            tt_cap = TurtleCapture(
                turtle, capture_interval=100, 
                capture_x=capture_x, capture_y=capture_y,
                capture_width=capture_width, capture_height=capture_height,
                capture_filename_prefix='sy',
                output_dir=OUTPUT_DIR
            )
            

        turtle.clear()
        turtle.teleport(0, 0)

        def draw_final_shapes():
            turtle.clear()
            # turtle.speed(0)
            for s in final_shapes[:]:
                s.turtle_draw(turtle)
                
            # level_fill = 'white'
            print('sy_all_points size:', len(sy_all_points))
            # for p in final_level_wise_polygons:
            #    level_fill = 'white' if level_fill == 'black' else 'black'
            #     p.turtle_draw(turtle, fill=level_fill)
            print('sy_all_points size:', len(sy_all_points))

            turtle.teleport(0, 0)
            if tt_cap:
                turtle.screen.ontimer(tt_cap.stop_capture, t=3000)


        def _show_count_down(seconds_left):
            turtle.clear()
            turtle.write(seconds_left)
            if seconds_left > 1:
                turtle.screen.ontimer(partial(_show_count_down, seconds_left-1), t=1000)
        
        if tt_cap:
            tt_cap.start()
            
        wait_time = 5
        turtle.screen.ontimer(draw_final_shapes, t=wait_time*1000)

        turtle.screen.ontimer(partial(_show_count_down, wait_time-1))

        turtle.screen.mainloop()
        
        if tt_cap:
            tt_cap.stop()
        
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

    img.save(OUTPUT_DIR / f'sy-{radius}.png')
    

def generate_sy_png_gif(radius, sy_shapes, save_individual_gif_frames=False):
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
    gif_frames.append(combined_image)

    combined_image.save(OUTPUT_DIR / f'sy-{radius}-filled.png')

    gif_frames[0].save(OUTPUT_DIR / f'sy-{radius}-filled.gif', save_all = True, append_images = gif_frames[1:],
                        optimize = False, duration = 1000, loop=0)
    if save_individual_gif_frames:
        num_images = len(gif_frames)
        digits = int(math.log10(num_images)+1)
        for frame_idx, frame_image in enumerate(gif_frames):
            frame_image.save(OUTPUT_DIR / f'sy-{radius}-filled-frame-{frame_idx:0>{digits}}.png')
        print('Filled image PNG frames generated:', num_images)
        print('Use following command to generate video:')
        print(f'''ffmpeg -framerate 2 -i "sy-{radius}-filled-frame-%0{digits}d.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p out-{radius}-filled.mp4''')


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
    pathlib.Path(OUTPUT_DIR / f'sy-{radius}.svg').write_text(svg_str)

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
    pathlib.Path(OUTPUT_DIR / f'sy-{radius}-outline.svg').write_text(svg_str)


def main():
    radius = 600
    sy_shapes = build_sy_shapes(radius, show_turtle=False, show_turtle_intermediate_steps=False, enable_screen_capture=False)
    # generate_sy_png_outline(radius, sy_shapes)
    print(sy_shapes)
    generate_sy_png_gif(radius, sy_shapes)
    # generate_sy_svg(radius, sy_shapes)
    
    num_levels = len(sy_shapes)
    level_height = radius/num_levels
    point_store, faces = convert_all_shapes_2d_to_xz3d_faces(sy_shapes[:], level_height, start_y=-150)
    
    # print("3D Point Store:", point_store)
    # print("3D Faces:", faces)

    print("Total 3D Points in store:", len(point_store))
    print("Total 3D Faces:", len(faces))
    print("Levels:", num_levels)
    print("Level height:", level_height)
    
    json_string = generate_vs_fs_js(point_store, faces, 1000)
    pathlib.Path(OUTPUT_DIR / f'sy-3D-{radius}-vs-fs.js').write_text(json_string)

    # Try triangles
    triangulating_shapes = sy_shapes[:]
    triangle_point_store, triangle_faces = convert_all_shapes_to_3d_triangles(triangulating_shapes, level_height, start_y=-150)
    # print("3D Triangle Point Store:", triangle_point_store)
    # print("3D Triangle Faces:", triangle_faces)

    print("Total 3D Triangle Points in store:", len(triangle_point_store))
    print("Total 3D Triangle Faces:", len(triangle_faces))
    print("Levels:", num_levels)
    print("Level height:", level_height)

    json_string = generate_vs_fs_js(triangle_point_store, triangle_faces, 1000)
    pathlib.Path(OUTPUT_DIR / f'sy-3D-triangles-{radius}-vs-fs.js').write_text(json_string)

    gltf_file_path = OUTPUT_DIR / f'sy-3D-triangles-{radius}.gltf'
    gltf = make_gltf(triangle_point_store, triangle_faces, scale_down_factor=1000)
    # print(gltf)
    save_gltf_to_file(gltf, gltf_file_path)


if __name__ == '__main__':
    main()

