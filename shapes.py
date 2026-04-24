
import abc
import math
from abc import ABC
from turtle import Turtle

import json

from PIL import ImageDraw


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


class Shape(abc.ABC):
    @staticmethod
    def to_image_coords(xy, image_centre_xy):
        tx, ty = image_centre_xy
        x, y = xy
        return x + tx, ty - y

    @abc.abstractmethod
    def turtle_draw(self, tt: Turtle, fill=None):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def pil_draw(self, img_draw: ImageDraw, image_centre_xy, fill=None, outline=(255, 255, 255), width=1):
        raise NotImplementedError()

    @abc.abstractmethod
    def svg_draw(self, element_id, image_centre_xy, fill, stroke):
        raise NotImplementedError()


class CanDo3D(abc.ABC):
    
    @abc.abstractmethod
    def get_2D_points(self):
        raise NotImplementedError()

    def get_3D_triangles(self, bottom_y, level_height, point_3d_store):
        raise NotImplementedError()


class CenteredShape(CanDo3D, ABC):
    def get_3D_triangles(self, bottom_y, level_height, point_3d_store):
        points_2d = self.get_2D_points()

        if len(points_2d) < 3:
            raise NotImplementedError('Not implemented: Less than 3 points')

        top_y = bottom_y + level_height

        triangles = []

        if len(points_2d) == 3:
            (x1, y1), (x2, y2), (x3, y3) = points_2d

            tp1 = (x1, top_y, y1)
            tp1_idx = get_or_add_3d_point_store_index(tp1, point_3d_store)
            tp2 = (x2, top_y, y2)
            tp2_idx = get_or_add_3d_point_store_index(tp2, point_3d_store)
            tp3 = (x3, top_y, y3)
            tp3_idx = get_or_add_3d_point_store_index(tp3, point_3d_store)

            bp1 = (x1, bottom_y, y1)
            bp1_idx = get_or_add_3d_point_store_index(bp1, point_3d_store)
            bp2 = (x2, bottom_y, y2)
            bp2_idx = get_or_add_3d_point_store_index(bp2, point_3d_store)
            bp3 = (x3, bottom_y, y3)
            bp3_idx = get_or_add_3d_point_store_index(bp3, point_3d_store)

            triangles.append([tp1_idx, tp2_idx, tp3_idx])
            triangles.append([bp3_idx, bp2_idx, bp1_idx])

            triangles.append([bp1_idx, tp2_idx, tp1_idx])
            triangles.append([bp1_idx, bp2_idx, tp2_idx])

            triangles.append([bp2_idx, tp3_idx, tp2_idx])
            triangles.append([bp2_idx, bp3_idx, tp3_idx])

            triangles.append([bp3_idx, tp1_idx, tp3_idx])
            triangles.append([bp3_idx, bp1_idx, tp1_idx])

            return triangles

        num_points = len(points_2d)
        cx3d, cz3d = self._centre_xy
        bottom_center_xyz = cx3d, bottom_y, cz3d
        bottom_center_idx = get_or_add_3d_point_store_index(bottom_center_xyz, point_3d_store)

        top_center_xyz = cx3d, top_y, cz3d
        top_center_idx = get_or_add_3d_point_store_index(top_center_xyz, point_3d_store)

        for i, point in enumerate(points_2d):
            point_x, point_y = point
            next_point_x, next_point_y = points_2d[(i+1)%num_points]

            # Bottom
            bottom_point_3d = point_x, bottom_y, point_y
            cur_bottom_point_idx = get_or_add_3d_point_store_index(bottom_point_3d, point_3d_store)

            next_bottom_point_3d = next_point_x, bottom_y, next_point_y
            next_bottom_point_idx = get_or_add_3d_point_store_index(next_bottom_point_3d, point_3d_store)

            bottom_triangle = [bottom_center_idx, next_bottom_point_idx, cur_bottom_point_idx]
            triangles.append(bottom_triangle)

            # Top
            top_point_3d = point_x, top_y, point_y
            cur_top_point_idx = get_or_add_3d_point_store_index(top_point_3d, point_3d_store)

            next_top_point_3d = next_point_x, top_y, next_point_y
            next_top_point_idx = get_or_add_3d_point_store_index(next_top_point_3d, point_3d_store)

            top_triangle = [top_center_idx, cur_top_point_idx, next_top_point_idx]
            triangles.append(top_triangle)

            # Side
            side_upper_triangle = [cur_bottom_point_idx, next_top_point_idx, cur_top_point_idx]
            side_lower_triangle = [cur_bottom_point_idx, next_bottom_point_idx, next_top_point_idx]

            triangles.append(side_upper_triangle)
            triangles.append(side_lower_triangle)

        return triangles



def make_point3d_store():
    return []

    
def get_or_add_3d_point_store_index(point3D, point3d_store):
    x, y, z = point3D
    x = float(x)
    y = float(y)
    z = float(z)
    point3D = x, y, z
    if point3D not in point3d_store:
        point3d_store.append(point3D)
    return point3d_store.index(point3D)


def convert_2d_shape_to_xz3d_faces(shape: CanDo3D, bottom_y, top_y, point3d_store):
    shape_2D_points = shape.get_2D_points()
    
    point_3d_idx_bottom_list = []
    point_3d_idx_top_list = []
    
    faces = []
    
    for point2D in shape_2D_points:
        x2, y2 = point2D
        x3, y3, z3 = x2, bottom_y, y2
        
        bottom_point_3d_idx = get_or_add_3d_point_store_index((x3, y3, z3), point3d_store)
        point_3d_idx_bottom_list.append(bottom_point_3d_idx)
        
        x3, y3, z3 = x2, top_y, y2
        top_point_3d_idx = get_or_add_3d_point_store_index((x3, y3, z3), point3d_store)
        point_3d_idx_top_list.append(top_point_3d_idx)
    
    faces.append(point_3d_idx_bottom_list)
    faces.append(point_3d_idx_top_list)
    for i, bottom_pt_idx in enumerate(point_3d_idx_bottom_list):
        if i < len(point_3d_idx_bottom_list)-1:
            faces.append([
                bottom_pt_idx, point_3d_idx_bottom_list[i+1],
                point_3d_idx_top_list[i+1], point_3d_idx_top_list[i]
            ])
            
    return faces


def convert_all_shapes_2d_to_xz3d(shapes: list[CanDo3D], level_height, start_y=0):
    point_3d_store = make_point3d_store()
    all_shape_faces = []
    for i, shape in enumerate(shapes):
        bottom_y = start_y + i * level_height
        top_y = bottom_y + level_height
        faces = convert_2d_shape_to_xz3d_faces(shape, bottom_y, top_y, point_3d_store)
        all_shape_faces.extend(faces)
    
    return point_3d_store, all_shape_faces

def convert_all_shapes_to_3d_triangles(shapes: list[CanDo3D], level_height, start_y=0):
    point_3d_store = make_point3d_store()
    all_shape_faces = []
    for i, shape in enumerate(shapes):
        bottom_y = start_y + i * level_height
        top_y = bottom_y + level_height
        try:
            faces = shape.get_3D_triangles(bottom_y, level_height, point_3d_store)
            all_shape_faces.extend(faces)
        except NotImplementedError as nie:
            print(f'Could not triangulate: shapes[{i}] <{type(shape)}>')
            # faces = convert_2d_shape_to_xz3d_faces(shape, bottom_y, top_y, point_3d_store)

    return point_3d_store, all_shape_faces


JS_PROG_STRING = '''
const vs = {vs};

const fs = {fs};

'''

def generate_vs_fs_js(point_3d_store, faces, scale_down_factor=1.0):
    vs = [{'x': x/scale_down_factor, 'y': y/scale_down_factor, 'z': z/scale_down_factor} for x, y, z in point_3d_store]
    fs = faces[:]

    js_string = JS_PROG_STRING.format(vs=json.dumps(vs), fs=json.dumps(fs))
    js_string = js_string.replace('"x"', "x").replace('"y"', "y").replace('"z"', "z")

    return js_string
    

class Text(Shape):
    def __init__(self, txt, xy, font=None, h_align=None, v_align=None, color=None):
        x, y = xy
        self._xy = x, y
        self._text = txt
        self._font = font if font else ('Arial', 8, 'normal')
        self._h_align = h_align if h_align else 'left'
        self._v_align = v_align
        self._color = color

    def turtle_draw(self, tt: Turtle, fill=None, outline=None):
        x, y = self._xy
        if not self._v_align or self._v_align == 'top':
            tt.teleport(x, y)
        else:
            f_sz = self._font[1]
            if self._v_align == 'bottom':
                dy = -f_sz
            elif self._v_align == 'center':
                dy = -f_sz/2
            else:
                raise Exception(f'Bad v_align value: {self._v_align}')
            tt.teleport(x, y + dy)
            
        if self._color:
            old_color = tt.color()[0]
            tt.color(self._color)
        
        tt.write(self._text, align=self._h_align, font=self._font)
        
        if self._color:
            tt.color(old_color)
    
    def pil_draw(self, img_draw: ImageDraw, image_centre_xy, fill=None, outline=(255, 255, 255), width=1):
        raise NotImplementedError()

    def svg_draw(self, element_id, image_centre_xy, fill, stroke):
        raise NotImplementedError()
    

class Circle(Shape, CenteredShape):
    def __init__(self, centre_xy, radius):
        cx, cy = centre_xy
        self._centre_xy = cx, cy
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    def turtle_draw(self, tt: Turtle, fill=None, outline=None):
        if not tt:
            return
        cx, cy = self._centre_xy
        tt.teleport(cx+self._radius, cy)
        tt.setheading(90)
        
        if outline:
            tt.pencolor(outline)

        if fill:
            tt.fillcolor(fill)
            tt.begin_fill()

        tt.circle(self._radius)

        if fill:
            tt.end_fill()
    
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
    
    def get_2D_points(self):
        cx, cy = self._centre_xy
        circle_points_48 = tuple([
            (
                self._radius * math.cos(i*math.pi/24) + cx,
                self._radius * math.sin(i*math.pi/24) + cy
            ) for i in range(48)
        ])
        return circle_points_48



class Polygon(Shape, CanDo3D):
    def __init__(self, points):
        self._points = [(x, y) for x, y in points]

    def __len__(self):
        return len(self._points)

    def __getitem__(self, idx):
        return self._points[idx]    

    def turtle_draw(self, tt: Turtle, fill=None, outline=None):
        if not tt:
            return
        ox, oy = self._points[0]
        tt.teleport(ox, oy)
        if outline:
            tt.pencolor(outline)
        if fill:
            tt.fillcolor(fill)
            tt.begin_fill()
        
        for x, y in self._points[1:]:
            tt.goto(x, y)
        tt.goto(ox, oy)
        
        if fill:
            tt.end_fill()

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

    def get_2D_points(self):
        return tuple(self._points[:])


def get_point_angle(point, center_xy=(0, 0)):
    cx, cy = center_xy
    px, py = point
    x = px - cx
    y = py - cy
    if y == 0:
        if x >= 0:  # Right +ve x-axis
            return 0
        return math.pi  # Left -ve x-axis
    if x == 0:
        if y > 0:  # Up +ve y-axis
            return math.pi/2
        return 3*math.pi/2 # Down -ve y-axis
    angle = math.atan(abs(y/x))

    if x > 0 and y < 0:  # 4th quadrant
        return 2*math.pi - angle
    if x < 0 and y > 0:  # 2nd quadrant
        return math.pi - angle
    if x < 0 and y < 0:  # 3rd quadrant
        return math.pi + angle

    return angle  # 1st quandrant


def sort_points_by_direction(point_list, center_xy=(0, 0)):
    return sorted(point_list, key=lambda pt: get_point_angle(pt, center_xy))


class CenteredPolygon(Polygon, CenteredShape):
    def __init__(self, points, centre_xy= (0, 0)):
        cx, cy = centre_xy
        self._centre_xy = float(cx), float(cy)
        points = sort_points_by_direction(
            [(float(x), float(y)) for x, y in points], center_xy=self._centre_xy)
        super().__init__(points)


class BezierCurve(Shape):
    DEFAULT_RESOLUTION = 10
    def __init__(self, guide_points, resolution=DEFAULT_RESOLUTION):
        self._guide_points = [(x, y) for x, y in guide_points]
        self._resolution = int(resolution) if resolution else self.DEFAULT_RESOLUTION

    @staticmethod
    def calculate_bezier_curve_points(guide_points, resolution=DEFAULT_RESOLUTION):
        n = len(guide_points) - 1
        coeffs = [math.comb(n, i) for i in range(n+1)]
        curve_points = []
        for tn in range(resolution+1):
            t = tn/resolution
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
        return self.calculate_bezier_curve_points(guide_points, self._resolution)

    def turtle_draw(self, tt: Turtle, fill=None, outline=None):
        if not tt:
            return
        curve_pts = self.curve_points()
        ox, oy = curve_pts[0]
        end_x, end_y = curve_pts[-1]
        tt.teleport(ox, oy)
        
        if outline:
            tt.pencolor(outline)
        
        if fill and ox == end_x and oy == end_y:
            tt.fillcolor(fill)
            tt.begin_fill()
            
        for x, y in curve_pts[1:]:
            tt.goto(x, y)
        
        if fill and ox == end_x and oy == end_y:
            tt.end_fill()

    def pil_draw(self, img_draw: ImageDraw, image_centre_xy, fill=None, outline=(255, 255, 255), width=1):
        curve_points = self.curve_points(to_image_coords_ox_oy=image_centre_xy)
        img_draw.line(curve_points, fill=fill, width=width)
    
    def svg_draw(self, element_id, image_centre_xy, fill, stroke):
        if image_centre_xy:
            guide_points = [super().to_image_coords(pt, image_centre_xy) for pt in self._guide_points]
        else:
            guide_points = self._guide_points
        start_x, start_y = guide_points[0]
        guide_points_str = ', '.join([' '.join(xy) for xy in guide_points[1:]])
        svg_bezier_path_str = f'''<path id="{element_id}" d="M {start_x} {start_y} C {guide_points_str}" fill="{fill}" stroke="{stroke}"/>'''
        return svg_bezier_path_str

