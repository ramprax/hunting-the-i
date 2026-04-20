
import abc
import math
from turtle import Turtle

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

class Text(Shape):
    def __init__(self, txt, xy, font=None, h_align=None, v_align=None, color=None):
        x, y = xy
        self._xy = x, y
        self._text = txt
        self._font = font if font else ('Arial', 8, 'normal')
        self._h_align = h_align if h_align else 'left'
        self._v_align = v_align
        self._color = color

    def turtle_draw(self, tt: Turtle, fill=None):
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
    

class Circle(Shape):
    def __init__(self, centre_xy, radius):
        cx, cy = centre_xy
        self._centre_xy = cx, cy
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    def turtle_draw(self, tt: Turtle, fill=None):
        if not tt:
            return
        cx, cy = self._centre_xy
        tt.teleport(cx+self._radius, cy)
        tt.setheading(90)
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


class Polygon(Shape):
    def __init__(self, points):
        self._points = [(x, y) for x, y in points]

    def __len__(self):
        return len(self._points)

    def __getitem__(self, idx):
        return self._points[idx]    

    def turtle_draw(self, tt: Turtle, fill=None):
        if not tt:
            return
        ox, oy = self._points[0]
        tt.teleport(ox, oy)
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


class BezierCurve(Shape):
    def __init__(self, guide_points, resolution=100):
        self._guide_points = [(x, y) for x, y in guide_points]
        self._resolution = int(resolution) if resolution else 100

    @staticmethod
    def calculate_bezier_curve_points(guide_points, resolution=100):
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

    def turtle_draw(self, tt: Turtle, fill=None):
        if not tt:
            return
        curve_pts = self.curve_points()
        ox, oy = curve_pts[0]
        end_x, end_y = curve_pts[-1]
        tt.teleport(ox, oy)
        
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
            guide_points = [to_image_coords(pt, image_centre_xy) for pt in self._guide_points]
        else:
            guide_points = self._guide_points
        start_x, start_y = guide_points[0]
        guide_points_str = ', '.join([' '.join(xy) for xy in guide_points[1:]])
        svg_bezier_path_str = f'''<path id="{element_id}" d="M {start_x} {start_y} C {guide_points_str}" fill="{fill}" stroke="{stroke}"/>'''
        return svg_bezier_path_str

