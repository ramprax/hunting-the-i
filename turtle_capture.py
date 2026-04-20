from functools import partial
from pathlib import Path
from threading import Event
from concurrent.futures import ThreadPoolExecutor, wait
import turtle

from PIL import Image


def eps_to_png(eps_file, png_file):
    print(f'Converting {eps_file} to png')
    psimage = Image.open(eps_file)
    print(f'{eps_file} WxH:', psimage.width, psimage.height)
    # psimage.resize()
    psimage.save(png_file) # or any other image format that PIL supports.
    print(f'eps_to_png({eps_file}): Done')


class TurtleCapture:
    def __init__(self, turtle, capture_interval, capture_x, capture_y, capture_width, capture_height, capture_filename_prefix, output_dir):
        self._turtle = turtle
        self._canvas = turtle.screen.getcanvas()
        self._capture_interval = int(capture_interval)
        
        self._capture_x = capture_x
        self._capture_y = capture_y
        self._capture_width = capture_width
        self._capture_height = capture_height
        
        self._should_capture = True
        self._running = Event()
        self._running.clear()
        
        self._capture_filename_prefix = capture_filename_prefix
        self._output_dir = Path(output_dir)
        
        self._tpe = ThreadPoolExecutor()
        self._futures = []
       
    def _do_capture(self, capture_count):
        ps_filename = self._output_dir / f'{self._capture_filename_prefix}_capture_{capture_count:04d}.eps'
        # print(capture_x, capture_y, capture_width, capture_height)
        self._canvas.postscript(file=str(ps_filename), x=self._capture_x, y=self._capture_y, width=self._capture_width, height=self._capture_height)
        if self._should_capture:
            self._turtle.screen.ontimer(partial(self._do_capture, capture_count+1), t=self._capture_interval)
        else:
            for cc in range(capture_count+1):
                ps_filename = self._output_dir / f'{self._capture_filename_prefix}_capture_{cc:04d}.eps'
                png_filename = self._output_dir / f'{self._capture_filename_prefix}_capture_{cc:04d}.png'
                ft = self._tpe.submit(eps_to_png, ps_filename, png_filename)
                self._futures.append(ft)
            self._running.clear()

        print('capture_count:', capture_count)

    def stop_capture(self):
        self._should_capture = False

    def start(self):
        if not self._should_capture:
            raise Exception('Error: Cannot start a stopped capture loop')
        if self._running.is_set():
            raise Exception('Error: Cannot start an already running capture loop')
        self._turtle.screen.ontimer(partial(self._do_capture, 0), t=self._capture_interval)
        self._running.set()
    
    def stop(self):
        self.stop_capture()
        while self._running.is_set():
            self._running.wait(0.1)
        wait(self._futures)

        print('Turtle captured PNG frames generated:', len(self._futures))
        print('Use following command to generate video:')
        print(f'''ffmpeg -framerate 24 -i "{self._capture_filename_prefix}_capture_%04d.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p out.mp4''')

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args, **kwargs):
        self.stop()
        print(args)
        print(kwargs)
        return None

