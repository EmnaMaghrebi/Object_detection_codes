conda create --name synthetic_ds opencv numpy matplotlib -y
conda activate synthetic_ds
pip install pyrender

-- To view Scene:
python setup.py

-- Close windows that open (1x Scene viewer and 1x matplotlib display of bounding box)
-- type 'y' or 'n' to generate dataset. Parameters are modifiable in setup.py file

-- See file scene_axes_angles_diagram.jpeg to see how the axes/angles are set up in 
-- the Pyrender scene

-- IMPORTANT!
-- to avoid getting error: AttributeError: module 'pyglet.gl' has no attribute 'xlib'
-- modify line 52 in pyglet_platform.py:

   def make_uncurrent(self):
       import pyglet
       pyglet.gl.xlib.glx.glXMakeContextCurrent(self._window.context.x_display, 0, 0, None)

-- ==> TO ==>:

    def make_uncurrent(self):
        try:
            import pyglet.gl.xlib
            pyglet.gl.xlib.glx.glXMakeContextCurrent(self._window.context.x_display, 0, 0, None)
        except:
            pass

-- Then run again:
python setup.py