from __future__ import division
import os
from numpy import *
from PIL import Image
from dataclasses import dataclass
from matplotlib import pyplot
plt = pyplot
from matplotlib.path import Path
import matplotlib.patches as patches
from tqdm import tqdm
import io
import math
from svgpathtools import parse_path

@dataclass
class StrokeSettings:
    line_width: int = 5
    canvas_size: int = 1000
    imsize: int = 6
    edgecolor: str = 'black'
    fps: int = 15
    stroke_duration_multiplier: float = 0.75
    min_stroke_duration: float = 0.1

def get_verts_and_codes(svg_list):
    '''
    parse into x,y coordinates and output list of lists of coordinates

    '''
    lines = []
    Verts = []
    Codes = []
    for stroke_ind, stroke in enumerate(svg_list):
        x = []
        y = []
        parsed = parse_path(stroke)
        for i, p in enumerate(parsed):
            if i != len(parsed) - 1:  # last line segment
                x.append(p.start.real)
                y.append(p.start.imag)
            else:
                x.append(p.start.real)
                y.append(p.start.imag)
                x.append(p.end.real)
                y.append(p.end.imag)
        lines.append(zip(x, y))
        verts, codes = polyline_pathmaker(lines)
        Verts.append(verts)
        Codes.append(codes)
    return Verts, Codes


def make_svg_list(stroke_recs):
    '''
    grab sample drawing's strokes and make a list of svg strings from it
    '''
    svg_list = []
    for single_stroke in stroke_recs:
        svg_string = single_stroke['svg']
        svg_list.append(svg_string)

    return svg_list


def render_and_save(Verts,
                    Codes,
                    stroke_settings: StrokeSettings = StrokeSettings(),
                    save_dir=os.getcwd(),
                    base_filename='strokes'):
    '''
    input:
        line_width: how wide of strokes do we want? (int)
        imsize: how big of a picture do we want? (setting the size of the figure)
        canvas_size: original canvas size on tablet?
        out_path: where do you want to save your images? currently hardcoded below.
    output:
        rendered sketches into nested directories

    '''
    verts = Verts[0]
    codes = Codes[0]
    for i, verts in enumerate(Verts):
        codes = Codes[i]
        fig = plt.figure(figsize=(stroke_settings.imsize, stroke_settings.imsize), frameon=False)
        ax = plt.subplot(111)
        ax.axis('off')
        ax.set_xlim(0, stroke_settings.canvas_size)
        ax.set_ylim(0, stroke_settings.canvas_size)

        # remove padding for xaxis and y axis
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        # remove further paddings
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)

        ### render sketch so far
        if len(verts) > 0:
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', edgecolor=stroke_settings.edgecolor, lw=stroke_settings.line_width)
            ax.add_patch(patch)
            plt.gca().invert_yaxis()  # y values increase as you go down in image

        ## save out as png
        ## maybe to make it not render every single thing, use plt.ioff
        os.makedirs(save_dir, exist_ok=True)
        # saving stroke count and not index so i+1
        fname = '{}_{}.png'.format(base_filename, i+1)
        filepath = os.path.join(save_dir, fname)
        fig.savefig(filepath, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)


def polyline_pathmaker(lines):
    x = []
    y = []

    codes = [Path.MOVETO]
    for i, l in enumerate(lines):
        l = list(l)  # materialize so len() works
        for _i, _l in enumerate(l):
            x.append(_l[0])
            y.append(_l[1])
            if _i < len(l) - 1:
                codes.append(Path.LINETO)
            else:
                if i != len(lines) - 1:
                    codes.append(Path.MOVETO)
    verts = list(zip(x, y))  # also make this a list if reused
    return verts, codes


def path_renderer(verts, codes):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    if len(verts) > 0:
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=2)
        ax.add_patch(patch)
        ax.set_xlim(0, 1638)
        ax.set_ylim(0, 1638)
        ax.axis('off')
        plt.gca().invert_yaxis()  # y values increase as you go down in image
        plt.show()
    else:
        ax.set_xlim(0, 1638)
        ax.set_ylim(0, 1638)
        ax.axis('off')
        plt.show()
    plt.savefig()
    plt.close()


def flatten(x):
    return [val for sublist in x for val in sublist]

def create_stroke_animation_gif(stroke_recs, output_path, stroke_settings:StrokeSettings=StrokeSettings()):
    """
    Create a GIF animation showing strokes being drawn over time based on actual stroke timing.
    """
    if not stroke_recs:
        return
    
    # Get SVG data and parse into verts/codes
    svg_list = make_svg_list(stroke_recs)
    Verts, Codes = get_verts_and_codes(svg_list)
    
    if not Verts:
        return
    
    # Calculate timing for each stroke
    stroke_timings = []
    for i, strec in enumerate(stroke_recs):
        start_time = strec.get('startTime', 0)
        end_time = strec.get('submitTime', start_time + 1000)  # Default 1 second
        import builtins

        duration = builtins.max(stroke_settings.min_stroke_duration,
                        (end_time - start_time) / 1000.0 * stroke_settings.stroke_duration_multiplier)
        stroke_timings.append(duration)
    
    # Calculate total animation time and frame count
    total_duration = sum(stroke_timings)
    total_frames = math.ceil(total_duration * stroke_settings.fps) + 1
    frame_dt = 1.0 / stroke_settings.fps
    
    # Store frames for GIF creation
    frames = []
    
    # Track drawing progress
    current_time = 0.0
    current_stroke = 0
    stroke_progress = 0.0  # 0.0 to 1.0 for current stroke
    
    for frame in tqdm(range(total_frames), desc="Creating animation frames"):
        # Create matplotlib figure
        fig = plt.figure(figsize=(6, 6), facecolor='white', dpi=100)
        ax = fig.add_subplot(111)
        ax.set_xlim(0, stroke_settings.canvas_size)
        ax.set_ylim(0, stroke_settings.canvas_size)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.patch.set_facecolor('white')
        
        # Draw all completed strokes
        for stroke_idx in range(current_stroke):
            if stroke_idx < len(Verts):
                verts = Verts[stroke_idx]
                codes = Codes[stroke_idx]
                if len(verts) > 0:
                    path = Path(verts, codes)
                    patch = patches.PathPatch(path, facecolor='none', 
                                            edgecolor=stroke_settings.edgecolor, lw=stroke_settings.line_width)
                    ax.add_patch(patch)
        
        # Draw partial current stroke
        if current_stroke < len(Verts) and stroke_progress > 0:
            verts = Verts[current_stroke]
            codes = Codes[current_stroke]
            
            if len(verts) > 0:
                # Calculate how many vertices to show based on progress
                num_verts_to_show = builtins.max(1, int(len(verts) * stroke_progress))
                partial_verts = verts[:num_verts_to_show]
                partial_codes = codes[:num_verts_to_show]
                
                if len(partial_verts) > 0:
                    # Adjust the last code if needed
                    if len(partial_codes) > 0:
                        partial_codes = list(partial_codes)
                        if partial_codes[-1] != Path.MOVETO:
                            partial_codes[-1] = Path.LINETO
                    
                    path = Path(partial_verts, partial_codes)
                    patch = patches.PathPatch(path, facecolor='none', 
                                            edgecolor=stroke_settings.edgecolor, lw=stroke_settings.line_width)
                    ax.add_patch(patch)
        
        plt.gca().invert_yaxis()  # Match drawing coordinate system
        plt.tight_layout(pad=0)
        
        # Convert matplotlib figure to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor='white', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img.copy())
        plt.close(fig)
        buf.close()
        
        # Update timing and stroke progress
        current_time += frame_dt
        
        # Find which stroke we should be drawing
        elapsed_time = 0.0
        for stroke_idx, duration in enumerate(stroke_timings):
            if elapsed_time + duration > current_time:
                current_stroke = stroke_idx
                stroke_progress = (current_time - elapsed_time) / duration
                break
            elapsed_time += duration
        else:
            # All strokes complete
            current_stroke = len(Verts) - 1
            stroke_progress = 1.0
    
    # Save as GIF with optimization
    if frames:
        # Add a pause at the end to show final result
        for _ in range(stroke_settings.fps):  # 1 second pause
            # And when repeating last frame
            frames.append(frames[-1].copy())
        
        duration_ms = int(1000 / stroke_settings.fps)  # Convert to milliseconds per frame
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,  # Loop forever
            optimize=True,  # Optimize for smaller file size
            quality=95
        )
        print(f"GIF animation saved to {output_path}")
    
    # Clean up
    for frame in frames:
        frame.close()