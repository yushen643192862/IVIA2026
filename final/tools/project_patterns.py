"""Display structured-light patterns fullscreen on a selected Windows display."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scanner.structured_light import pattern_sequence, render_pattern


def enable_dpi_awareness():
    if sys.platform != 'win32':
        return
    user32 = ctypes.WinDLL('user32',use_last_error=True)
    try:
        function = user32.SetProcessDpiAwarenessContext
        function.argtypes = [ctypes.c_void_p]
        function.restype = wintypes.BOOL
        if function(ctypes.c_void_p(-4)):
            return
    except AttributeError:
        pass
    try:
        ctypes.WinDLL('shcore').SetProcessDpiAwareness(2)
    except (AttributeError,OSError):
        user32.SetProcessDPIAware()


def list_displays():
    if sys.platform != 'win32':
        raise RuntimeError('Display enumeration currently supports Windows only')
    user32 = ctypes.WinDLL('user32',use_last_error=True)

    class MonitorInfo(ctypes.Structure):
        _fields_ = [('size',wintypes.DWORD),('monitor',wintypes.RECT),
                    ('work',wintypes.RECT),('flags',wintypes.DWORD),('device',wintypes.WCHAR*32)]

    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL,ctypes.c_void_p,ctypes.c_void_p,
                                      ctypes.POINTER(wintypes.RECT),wintypes.LPARAM)
    user32.GetMonitorInfoW.argtypes = [ctypes.c_void_p,ctypes.POINTER(MonitorInfo)]
    user32.GetMonitorInfoW.restype = wintypes.BOOL
    user32.EnumDisplayMonitors.argtypes = [ctypes.c_void_p,ctypes.c_void_p,callback_type,wintypes.LPARAM]
    user32.EnumDisplayMonitors.restype = wintypes.BOOL
    monitors, errors = [], []

    @callback_type
    def receive(handle, hdc, rect, data):
        info = MonitorInfo(); info.size = ctypes.sizeof(info)
        if not user32.GetMonitorInfoW(handle,ctypes.byref(info)):
            errors.append(ctypes.get_last_error())
            return False
        match = re.search(r'DISPLAY(\d+)$',info.device)
        monitors.append({'id':int(match.group(1)) if match else len(monitors)+1,
                         'device':info.device,'primary':bool(info.flags & 1),
                         'x':info.monitor.left,'y':info.monitor.top,
                         'width':info.monitor.right-info.monitor.left,
                         'height':info.monitor.bottom-info.monitor.top})
        return True

    if not user32.EnumDisplayMonitors(None,None,receive,0) or errors:
        raise ctypes.WinError(errors[0] if errors else ctypes.get_last_error())
    return sorted(monitors,key=lambda m:m['id'])


def projector_canvas_geometry(canvas, monitor):
    """Record the actual display origin, not just the requested monitor geometry."""
    actual = {'x':canvas.winfo_rootx(),'y':canvas.winfo_rooty(),
              'width':canvas.winfo_width(),'height':canvas.winfo_height()}
    expected = {key:monitor[key] for key in actual}
    if actual!=expected:
        raise RuntimeError(f'Projector canvas geometry {actual} differs from display {expected}; stop capture to avoid shifted coordinates')
    return actual


def create_projector_window(monitor, title='Structured light projector — Esc exits'):
    # Conda's Tcl runtime may not be discovered automatically from a venv.
    for variable, directory in [('TCL_LIBRARY','tcl8.6'),('TK_LIBRARY','tk8.6')]:
        candidate = Path(sys.base_prefix)/'Library/lib'/directory
        if candidate.is_dir():
            os.environ.setdefault(variable,str(candidate))
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    root.title(title)
    root.overrideredirect(True)
    root.attributes('-topmost',True)
    root.configure(background='black',cursor='none')
    width,height = monitor['width'],monitor['height']
    canvas = tk.Canvas(root,width=width,height=height,highlightthickness=0,borderwidth=0,
                       background='black',cursor='none')
    canvas.pack(fill='both',expand=True)
    item = canvas.create_image(0,0,anchor='nw')
    root.geometry(f"{width}x{height}+0+0")
    root.update_idletasks()
    root.deiconify()
    root.update_idletasks()
    # SetWindowPos uses physical virtual-desktop coordinates, including negative
    # coordinates. Tk's '-1920' geometry otherwise means distance from right edge.
    user32 = ctypes.WinDLL('user32',use_last_error=True)
    user32.GetParent.argtypes = [ctypes.c_void_p]
    user32.GetParent.restype = ctypes.c_void_p
    user32.SetWindowPos.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,
                                   ctypes.c_int,ctypes.c_int,wintypes.UINT]
    user32.SetWindowPos.restype = wintypes.BOOL
    handle = user32.GetParent(root.winfo_id()) or root.winfo_id()
    if not user32.SetWindowPos(handle,ctypes.c_void_p(-1),monitor['x'],monitor['y'],width,height,0x0040):
        root.destroy()
        raise ctypes.WinError(ctypes.get_last_error())
    # A monitor move can enqueue WM_DPICHANGED/WM_SIZE. Process those messages
    # before checking the canvas; update_idletasks alone misses native events.
    root.update()
    if not user32.SetWindowPos(handle,ctypes.c_void_p(-1),monitor['x'],monitor['y'],width,height,0x0040):
        root.destroy()
        raise ctypes.WinError(ctypes.get_last_error())
    root.update()
    actual = (canvas.winfo_width(),canvas.winfo_height())
    if actual != (width,height):
        root.destroy()
        raise RuntimeError(f'Fullscreen canvas {actual} differs from display {(width,height)}; check DPI/display settings')
    try:
        projector_canvas_geometry(canvas,monitor)
    except Exception:
        root.destroy()
        raise
    return root,canvas,item


def play(monitor, frames, args, output):
    import tkinter as tk
    root,canvas,item = create_projector_window(monitor)
    width,height = monitor['width'],monitor['height']
    state = {'index':0,'auto':args.auto,'timer':None,'ready':None,'generation':0}
    events = (output/'display_events.jsonl').open('a',encoding='utf-8',buffering=1)

    def log(event, **extra):
        events.write(json.dumps({'event':event,'utc':datetime.now(timezone.utc).isoformat(),
                                 'monotonic':time.perf_counter(),**extra})+'\n')

    def cancel():
        for key in ('timer','ready'):
            if state[key] is not None:
                root.after_cancel(state[key]); state[key] = None

    def show(index):
        cancel()
        state['generation'] += 1
        generation = state['generation']
        state['index'] = max(0,min(index,len(frames)-1))
        frame = frames[state['index']]
        pixels = render_pattern(frame,width,height,args.cell_size,args.seed)
        # Binary PPM lets Tk display numpy pixels without PIL or resampling.
        rgb = pixels[:,:,None].repeat(3,axis=2)
        photo = tk.PhotoImage(data=f'P6\n{width} {height}\n255\n'.encode()+rgb.tobytes(),format='PPM')
        canvas.itemconfigure(item,image=photo)
        canvas.photo = photo
        root.update_idletasks()
        log('submitted',index=frame['index'],name=frame['name'])
        print(f"[{frame['index']+1}/{len(frames)}] {frame['name']} | auto={state['auto']}",flush=True)

        def ready():
            state['ready'] = None
            if generation != state['generation']:
                return
            log('software_settle_elapsed',index=frame['index'],name=frame['name'])
            print('READY (software timer; camera not triggered)',flush=True)
            if state['auto'] and state['index']<len(frames)-1:
                state['timer'] = root.after(args.interval_ms,lambda:show(state['index']+1))
            elif state['auto']:
                state['auto'] = False
                print('Sequence finished; last frame held. R restarts; Esc exits.',flush=True)
        state['ready'] = root.after(args.settle_ms,ready)

    def toggle(event=None):
        state['auto'] = not state['auto']; show(state['index'])

    def close(event=None):
        cancel(); log('closed'); root.destroy()

    root.bind('<Escape>',close)
    root.bind('<space>',lambda event:show(state['index']+1))
    root.bind('<Right>',lambda event:show(state['index']+1))
    root.bind('<Left>',lambda event:show(state['index']-1))
    root.bind('<BackSpace>',lambda event:show(state['index']-1))
    root.bind('r',lambda event:show(0))
    root.bind('R',lambda event:show(0))
    root.bind('a',toggle)
    root.bind('A',toggle)
    root.protocol('WM_DELETE_WINDOW',close)
    root.focus_force()
    print('Space/Right: next | Left: previous | A: auto | R: restart | Esc: close',flush=True)
    try:
        show(0)
        root.mainloop()
    finally:
        events.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-displays',action='store_true')
    parser.add_argument('--display',type=int,default=2,help='Windows DISPLAY device number, default 2')
    parser.add_argument('--mode',choices=['graycode','random','checkerboard','white','black'],default='graycode')
    parser.add_argument('--cell-size',type=int,help='Pattern cell size in projector pixels; graycode default 1')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--auto',action='store_true',help='Advance once through sequence automatically')
    parser.add_argument('--interval-ms',type=int,default=1000,help='Hold duration after settling, default 1000 ms')
    parser.add_argument('--settle-ms',type=int,default=300,help='Software settling delay, default 300 ms')
    parser.add_argument('--output',type=Path,help='New session/export directory')
    parser.add_argument('--export-only',action='store_true',help='Save PNGs without opening a window')
    parser.add_argument('--width',type=int,help='Export-only image width')
    parser.add_argument('--height',type=int,help='Export-only image height')
    args = parser.parse_args()
    if args.seed<0 or args.interval_ms<1 or args.settle_ms<0:
        parser.error('Seed/settle delay must be nonnegative; interval must be positive')
    if args.cell_size is None:
        args.cell_size = {'graycode':1,'random':8,'checkerboard':64,'white':1,'black':1}[args.mode]
    if args.cell_size<1:
        parser.error('Cell size must be positive')
    enable_dpi_awareness()
    if args.list_displays:
        print(json.dumps(list_displays(),ensure_ascii=False,indent=2)); return
    if args.export_only:
        if not args.width or not args.height or min(args.width,args.height)<1:
            parser.error('--export-only requires positive --width and --height')
        monitor = {'id':None,'width':args.width,'height':args.height}
    else:
        if args.width is not None or args.height is not None:
            parser.error('Fullscreen uses native display resolution; --width/--height are export-only')
        monitors = list_displays()
        monitor = next((m for m in monitors if m['id']==args.display),None)
        if monitor is None:
            parser.error(f'DISPLAY{args.display} is not available. Set Win+P to Extend and run --list-displays. Found: {monitors}')
    output = (args.output or ROOT/'outputs/projector'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory is not empty; choose a new session directory')
    frames = pattern_sequence(monitor['width'],monitor['height'],args.mode,args.cell_size)
    output.mkdir(parents=True,exist_ok=True)
    manifest = {'display':monitor,'mode':args.mode,'cell_size':args.cell_size,'seed':args.seed,
                'width':monitor['width'],'height':monitor['height'],
                'logical_width':math.ceil(monitor['width']/args.cell_size),
                'logical_height':math.ceil(monitor['height']/args.cell_size),
                'settle_ms':args.settle_ms,'interval_ms':args.interval_ms,'frames':frames,
                'note':'White/black reference frames precede graycode frames. Match captures by frame name. '
                       'No camera trigger, projector calibration, depth decoding or hardware timing guarantee.'}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Session:',output,'| Frames:',len(frames),flush=True)
    if args.export_only:
        import cv2
        for frame in frames:
            pixels = render_pattern(frame,monitor['width'],monitor['height'],args.cell_size,args.seed)
            cv2.imencode('.png',pixels)[1].tofile(str(output/(frame['name']+'.png')))
        print('Export complete',flush=True)
    else:
        play(monitor,frames,args,output)


if __name__=='__main__':
    try:
        main()
    except (OSError,ValueError,RuntimeError) as exc:
        print('ERROR:',exc,file=sys.stderr)
        sys.exit(1)
