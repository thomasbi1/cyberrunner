import cv2 as cv
import matplotlib

import platform


def init_capture(device, idx_cam, video_path, init_video_frame_idx):

    if device == "CAM":
        osname = platform.platform()
        if osname.startswith("Windows"):
            cap = cv.VideoCapture(idx_cam, cv.CAP_DSHOW)
        elif osname.startswith("Linux"):
            cap = cv.VideoCapture(idx_cam)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1200)
        cap.set(cv.CAP_PROP_FPS, 55)
    elif device == "VIDEO":
        cap = cv.VideoCapture(video_path)
        cap.set(cv.CAP_PROP_POS_FRAMES, init_video_frame_idx)

    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # float `height`

    return cap, width, height


def init_win_subimages():
    cs = [
        (450, 330),
        (650, 330),
        (650, 100),
        (450, 100),
        (550, 200),
    ]  # ul corners coordinates of plate corners subimages windows
    for i, c in enumerate(cs):
        cv.namedWindow("sub_" + str(i))
        cv.moveWindow("sub_" + str(i), c[0] - 200, c[1] + 500)


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == "TkAgg":
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == "WXAgg":
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)
