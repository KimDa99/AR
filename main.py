import numpy as np
import cv2 as cv
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GLUT import glutInit

def select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=10, wnd_name='Camera Calibration'):
    # Open a video
    video = cv.VideoCapture(video_file)
    assert video.isOpened()

    # Select images
    img_select = []
    while True:
        # Grab an image from the video
        valid, img = video.read()
        if not valid:
            break

        if select_all:
            img_select.append(img)
        else:
            # Show the image
            display = img.copy()
            cv.putText(display, f'NSelect: {len(img_select)}', (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
            cv.imshow(wnd_name, display)

            # Process the key event
            key = cv.waitKey(wait_msec)
            if key == ord(' '):             # Space: Pause and show corners
                complete, pts = cv.findChessboardCorners(img, board_pattern)
                cv.drawChessboardCorners(display, board_pattern, pts, complete)
                cv.imshow(wnd_name, display)
                key = cv.waitKey()
                if key == ord('\r'):
                    img_select.append(img) # Enter: Select the image
            if key == 27:                  # ESC: Exit (Complete image selection)
                break

    cv.destroyAllWindows()
    return img_select

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    # Find 2D corner points from given images
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
    assert len(img_points) > 0

    # Prepare 3D points of the chess board
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points) # Must be `np.float32`

    # Calibrate the camera
    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)

def render_scene(K, dist_coeff, box_lower, box_upper):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Load camera parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Set up the camera projection matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, 1, 0.1, 100)

    # Set up the camera view matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Apply distortion correction
    map1, map2 = cv.initUndistortRectifyMap(K, dist_coeff, None, None, (640, 480), cv.CV_32FC1)
    glOrtho(0, 640, 0, 480, -1, 1)
    glPushMatrix()
    glTranslatef(320, 240, 0)
    glRotatef(180, 0, 1, 0)
    glScalef(1, -1, 1)

    # Render the 3D box
    glBegin(GL_QUADS)
    glColor3f(1.0, 0.0, 0.0)  # Red
    glVertex3f(*box_lower[0])
    glVertex3f(*box_lower[1])
    glVertex3f(*box_lower[2])
    glVertex3f(*box_lower[3])

    glColor3f(0.0, 1.0, 0.0)  # Green
    glVertex3f(*box_upper[0])
    glVertex3f(*box_upper[1])
    glVertex3f(*box_upper[2])
    glVertex3f(*box_upper[3])

    glColor3f(0.0, 0.0, 1.0)  # Blue
    glVertex3f(*box_lower[0])
    glVertex3f(*box_lower[1])
    glVertex3f(*box_upper[1])
    glVertex3f(*box_upper[0])

    glVertex3f(*box_lower[1])
    glVertex3f(*box_lower[2])
    glVertex3f(*box_upper[2])
    glVertex3f(*box_upper[1])

    glVertex3f(*box_lower[2])
    glVertex3f(*box_lower[3])
    glVertex3f(*box_upper[3])
    glVertex3f(*box_upper[2])

    glVertex3f(*box_lower[3])
    glVertex3f(*box_lower[0])
    glVertex3f(*box_upper[0])
    glVertex3f(*box_upper[3])
    glEnd()

    glPopMatrix()
    glutSwapBuffers()

if __name__ == '__main__':
    video_file = 'data/chessboard.avi'
    board_pattern = (10, 7)
    board_cellsize = 0.025

    img_select = select_img_from_video(video_file, board_pattern)
    assert len(img_select) > 0, 'There are no selected images!'
    rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize)

    # Print calibration results
    print('## Camera Calibration Results')
    print(f'* The number of selected images = {len(img_select)}')
    print(f'* RMS error = {rms}')
    print(f'* Camera matrix (K) = \n{K}')
    print(f'* Distortion coefficient (k1, k2, p1, p2, k3, ...) = {dist_coeff.flatten()}')
        # Define the vertices of the 3D box
    box_lower = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    box_upper = np.array([[0, 0, -1], [1, 0, -1], [1, 1, -1], [0, 1, -1]], dtype=np.float32)

    # Initialize OpenGL
        
    if bool(glutInit):
        glutInit()
    else:
        print("glutInit function is not available")
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(640, 480)
    glutCreateWindow(b'AR Box')

    # Set up OpenGL functions
    glutDisplayFunc(lambda: render_scene(K, dist_coeff, box_lower, box_upper))
    glutIdleFunc(lambda: None)

    # Enter the main loop
    glutMainLoop()
