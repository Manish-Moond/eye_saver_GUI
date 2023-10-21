import brightness
import tkinter
from PIL.Image import preinit
from PIL import Image
import cv2
import PIL.Image
import PIL.ImageTk
import time
import dlib
from numpy.core import shape_base
from numpy.core.fromnumeric import shape
import numpy as np
import imutils
from scipy import spatial
from scipy.spatial import distance as dist
from imutils import face_utils


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.predictor_path = '<folder path where shape_predictor_68_face_landmarks.dat is located>/shape_predictor_68_face_landmarks.dat'
        self.predictor = dlib.shape_predictor(self.predictor_path)
        self.detector = dlib.get_frontal_face_detector()
        self.lStart, self.lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.rStart, self.rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.COUNTER = 0
        self.TOTAL = 0
        self.EYE_AR_CONSEC_FRAMES = 3
        self.EYE_AR_THRESH = 0.24
        # Booline values
        self.dist_Bool = False
        self.ebc_Bool = False
        self.start_Bool = False
        # images for buttons
        distance_btn_image = tkinter.PhotoImage(file='./images/distance.png')
        eyeBlinkCount_btn_image = tkinter.PhotoImage(
            file='./images/eyeCount.png')
        start_btn_image = tkinter.PhotoImage(file="./images/start.png")

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.window.geometry('850x550')
        self.window.configure(bg="white")
        self.canvas = tkinter.Canvas(window)
        self.canvas.place(x=505, y=23)

        # Button for measuring distance
        tkinter.Button(window, image=distance_btn_image, border=0, borderwidth=0, highlightthickness=0,
                       fg='white', command=self.dm).place(x=110, y=110)

        # Button for counting eyeblink
        tkinter.Button(window, image=eyeBlinkCount_btn_image, border=0, borderwidth=0, highlightthickness=0,
                       fg='white', command=self.ebc).place(x=110, y=210)

        # Button for start
        tkinter.Button(window, image=start_btn_image, border=0, borderwidth=0, highlightthickness=0,
                       fg='white', command=self.start).place(x=90, y=380)

        # Button for stop
        tkinter.Button(window, text="Stop", border=0, borderwidth=0, highlightthickness=0, padx=12, pady=6,
                       bg='red', fg='white', command=self.stop).place(x=720, y=500)
        
        # Button for clearing label
        tkinter.Button(window, text='Clear', border=0, borderwidth=0, highlightthickness=0, padx=12, pady=6,
                       bg='red', fg='white', command=self.clear).place(x=800, y=500)

        # Brightness controller 
        v1 = tkinter.DoubleVar()
        v1.set(brightness.current_brightness)
        tkinter.Scale(window, from_=1, to=100, bg="white", label="Brightness",
                      highlightthickness=0, length=200, command=lambda val: brightness.set_brightness(val), variable=v1).place(x=300, y=100)

        tkinter.Scale(window, from_=1, to=100, bg="white", label="Yellow",
                      highlightthickness=0, length=200).place(x=400, y=100)

        # Label for displaing distance
        self.dist_label = tkinter.Label(
            window, text="", pady=12, padx=18,
            bg="#008cff", fg='white')
        self.dist_label.place(x=420, y=380)

        # Label for displaing eye blink count
        self.blink_count_label = tkinter.Label(
            window, text="", pady=12, padx=18,
            bg="#008cff", fg='white')
        self.blink_count_label.place(x=590, y=380)

        self.delay = 1
        self.update()

        self.window.mainloop()

    def dm(self):
        self.dist_Bool = True
        self.ebc_Bool = False
        self.start_Bool = False

    def ebc(self):
        self.dist_Bool = False
        self.ebc_Bool = True
        self.start_Bool = False

    def stop(self):
        self.dist_Bool = False
        self.ebc_Bool = False
        self.start_Bool = False

    def start(self):
        self.dist_Bool = False
        self.ebc_Bool = False
        self.start_Bool = True

    def clear(self):
        self.dist_label['text']  = ""
        self.blink_count_label['text'] = ""
        self.TOTAL = 0

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            newframe = cv2.resize(
                frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
            self.photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(newframe))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        if self.dist_Bool and self.start_Bool is False:
            self.distance(frame)
        if self.ebc_Bool and self.start_Bool is False:
            self.eye_blink_count(frame)
        if self.start_Bool:
            self.distance(frame)
            self.eye_blink_count(frame)

        self.window.after(self.delay, self.update)

    def distance(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            area = cv2.contourArea(
                np.reshape(shape, (68, 1, 2)))
            self.dist_label['text'] = area
            if area > 15000:
                self.dist_label['bg'] = 'red'
            else:
                self.dist_label['bg'] = 'green'

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def eye_blink_count(self, frame):
        leftEAR = 0
        rightEAR = 0
        ear = 0
        leftEye = 0
        rightEye = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR)/2.0
            try:
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(
                    frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(
                    frame, [rightEyeHull], -1, (0, 255, 0), 1)
                if ear < self.EYE_AR_THRESH:
                    self.COUNTER += 1
                else:
                    if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        self.TOTAL += 1
                        self.COUNTER = 0
                        self.blink_count_label['text'] = self.TOTAL
            except Exception as e:
                print(e, 'eye blink count error')


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            frame = cv2.flip(frame, 1)
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (False, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


App(tkinter.Tk(), "Tkinter and OpenCV")
