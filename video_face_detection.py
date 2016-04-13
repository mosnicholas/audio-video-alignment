import cv2

def add_bounding_box_video_opencv(video_path, video_out_path, cascade_path):
    faceCascade = cv2.CascadeClassifier(cascade_path)
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.cv.CV_CAP_PROP_FPS)
    height = int(video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    width = int(video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.cv.CV_FOURCC(*'mp4v')

    video_writer = cv2.VideoWriter(video_out_path, fourcc, 1, (width, height))

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        video_writer.write(frame)

    # When everything is done, release the capture
    video_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video = 'data/clipped/trump/video/china-059.mp4'
    out = './trial-059.avi'
    cascade_file = 'data/haarcascade_frontalface_default.xml'
    add_bounding_box_video(video, out, cascade_file)