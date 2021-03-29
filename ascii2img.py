import numpy as np
import cv2
import time
from PIL import Image, ImageOps, ImageDraw, ImageFont


# image to numpy arr with appropriate size
def preprocess_image(image, char_img_size):
    img = ImageOps.grayscale(Image.fromarray(image))
    img = img.resize(char_img_size)
    np_img = np.array(img)
    return np_img

# gray scale image to numpy arr of chars
def convert(img):
    ascii_chars = ['.', ',', ':', ';', '+', '*', '?', '%', 'S', '#', '@']
    vertical, horizontal = img.shape
    np_img = np.floor(np.array(img) / 255 * 10).astype('uint8')
    asc = []

    for i in range(vertical):
        row_temp = ''.join([ascii_chars[np_img[i, j]] for j in range(horizontal)])
        asc.append(row_temp)
    asc_img = '\n'.join(asc)
    return asc_img

# text to image ((19, 10) px -> 1 char when font_size = 16)
def text_to_image(text, size):
    img_size = (size[0], size[1])
    image_out = Image.new(mode='RGB', size=img_size, color=(0, 0, 0))
    draw = ImageDraw.Draw(image_out)
    font = ImageFont.truetype('fonts/DejaVuSansMono-Bold.ttf', 16)
    draw.multiline_text((0, 0), text, font=font, fill=(255, 255, 255))
    return image_out

def fill_dropped_frame(video_writer, frame, fps, t0, t1):
    time_taken = t1 - t0
    time_per_frame = 1 / fps
    frame_needed = time_taken // time_per_frame - 1
    if time_taken > time_per_frame:
        for _ in range(int(frame_needed)):
            video_writer.write(frame)

# real time webcam capture and convert!
def capture_and_convert(size=(640, 480), resize_factor=1., source_path=0, fps=25., save_path=None):
    cap = cv2.VideoCapture(source_path)
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)

    if size is not None:
        size = (size[0] * resize_factor, size[1] * resize_factor)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    else:
        size = (int(cap.get(3) * resize_factor), int(cap.get(4) * resize_factor))
    char_img_size = (size[0] // 10, size[1] // 19)

    if save_path is not None:
        codec_info_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_path, codec_info_fourcc, fps, size)

    while cap.isOpened():
        t0 = time.time()
        _, frame = cap.read()
        if frame is None:
            break
        gray = preprocess_image(frame, char_img_size)
        converted = convert(gray)
        pil_img = text_to_image(converted, size)
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imshow("ASCII WORLD", cv_img)
        t1 = time.time()

        if save_path is not None:
            video_writer.write(cv_img)
            if source_path == 0:
                fill_dropped_frame(video_writer, cv_img, fps, t0, t1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# test time!
capture_and_convert(size=None, resize_factor=1, source_path='test0.mp4', save_path='/save_path/test1.mp4')
