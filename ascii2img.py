import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw, ImageFont


# image to numpy arr with appropriate size
def preprocess_image(image, target_size):
    img = ImageOps.grayscale(Image.fromarray(image))
    img = img.resize(target_size)
    np_img = np.array(img)
    return np_img

# gray scale image to numpy arr of chars
def convert(img):
    ascii_chars = ['.', ',', ':', ';', '+', '*', '?', '%', 'S', '#', '@']
    vertical, horizontal = img.shape
    np_img = np.floor(np.array(img) / 255 * 10).astype('uint8')
    asc = []

    # now, it's not so efficient.. but it works!
    for i in range(vertical):
        row_temp = ''.join([ascii_chars[np_img[i, j]] for j in range(horizontal)])
        asc.append(row_temp)
    asc_img = '\n'.join(asc)

    ## faster? not that much...
    # vectorized = np.vectorize(ascii_chars.__getitem__)(np_img)
    # asc_img = '\n'.join([''.join(vectorized[i]) for i in range(len(vectorized))])
    return asc_img

# text to image ((20, 10) px -> 1 char when window size is (600, 600), font_size = 16)
def text_to_image(text, size):
    img_size = (size[1] * 10, size[0] * 19)
    image_out = Image.new(mode='RGB', size=img_size, color=(0, 0, 0))
    draw = ImageDraw.Draw(image_out)
    font = ImageFont.truetype('fonts/DejaVuSansMono-Bold.ttf', 16)
    draw.multiline_text((0, 0), text, font=font, fill=(255, 255, 255))
    return image_out

# real time webcam capture and convert!
def capture_and_convert(size=(75, 25)):
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        gray = preprocess_image(frame, size)
        converted = convert(gray)
        pil_img = text_to_image(converted, tuple(reversed(size)))
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imshow("ASCII WORLD", cv_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cao.release()
    cv2.destroyAllWindows()


# test time!
capture_and_convert()
