import cv2
import numpy as np
import io

def read_and_process (image_byte, path: str):
    file_bytes = np.asarray(bytearray(io.BytesIO(image_byte).read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (640, 640), interpolation = cv2.INTER_AREA)
    edges = cv2.Canny(img, 100, 150)
    cv2.imwrite(path, edges)