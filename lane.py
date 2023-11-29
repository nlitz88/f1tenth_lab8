"""Lane detection functions."""
from typing import Optional
import cv2 as cv
from matplotlib.backend_bases import MouseButton
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

def load_image(image_filepath: Path) -> np.ndarray:

    # Check if provided image exists.
    if not image_filepath.exists():
        raise Exception(f"Provided image {image_filepath} doesn't exist!")
    # If it does exist, read in the image.
    image = cv.imread(str(image_filepath))
    return image

def blur_image(image: np.ndarray,
               blur_radius: Optional[int]=5) -> np.ndarray:
    
    assert blur_radius % 2 == 1
    
    blurred_image = cv.GaussianBlur(src=image,
                                    ksize=(blur_radius, blur_radius),
                                    sigmaX=0)

    return blurred_image

def mask_image(hsv_image: np.ndarray,
               lower_hsv_value: np.ndarray,
               upper_hsv_value: np.ndarray) -> np.ndarray:
    return cv.inRange(src=hsv_image,
                      lowerb=lower_hsv_value,
                      upperb=upper_hsv_value)

# Game plan for lane detection:

# Blur the image using guassian filter or something like that.

# Once I've done this, maybe filter out any regions that aren't that yellow
# color? I.e., use some basic kind of thresholding to just filter for that
# yellow color.

# THEN, call find_contours on that?

# Read up on some online guides first--or maybe just play around with these
# functions too.
#[26, 126, 194]

if __name__ == "__main__":
    lane_image = load_image(Path(r"./resource/lane.png"))
    blurred_image = blur_image(image=lane_image, blur_radius=17)
    hsv_image = cv.cvtColor(src=blurred_image, code=cv.COLOR_BGR2HSV)
    lower_hsv_bound = np.array([21, 42, 70])
    upper_hsv_bound = np.array([37, 200, 200])
    masked_hsv_image = mask_image(hsv_image=hsv_image,
                                  lower_hsv_value=lower_hsv_bound,
                                  upper_hsv_value=upper_hsv_bound)
    
    cv.imshow("Hsv image", hsv_image)
    cv.imshow("Mask", masked_hsv_image)
    cv.waitKey(10000)
    cv.cvtColor(src=masked_hsv_image, code=cv.COLOR_HSV2BGR)

    # plt.subplot(121),plt.imshow(lane_image[:,:,::-1]),plt.title('Original')
    # plt.xticks([]), plt.yticks([])
    # Can't display HSV (expects BGR), so just display blurred BGR image and
    # then get HSV color value at the same indices.



    # plt.imshow(blurred_image[:,:,::-1]),plt.title('Blurred + hsv')
    # # # plt.subplot(122),plt.imshow(cv.cvtColor(src=masked_hsv_image, code=cv.COLOR_HSV2BGR)),plt.title('Masked HSV')
    # plt.xticks([]), plt.yticks([])

    # def on_click(event):
    #     if event.button is MouseButton.LEFT and event.inaxes:
    #         print(f"Clicked at location x: {event.xdata}, y: {event.ydata}")
    #         print(f"HSV color at location: {hsv_image[int(event.ydata), int(event.xdata), :]}")
    # binding_id = plt.connect('button_press_event', on_click)
    # plt.show()

    cv.destroyAllWindows()