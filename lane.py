"""Lane detection functions."""
from typing import Optional
import cv2 as cv
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

    plt.subplot(121),plt.imshow(image[:,:,::-1]),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blurred_image[:,:,::-1]),plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()
    return blurred_image

# Game plan for lane detection:

# Blur the image using guassian filter or something like that.

# Once I've done this, maybe filter out any regions that aren't that yellow
# color? I.e., use some basic kind of thresholding to just filter for that
# yellow color.

# THEN, call find_contours on that?

# Read up on some online guides first--or maybe just play around with these
# functions too.

if __name__ == "__main__":
    lane_image = load_image(Path(r"./resource/lane.png"))
    blurred_image = blur_image(image=lane_image, blur_radius=11)