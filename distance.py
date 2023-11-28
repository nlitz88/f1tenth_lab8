"""Module with functions for camera calibration and 3D measurement."""

from typing import Optional, Tuple
import cv2 as cv
import numpy as np
from pathlib import Path
import pickle


# Referencing https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# for details on the calibration process.

def compute_camera_matrix(calibration_image_dir: Path,
                          chessboard_inside_row_width: int,
                          chessboard_inside_column_height: int,
                          corner_point_refining_window_size: Optional[int] = 11) -> Tuple[np.ndarray, np.ndarray]:
    """Computes a camera's camera matrix (K) and its distortion coefficients
    provided a directory of chessboard calibration images.

    Args:
        calibration_image_dir (Path): Directory containing at least 10
        chessboard calibration images.
        chessboard_inside_row_width (int): The number of inner corners within a
        chessboard. I.e., where the black squares meet inside the chessboard
        along a row.
        chessboard_inside_column_height (int): The number of inner corners
        within a chessboard along a column.
        corner_point_refining_window_size (Optional[int], optional): Window size
        for point corner point refinement. Defaults to 11.

    Raises:
        Exception: Throws exception if it fails anywhere along the way.
        Specifically, if it fails to compute the camera matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing the camera matrix,
        distortion coefficients.
    """

    # Set path to calibration image directory.
    calibration_image_dir = Path(r"./calibration")
    calibration_image_paths = [path for path in calibration_image_dir.iterdir() if path.is_file()]
    
    # Define other calibration parameters.
    chessboard_internal_corners_row_width = 6
    chessboard_internal_corners_column_height = 8
    chessboard_size = (chessboard_internal_corners_row_width, chessboard_internal_corners_column_height)
    # Also copied from tutorial: Corner point refining steps.
    corner_point_refining_window_size = 11
    corner_point_refining_criteria =  (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    debug_display_corners = False

    # Create lists for resulting 3D points and 2D points.
    world_points = []
    image_points = []

    # Precompute the 3D point array corresponding to each of the hxw corner
    # points found in a single image. The tutorial constructs it rather
    # compactly, so using their code here:
    chessboard_world_points = np.zeros((chessboard_internal_corners_row_width*chessboard_internal_corners_column_height, 3), np.float32)
    chessboard_world_points[:,:2] = np.mgrid[0:chessboard_internal_corners_column_height,0:chessboard_internal_corners_row_width].T.reshape(-1,2)

    # For each image:
    for image_path in calibration_image_paths:
        
        # 1. Load the current image into memory.
        image = cv.imread(str(image_path))
        # cv.imshow(image_path.parts[-1], image)
        # cv.waitKey(1000)

        # 2. Attempt to find the corners points in the chessboard image.
        retval, corners = cv.findChessboardCorners(image=image,
                                                   patternSize=chessboard_size)
        
        # 3. If findChessboardCorners was able to locate the corners within the
        #    chessboard image, map those corner points to their 3D world points
        #    and store them away alongside their respective 2D image points.
        if retval == True:

            # Map those corner points to 3D world points based on their relative
            # position to the top-left-most corner point. Because these are all
            # measured from the origin of the world frame, these will be the
            # same for every image that corners are found in. Therefore, just
            # take the 3D point array computed above (that is common to every
            # chessboard image with corners) and add it to the list of 3D world
            # points.
            world_points.append(chessboard_world_points)

            # Refine the 2D image points using cornerSubPix according to
            # tutorial.
            # Create a grayscale version of the image to use with cornerSubPix
            # function.
            image_gray = cv.cvtColor(src=image, code=cv.COLOR_BGR2GRAY)
            refined_image_corner_points = cv.cornerSubPix(image=image_gray, 
                                                          corners=corners,
                                                          winSize=(corner_point_refining_window_size,corner_point_refining_window_size),
                                                          zeroZone=(-1,-1),
                                                          criteria=corner_point_refining_criteria)

            # Add the newly refined 2D corner point to the list of image points.
            image_points.append(refined_image_corner_points)

            # Display corners if enabled.
            if debug_display_corners == True:
                # Draw the refined corners on the the original color image.
                cv.drawChessboardCorners(image=image,
                                         patternSize=chessboard_size,
                                         corners=refined_image_corner_points,
                                         patternWasFound=True)# 
                # Display the updated (drawn on) image.
                cv.imshow(image_path.parts[-1], image)
                cv.waitKey(500)

    cv.destroyAllWindows()

    # Once we have enough (10) 3D-->2D point correspondences from corners in our
    # chessboard calibration images, we can call the calibrateCamera function to
    # obtain the camera matrix == the camera's intrinsic parameters.
    # TODO Why is calibrateCamera expecting the dimensions in reverse order?
    # NOTE Don't need the R, T that this function returns.
    retval, cameraMatrix, distCoeffs, _, _ = cv.calibrateCamera(objectPoints=world_points,
                                                                        imagePoints=image_points,
                                                                        imageSize=image_gray.shape[::-1],
                                                                        cameraMatrix=None,
                                                                        distCoeffs=None)
    
    if retval == False:
        raise Exception(f"Failed to compute camera matrix!")
    
    print(f"Successfully computed camera matrix (K):\n{cameraMatrix}")
    print(f"Successfully computed camera distortion coefficients:\n{distCoeffs}")

    return cameraMatrix, distCoeffs

def get_camera_matrix(chessboard_inside_row_width: int,
                      chessboard_inside_column_height: int,
                      output_dir: Optional[Path] = Path.cwd(),
                      calibration_image_dir: Optional[Path] = Path.cwd()/"calibration") -> Tuple[np.ndarray, np.ndarray]:
    """Computes camera intrinsics provided calibration chessboard
    characteristics and calibration images. Will attempt to load existing
    intrinsics file from the specified output directory. If none found, will
    recompute the intrinsics and write them to disk.

    Args:
        chessboard_inside_row_width (int):The number of inner corners within a
        chessboard. I.e., where the black squares meet inside the chessboard
        along a row.
        chessboard_inside_column_height (int): The number of inner corners
        within a chessboard along a column.
        output_dir (Optional[Path], optional): Directory that intrinsics file
        will be written to. Defaults to directory you invoke this function from
        (your current working directory at runtime).
        calibration_image_dir (Optional[Path], optional): Directory containing
        at least 10 images of the chessboard for calibration. Defaults to
        Path.cwd()/"calibration".

    Raises:
        Exception: If provided output directory does not exist.
        Exception: If provided calibration image directory does not exist.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Camera matrix (K), distortion
        coefficients.
    """
    
    # Check inputs.
    if not output_dir.exists():
        raise Exception(f"Provided output directory {output_dir} doesn't exist!")
    if not calibration_image_dir.exists():
            raise Exception(f"Provided calibration image directory {calibration_image_dir} doesn't exist!")

    # Check the output_dir to see if a camera_matrix pickle file exists.
    file_name = "camera_matrix.pickle"
    camera_matrix_file = Path(output_dir)/file_name

    # If the camera matrix hasn't already been computed/generated, then we'll
    # compute it below.
    if not camera_matrix_file.exists():
        print(f"No existing camera matrix file found in provided output directory {output_dir}. Starting camera calibration and computing camera intrinsics now.")
        camera_matrix, distortion_coefficients = compute_camera_matrix(calibration_image_dir=calibration_image_dir,
                                                                    chessboard_inside_row_width=chessboard_inside_row_width,
                                                                    chessboard_inside_column_height=chessboard_inside_column_height)
        # Write the camera matrix and distortion coefficients to a new pickle
        # file.
        with camera_matrix_file.open(mode='wb') as pickle_file:
            camera_parameters = [camera_matrix, distortion_coefficients]
            pickle.dump(camera_parameters, pickle_file, pickle.HIGHEST_PROTOCOL)
            print(f"Finished writing newly generated camera parameters to file {camera_matrix_file}")
        
        # Return the newly computed values.
        return camera_matrix, distortion_coefficients
    
    # Otherwise, if there is an existing file, read the camera parameters from
    # that and return those.
    else:
        print(f"Existing camera parameters file {camera_matrix_file} found. Loading parameters now.")
        with camera_matrix_file.open(mode='rb') as pickle_file:
            camera_parameters = pickle.load(pickle_file)
            print(f"Successfully loaded camera parameters from {camera_matrix_file}.")
        # Return the parameters loaded from file.
        return camera_parameters[0], camera_parameters[1]
    
def load_image(image_filepath: Path) -> np.ndarray:

    # Check if provided image exists.
    if not image_filepath.exists():
        raise Exception(f"Provided image {image_filepath} doesn't exist!")
    # If it does exist, read in the image.
    image = cv.imread(str(image_filepath))
    return image

def undistort_image(image: np.ndarray,
                    camera_matrix: np.ndarray,
                    distortion_coefficients: np.ndarray) -> np.ndarray:
    # Compute new camera matrix based on the new image. See
    # https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    height, width = image.shape[:2]
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix=camera_matrix,
                                                          distCoeffs=distortion_coefficients,
                                                          imageSize=(width, height),
                                                          alpha=1.0,
                                                          newImgSize=(height, width))
    # Undistort the image using the distortion coefficients and the new camera
    # matrix.
    undistorted_image = cv.undistort(src=image,
                                     cameraMatrix=camera_matrix,
                                     distCoeffs=distortion_coefficients,
                                     dst=None,
                                     newCameraMatrix=new_camera_matrix)
    # Crop the undistorted image.
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]
    return undistort_image
    
def get_camera_height(depth_calibration_image: Path,
                      object_depth_m: float,
                      point_coords: Tuple[int, int]) -> float:
    # Returns the camera height with respect to the base link frame origin.
    # I.e., the offset in the z-direction going from the base_link origin to the
    # camera frame origin. Assumes the base_link is at the same level as the
    # ground.

    # Check if the provided calibration image file exists.
    if not depth_calibration_image.exists():
        raise Exception(f"Provided depth calibration image {depth_calibration_image} doesn't exist!")
    # Read in example image.
    image = cv.imread(str(depth_calibration_image))

    # Undistort

    # Undistort image using distortion coefficients obtained from calibration
    # (if necessary).

    # The upshot of this function is probably to figure out the extrinsics of
    # the camera with respect to the car origin. I.e., if we observe something
    # in the camera frame (have a 3D camera point), we are going to want to be
    # able to project that into 3D vehicle frame coordinates (I.e., from the
    # perspective of the car / base_link frame). There is technically a rotation
    # between these frames (as cameras/opencv conventionally use xyz not as
    # RHR), but the main thing is that there is ONLY a translation component
    # along the car's z-axis.
    pass

if __name__ == "__main__":

    # TODO: Add CLI here for parameterizing this calibration script.
    camera_matrix, dist_coef = get_camera_matrix(chessboard_inside_row_width=6,
                                                 chessboard_inside_column_height=8)
    