"""Module with functions for camera calibration and 3D measurement."""

from typing import Optional, Tuple
import cv2 as cv
from matplotlib.backend_bases import MouseButton
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt


# Referencing https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# for details on the calibration process.

def compute_camera_matrix(calibration_image_dir: Path,
                          chessboard_horizontal_inner_corners: int,
                          chessboard_vertical_inner_corners: int,
                          corner_point_refining_window_size: Optional[int] = 11) -> Tuple[np.ndarray, np.ndarray]:
    """Computes a camera's camera matrix (K) and its distortion coefficients
    provided a directory of chessboard calibration images.

    Args:
        calibration_image_dir (Path): Directory containing at least 10
        chessboard calibration images.
        chessboard_horizontal_inner_corners (int): The number of horizontal inner
        corners.
        chessboard_vertical_inner_corners (int): The number of vertical inner
        corners.
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
    
    # All images are 960x540 == widthxheight == 960 columns, 540 rows.

    # Define other calibration parameters.
    # Chessboard size: If the chessboard is oriented so that x is to the right
    # of the image and y is down, then the dimensions of the chessboard is said
    # to be the number of vertical inner corners x number horizontal inner
    # corners.
    chessboard_size = (chessboard_vertical_inner_corners, chessboard_horizontal_inner_corners)
    # Also copied from tutorial: Corner point refining steps.
    corner_point_refining_window_size = 11
    corner_point_refining_criteria =  (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    debug_display_corners = True

    # Create lists for resulting 3D points and 2D points.
    world_points = []
    image_points = []

    # Precompute the 3D point array corresponding to each of the hxw corner
    # points found in a single image. The tutorial constructs it rather
    # compactly, so using their code here:
    chessboard_world_points = np.zeros((chessboard_vertical_inner_corners*chessboard_horizontal_inner_corners, 3), np.float32)
    chessboard_world_points[:,:2] = np.mgrid[0:chessboard_vertical_inner_corners,0:chessboard_horizontal_inner_corners].T.reshape(-1,2)

    # For each image:
    for image_path in calibration_image_paths:
        
        # 1. Load the current image into memory.
        image = cv.imread(str(image_path))
        # NOTE: Each image is 960x540 == width, height == columns, rows.
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
                                         patternWasFound=True)
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

def get_camera_matrix(chessboard_horizontal_inner_corners: int,
                      chessboard_vertical_inner_corners: int,
                      output_dir: Optional[Path] = Path.cwd(),
                      calibration_image_dir: Optional[Path] = Path.cwd()/"calibration") -> Tuple[np.ndarray, np.ndarray]:
    """Computes camera intrinsics provided calibration chessboard
    characteristics and calibration images. Will attempt to load existing
    intrinsics file from the specified output directory. If none found, will
    recompute the intrinsics and write them to disk.

    Args:
        chessboard_horizontal_inner_corners (int):The number of inner corners within a
        chessboard. I.e., where the black squares meet inside the chessboard
        along a row.
        chessboard_vertical_inner_corners (int): The number of inner corners
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
                                                                    chessboard_horizontal_inner_corners=chessboard_horizontal_inner_corners,
                                                                    chessboard_vertical_inner_corners=chessboard_vertical_inner_corners)
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
                                                          newImgSize=(width, height))
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
    return undistorted_image
    
def get_camera_height(depth_calibration_image: Path,
                      object_depth_m: float,
                      point_coords: Tuple[int, int],
                      camera_matrix: np.ndarray) -> float:
    # Returns the camera height with respect to the base link frame origin.
    # I.e., the offset in the z-direction going from the base_link origin to the
    # camera frame origin. Assumes the base_link is at the same level as the
    # ground.

    # Attempt to load the provided calibration image.
    try:
        image = load_image(image_filepath=depth_calibration_image)
    except Exception as exc:
        print(f"Failed to load image {depth_calibration_image}. Quitting.")
        raise exc

    # Parse the camera matrix for its intrinsic parameters.
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1,1]
    
    # Basically, have to do the reverse of multiplying each point by the K
    # matrix. I.e., you multiply points in the film plane by K to project them
    # into the pixel plane. For the reverse, have to multiply by the inverse of
    # K (or just do the operations on each element in reverse, I guess). Should
    # verify this.
    # TODO: Try to multiply the points by the inverse of K and see if you get
    # the same result!
    # TODO: Make sure these points are being parsed in the correct order.
    point_x_px, point_y_px = point_coords
    
    point_x_film = (point_x_px - cx)/fx
    point_y_film = (point_y_px - cy)/fy

    point_x_cam = point_x_film*object_depth_m
    point_y_cam = point_y_film*object_depth_m
    point_z_cam = object_depth_m

    # Camera height is the y component.
    return point_y_cam

    # The upshot of this function is probably to figure out the extrinsics of
    # the camera with respect to the car origin. I.e., if we observe something
    # in the camera frame (have a 3D camera point), we are going to want to be
    # able to project that into 3D vehicle frame coordinates (I.e., from the
    # perspective of the car / base_link frame). There is technically a rotation
    # between these frames (as cameras/opencv conventionally use xyz not as
    # RHR), but the main thing is that there is ONLY a translation component
    # along the car's z-axis.

def get_point_coords_in_image(image: np.ndarray) -> Tuple[int, int]:
    """Returns the x,y pixel coordinates of where the mouse was clicked in the
    provided image.

    Args:
        image (np.ndarray): Image as ndarray.

    Returns:
        Tuple[int, int]: (x,y) pixel coordinates.
    """

    # Show image on plot.
    image_plot = plt.imshow(image[:,:,::-1])

    # Set up the click event listener.
    def on_click(event):
        if event.button is MouseButton.LEFT and event.inaxes:
            print(f"Clicked at location x: {event.xdata}, y: {event.ydata}")
    binding_id = plt.connect('button_press_event', on_click)

    plt.show()

    return 0,0

# What I'm debating right now is if I should perform all these transformations
# as matrix operations. 

# I think for the sake of my constrained timeline for this, I'll try to brute
# force implement this first--at least just packing everything in a single
# function. Not as robust/testable, but a quick prototype. If I have time, I'll
# try to construct the transformation matrix and see if I can do things that way
# too.

# def camera_ground_point_from_pixel_point(camera_height_m: float,
#                                          pixel_coords_m: Tuple[int, int]):
#     # Ycam = camera_height in car frame.
#     pass

# def car_ground_point_from_camera_point():
#     # Zcar = 0.
#     pass

def car_ground_point_from_pixel_point(camera_height_m: float,
                                      camera_matrix: np.ndarray,
                                      pixel_coords_px: Tuple[int, int]) -> Tuple[float, float]:
    """Computes the 3D car coordinates with Zcar == 0 for the provided point in
    the image. Will only work for 2D points that correspond to points at the
    same Z level as the base_link of the car (must be at the same height in the
    world as the car frame origin). I.e., the provided points in the image must
    be in the same ground plane as the car! This function only computes the Xcar
    and Ycar components of the point's location, as its Z-component is assumed
    to be zero.

    Args:
        camera_height_m (float): The offset along the Zcar axis from the
        base_link (car frame) origin to the camera origin. I.e., how far up the
        camera's origin is from the car frame origin. Technically, if we're
        measuring to points on the ground, this height should *really* be
        whatever the current height of the camera is with respect to the ground
        (which would include the height of the base_link off the ground), but
        we'll just approximate that the base link origin is at ground level.
        camera_matrix (np.ndarray): The camera matrix array obtained from
        calibration. This is needed to convert/project the provided pixel
        coordinates into the film plane frame, which will then be projected into
        the 3D camera frame.
        pixel_coords_px (Tuple[int, int]): The (x,y) PIXEL COORDINATES of the
        point you want to find the corresponding 3D points for. This MUST be a
        point ON THE GROUND / IN THE GROUND PLANE IN THE IMAGE! Otherwise, the
        resulting Xcar and Ycar components won't be accurate.

    Returns:
        Tuple[float, float]: The Xcar, Ycar location of the point in the car's
        frame.
    """

    # Grab the x and y pixel coordinates from the provided tuple. Also grab the
    # focal length from the camera matrix.
    x_px, y_px = pixel_coords_px
    focal_length_x_px = camera_matrix[0,0]
    focal_length_y_px = camera_matrix[1,1]

    # Create a homogenious coordinate out of the x and y pixel coordinates.
    pixel_coords_homogeneous = np.array([x_px, y_px, 1], np.int32).reshape((3,1))

    # Compute the film plane point by multiplying the pixel coordinate point by
    # the inverse of the camera matrix.
    film_plane_coords_homogeneous = np.matmul(np.linalg.inv(camera_matrix), pixel_coords_homogeneous)
    x_film, y_film = tuple(film_plane_coords_homogeneous[:2,0])

    # Compute the depth using similar triangles and the fact that the height of
    # the camera is equal to the Y-component of the 3D camera point.
    # TODO: pretty sure the units are wrong here. Refactor this to just use the
    # more straightforward approach.
    y_cam_m = camera_height_m
    z_cam_m = y_cam_m*focal_length_y_px/y_film
    x_cam_m = x_film*z_cam_m/focal_length_x_px

    print(f"Pixel coords:\n{pixel_coords_homogeneous}")
    print(f"Film coords:\n{film_plane_coords_homogeneous}")
    # print(f"Extracted film coords: x: {x_film_m}, y: {y_film_m}")
    # print(f"z_")

    return (0.0,0.0)



if __name__ == "__main__":

    # TODO: Add CLI here for parameterizing this calibration script.
    camera_matrix, distortion_coefficients = get_camera_matrix(chessboard_horizontal_inner_corners=6,
                                                               chessboard_vertical_inner_corners=8)

    height = get_camera_height(Path(r"./resource/cone_x40cm.png"),
                                object_depth_m=0.4,
                                point_coords=(661,500),
                                camera_matrix=camera_matrix)
    print(f"Computed height: {height} meters.")
    # Currently returning 0.13879840542827868 meters, or about 13 cm--which
    # seems reasonable.

    # Temporarily, load an image, and get a clicked point within it. Use this to
    # find the point on the cone we want to measure to.


    # image = load_image(Path(r"resource/cone_x40cm.png"))
    # x,y = get_point_coords_in_image(image)

    # TODO: Probably want to make function that gets the camera matrix and then
    # returns it as an object or something like that.
    # OR, rather, should update the get_camera_height function to accept a
    # camera matrix, rather than getting it inside. That would be better design.

    
    
    print(f"Camera matrix: {camera_matrix}")
    print(f"Inverted camera matrix: {np.linalg.inv(camera_matrix)}")

    x_car, y_car = car_ground_point_from_pixel_point(camera_height_m=height,
                                                     camera_matrix=camera_matrix,
                                                     pixel_coords_px=[0,0])
    # Distance to ground plate point == X component == depth.
    print(f"Distance to ground plane point in the x direction: {x_car}")
    