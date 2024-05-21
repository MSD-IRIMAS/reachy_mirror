# STEP 1: Import the necessary modules.
import asyncio
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp

from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto , goto_async
from reachy_sdk.trajectory.interpolation import InterpolationMode

from scipy.spatial.transform import Rotation as R

# STEP 1 BIS: Define all nessesary functions
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def picture_to_reachy(vector: np.ndarray) -> np.ndarray:
    '''
    Transforms a vertor in the picture coordinate system (x right, y down, z back) to a vector in Reachy's coordinate system. (x front, y left, z up)
    '''
    assert vector.shape == (3,)
    return np.array([-vector[2], vector[0], -vector[1]])

def limitize(vector: np.ndarray, right: bool=False) -> np.ndarray:
    '''
    Puts a vector in Reachy's coordinate system inside Reachy's physical limits.
    '''
    assert vector.shape == (3,)
    x_max=0.6
    x_min=0.2
    z_max=0.6
    z_min=-0.2
    y_max_r=0
    y_min_r=-0.8
    y_max_l=0.8
    y_min_l=0

    if vector[0]>x_max:
        vector[0]=x_max
    if vector[0]<x_min:
        vector[0]=x_min
    if right and vector[1]>y_max_r:
        vector[1]=y_max_r
    if right and vector[1]<y_min_r:
        vector[1]=y_min_r
    if not right and vector[1]>y_max_l:
        vector[1]=y_max_l
    if not right and vector[1]<y_min_l:
        vector[1]=y_min_l
    if vector[2]>z_max:
        vector[2]=z_max
    if vector[2]<z_min:
        vector[2]=z_min
    
    return vector

def limitize_head(vector: np.ndarray) -> np.ndarray:
    '''
    Puts a rotation in Reachy's coordinate system inside Reachy's physical limits.
    '''
    assert vector.shape == (3,)

    min_yaw=-80
    max_yaw=80
    max_pitch=56
    min_pitch=-21
    max_roll=38
    min_roll=-38
    

    if vector[0] < min_yaw:
        vector[0] = min_yaw
    if vector[0] > max_yaw:
        vector[0] = max_yaw
    if vector[1] < min_pitch:
        vector[1] = min_pitch
    if vector[1] > max_pitch:
        vector[1] = max_pitch
    if vector[2] < min_roll:
        vector[2] = min_roll
    if vector[2] > max_roll:
        vector[2] = max_roll
    
    return vector

def flip(vector: np.ndarray, axis: chr ='y') -> np.ndarray:
    '''
    Flips a vector along a given axis.
    '''
    assert vector.shape == (3,)

    if axis=='x':
        vector[0] = -vector[0]
    elif axis=='y':
        vector[1] = -vector[1]
    elif axis=='z':
        vector[2] = -vector[2]
    else:
        raise ValueError('Axis incorect.')
    
    return vector

def find_euler(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> list:
    '''
    Finds the euler angles of the rotation between the reference base and the XYZ base.
    '''
    assert X.shape == (3,)
    assert Y.shape == (3,)
    assert Z.shape == (3,)
    Zxy =  np.sqrt(Z[0] * Z[0] + Z[1] * Z[1])
    if Zxy > 0.0001:
        pre = np.arctan2(Y[0] * Z[1] - Y[1]*Z[0], X[0] * Z[1] - X[1] * Z[0])
        nut = np.arctan2(Zxy, Z[2])
        rot = -np.arctan2(-Z[0], Z[1])
    else:
        pre = 0.
        nut = 0. if Z[2] > 0.0001 else np.pi
        rot = -np.arctan2(X[1], X[0])
    return [pre, nut, rot]

def find_rotation(vector: np.ndarray, reference_vector: np.ndarray, method: str='Quaternions')-> np.ndarray:
    '''
    Finds the rotation matrix between the reference vector and the target vector using either the Euler method, the Rodrigues method or the Quaternions method. (default)
    '''
    assert vector.shape == (3,)
    assert reference_vector.shape == (3,)
    if method.lower()=="euler":
        #Euler Method
        y=np.array([0, vector[2], -vector[1]]) 
        x=np.cross(y,vector)
        return R.from_euler('xyz', find_euler(x, y, vector)).as_matrix()
    elif method.lower()=="rodrigues":
        #Rodrigues Method
        a=np.arccos(np.dot(reference_vector,vector)/(np.linalg.norm(reference_vector)*np.linalg.norm(vector)))
        if np.dot(reference_vector, vector) > 0.999999: #Vectors are colinear
            n=np.array([0,0,0])
        elif np.dot(reference_vector, vector) < -0.999999: #Vectors are opposite
            n=np.array([1,0,0])
        else:
            n=np.cross(reference_vector,vector)/np.linalg.norm(np.cross(reference_vector,vector))
        n_x=np.array([
            [0, -n_l[2], n_l[1]],
            [n_l[2], 0, -n_l[0]],
            [-n_l[1], n_l[0], 0]
        ])
        return np.eye(3)+np.sin(a)*n_x+(1-np.cos(a))*np.matmul(n_x, n_x)
    elif method.lower()=="quaternions":
        #Quaternions Method
        n=np.cross(reference_vector,vector)
        w=np.sqrt((np.linalg.norm(reference_vector)**2)*(np.linalg.norm(vector)**2))+np.dot(reference_vector,vector)
        if np.dot(reference_vector, vector) > 0.999999: #Vectors are colinear
            return R.from_quat([0,0,0,1]).as_matrix()
        elif np.dot(reference_vector, vector) < -0.999999: # Vectors are opposite
            return R.from_quat([1,0,0,0]).as_matrix()
        else:
            return R.from_quat([n[0],n[1],n[2],w]).as_matrix()
    else: 
        raise ValueError("Method not found.")

def create_pose_matrix(rotation: np.array, translation: np.array) -> np.array:
    assert rotation.shape == (3,3)
    assert translation.shape == (3,)
    return np.array([
        [rotation[0][0], rotation[0][1], rotation[0][2], translation[0]],
        [rotation[1][0], rotation[1][1], rotation[1][2], translation[1]],
        [rotation[2][0], rotation[2][1], rotation[2][2], translation[2]],
        [0,0,0,1]
    ])

async def moove():
    right=asyncio.create_task(goto_async({joint: pos for joint,pos in zip(reachy.r_arm.joints.values(), right_joints)}, 0.5, interpolation_mode=InterpolationMode.MINIMUM_JERK))
    left=asyncio.create_task(goto_async({joint: pos for joint,pos in zip(reachy.l_arm.joints.values(), left_joints)}, 0.5, interpolation_mode=InterpolationMode.MINIMUM_JERK))
    await left
    await right

# STEP 1 TER: Parse arguments
parser = argparse.ArgumentParser(prog='Reachy mirror', description='Mirror the movement of the arms of the person in front of the robot.', epilog='To exit the program when running, press q.')
parser.add_argument('-m', '--mirrored', action='store_false', help="Disable the mirror effet. (with this flag, mooving the left arm will moove the robot's left arm instead of the right arm)")
parser.add_argument('-d', '--debug', action='store_true', help="Activate debug mode. (run localy and uses the PC webcam instead of Reachy's)")
parser.add_argument('ip', type=str, help="reachy's IP adress.")
args = parser.parse_args()
mirrored = args.mirrored
debug = args.debug
ip = args.ip

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    num_poses=2)
detector = vision.PoseLandmarker.create_from_options(options)

# SEEP 2 BIS: Connect to the robot
if debug:
    reachy = ReachySDK(host='localhost')
else:
    reachy = ReachySDK(host=ip)
reachy.turn_off_smoothly('reachy')

#STEP 2 TER: Set up constants and global variables
recorded_joints_right = [
    reachy.r_arm.r_shoulder_pitch,
    reachy.r_arm.r_shoulder_roll,
    reachy.r_arm.r_arm_yaw,
    reachy.r_arm.r_elbow_pitch,
    reachy.r_arm.r_forearm_yaw,
    reachy.r_arm.r_wrist_pitch,
    reachy.r_arm.r_wrist_roll,
]

recorded_joints_left = [
    reachy.l_arm.l_shoulder_pitch,
    reachy.l_arm.l_shoulder_roll,
    reachy.l_arm.l_arm_yaw,
    reachy.l_arm.l_elbow_pitch,
    reachy.l_arm.l_forearm_yaw,
    reachy.l_arm.l_wrist_pitch,
    reachy.l_arm.l_wrist_roll,
]

recorded_joints_head = [
    reachy.head.neck_yaw,
    reachy.head.neck_pitch,
    reachy.head.neck_roll
]

x=np.array([1,0,0])
y=np.array([0,1,0])
z=np.array([0,0,1])
last_vector_left = np.zeros(3)
last_vector_right = np.zeros(3)
last_rotation = np.zeros(3)

reachy.turn_on('reachy')
if debug:
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        exit()

print("To quit, press q.")
while True:
    has_to_moove = False
    has_to_moove_head = False
    
    if not debug:
        r_frame = reachy.right_camera.last_frame
    else:
        ret, r_frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
    
    # STEP 3: Load the input image.
    image = mp.Image(mp.ImageFormat.SRGB, r_frame)

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    cv.imshow('Annotated image', annotated_image)

    # If there is a person detected and the wrists (points 15 and 16) are visible...
    if len(detection_result.pose_world_landmarks) > 0 and detection_result.pose_world_landmarks[0][16].presence>0.75 and detection_result.pose_world_landmarks[0][15].presence>0.75:
        #Compute the origin of Reachy's coordinate system (the point between point 11 and 12, the shoulders)
        l_11 = np.array([detection_result.pose_world_landmarks[0][11].x,detection_result.pose_world_landmarks[0][11].y,detection_result.pose_world_landmarks[0][11].z])
        l_12 = np.array([detection_result.pose_world_landmarks[0][12].x,detection_result.pose_world_landmarks[0][12].y,detection_result.pose_world_landmarks[0][12].z])
        origin=(l_11+l_12)/2
        
        #Get the location of the wrists...
        l_16 = np.array([detection_result.pose_world_landmarks[0][16].x,detection_result.pose_world_landmarks[0][16].y,detection_result.pose_world_landmarks[0][16].z])
        l_15 = np.array([detection_result.pose_world_landmarks[0][15].x,detection_result.pose_world_landmarks[0][15].y,detection_result.pose_world_landmarks[0][15].z])
        #And put them into Reachy's coordinate system
        if mirrored:
            l_15_r = flip(limitize(picture_to_reachy(l_16-origin), right=True))
            l_16_r = flip(limitize(picture_to_reachy(l_15-origin)))
        else:
            l_16_r = limitize(picture_to_reachy(l_16-origin), right=True)
            l_15_r = limitize(picture_to_reachy(l_15-origin))
        
        #If there is a significal amount of mouvement, schedule the moove
        if np.linalg.norm(l_15_r-last_vector_left) >= 0.1 or np.linalg.norm(l_16_r-last_vector_right) >= 0.1:
            has_to_moove = True
            last_vector_left = l_15_r
            last_vector_right = l_16_r
        
        l_19 = np.array([detection_result.pose_world_landmarks[0][19].x,detection_result.pose_world_landmarks[0][19].y,detection_result.pose_world_landmarks[0][19].z])
        l_20 = np.array([detection_result.pose_world_landmarks[0][20].x,detection_result.pose_world_landmarks[0][20].y,detection_result.pose_world_landmarks[0][20].z])
        z_l=picture_to_reachy(l_15-l_19) #Left hand Z vector
        z_r=picture_to_reachy(l_16-l_20) #Right hand Z vector

        #Do the same for the head
        l_07 = np.array([detection_result.pose_world_landmarks[0][7].x, detection_result.pose_world_landmarks[0][7].y, detection_result.pose_world_landmarks[0][7].z])
        l_08 = np.array([detection_result.pose_world_landmarks[0][8].x, detection_result.pose_world_landmarks[0][8].y, detection_result.pose_world_landmarks[0][8].z])
        y_head = picture_to_reachy(l_07-l_08)

        #Find the hand rotations
        left_angles = find_rotation(z_l, z)
        right_angles = find_rotation(z_r, z)

        #Find the head rotation
        head_angles = limitize_head(R.from_matrix(find_rotation(y_head, y)).as_euler('xyz', degrees=True))
        #And see if it is significative
        if np.linalg.norm(head_angles-last_rotation) >= 5:
            has_to_moove_head = True
            last_rotation = head_angles
        
        #Compute the pose matrixes
        if not mirrored:
            left_matrix = create_pose_matrix(left_angles, last_vector_left)
            right_matrix = create_pose_matrix(right_angles, last_vector_right)
        else:
            '''
            #Dummy matrixes for debuging, use when something goes worng.
            left_matrix = np.array([
                [0,0,-1,last_vector_left[0]],
                [0,1,0,last_vector_left[1]],
                [1,0,0,last_vector_left[2]],
                [0,0,0,1]
            ])
            right_matrix = np.array([
                [0,0,-1,last_vector_right[0]],
                [0,1,0,last_vector_right[1]],
                [1,0,0,last_vector_right[2]],
                [0,0,0,1]
            ])
            '''
            z_rot = R.from_euler('z', 180, degrees=True).as_matrix()
            left_matrix = create_pose_matrix(z_rot @ right_angles, last_vector_left)
            right_matrix = create_pose_matrix(z_rot @ left_angles, last_vector_right)
        #And calculate the joint positions
        left_joints = reachy.l_arm.inverse_kinematics(left_matrix)
        right_joints = reachy.r_arm.inverse_kinematics(right_matrix)

        #Finaly, moove the robot
        '''
        for joint, pos in zip(recorded_joints_left, left_joints):
            joint.goal_position = pos
        for joint, pos in zip(recorded_joints_right, right_joints):
            joint.goal_position = pos
        '''
        if has_to_moove:
            asyncio.run(moove())
        '''
        if has_to_moove_head:
            for joint, pos in zip(recorded_joints_head, head_angles):
                joint.goal_position = pos
        '''
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
if debug:
    cap.release()
cv.destroyAllWindows()
reachy.turn_off_smoothly('reachy')