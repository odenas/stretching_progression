from dataclasses import dataclass, field
import numpy as np
import time

from dataclasses import dataclass, field
import numpy as np
import time

from dataclasses import dataclass, field
import numpy as np
import time

@dataclass
class SmallCobraPose:
    # Configuration Fields
    elbow_angle_max: float = 100.0   # Tucked elbows, not straight
    min_chest_lift: float = 0.05    # Slight lift compared to full cobra
    hold_target_seconds: int = 5
    
    # State Persistence Fields
    is_holding: bool = False
    start_hold_time: float = 0.0
    current_hold_duration: float = 0.0

    def _calculate_angle(self, a, b, c) -> float:
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180 else angle

    def evaluate(self, landmarks):
        """
        Indices: 11=L Shldr, 13=L Elb, 15=L Wri, 23=L Hip
        """
        # 1. Extract Points
        l_shldr = [landmarks[11].x, landmarks[11].y]
        l_elb   = [landmarks[13].x, landmarks[13].y]
        l_wri   = [landmarks[15].x, landmarks[15].y]
        l_hip   = [landmarks[23].x, landmarks[23].y]

        # 2. Geometry Checks
        elbow_angle = self._calculate_angle(l_shldr, l_elb, l_wri)
        
        # Lift: Shoulder y is smaller than Hip y
        lift_dist = l_hip[1] - l_shldr[1]

        # 3. Rule Evaluation
        # We want elbows bent and chest slightly off the floor
        frame_perfect = (elbow_angle <= self.elbow_angle_max and 
                         lift_dist >= self.min_chest_lift)

        # 4. Persistence Logic
        if frame_perfect:
            if not self.is_holding:
                self.is_holding = True
                self.start_hold_time = time.time()
            else:
                self.current_hold_duration = time.time() - self.start_hold_time
        else:
            self.is_holding = False
            self.start_hold_time = 0.0
            self.current_hold_duration = 0.0

        # 5. Feedback
        if frame_perfect:
            msg = f"SMALL COBRA: HOLDING ({int(self.current_hold_duration)}s)"
        else:
            if elbow_angle > self.elbow_angle_max:
                msg = "Keep your elbows tucked and bent"
            else:
                msg = "Peel your chest slightly off the mat"
            
        return frame_perfect, msg
    


@dataclass
class CobraPose:
    # Configuration Fields
    straight_arm_min: float = 150.0  # Slightly more lenient than Warrior II
    chest_lift_threshold: float = 0.15 # Minimum vertical distance between hip and shoulder
    hold_target_seconds: int = 5
    
    # State Persistence Fields
    is_holding: bool = False
    start_hold_time: float = 0.0
    current_hold_duration: float = 0.0
    start_msg = "Adjust pose to start timer"

    def _calculate_angle(self, a, b, c) -> float:
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180 else angle

    def evaluate(self, landmarks):
        """
        Indices: 11=L Shldr, 13=L Elb, 15=L Wri, 23=L Hip, 7=L Ear
        """
        # 1. Extract Points
        l_shldr = [landmarks[11].x, landmarks[11].y]
        l_elb   = [landmarks[13].x, landmarks[13].y]
        l_wri   = [landmarks[15].x, landmarks[15].y]
        l_hip   = [landmarks[23].x, landmarks[23].y]
        l_ear   = [landmarks[7].x, landmarks[7].y]

        # 2. Geometry Checks
        arm_angle = self._calculate_angle(l_shldr, l_elb, l_wri)
        
        # Vertical lift: Shoulder y should be 'lower' (smaller value) than Hip y
        lift_dist = l_hip[1] - l_shldr[1]
        
        # Neck check: Ear should be above shoulder
        neck_clearance = l_shldr[1] - l_ear[1]

        # 3. Rule Evaluation
        frame_perfect = (arm_angle >= self.straight_arm_min and 
                         lift_dist >= self.chest_lift_threshold and
                         neck_clearance > 0.05)

        # 4. Persistence Logic
        if frame_perfect:
            if not self.is_holding:
                self.is_holding = True
                self.start_hold_time = time.time()
            else:
                self.current_hold_duration = time.time() - self.start_hold_time
        else:
            self.is_holding = False
            self.start_hold_time = 0.0
            self.current_hold_duration = 0.0

        # 5. Feedback
        if frame_perfect:
            msg = f"COBRA: HOLDING ({int(self.current_hold_duration)}s)"
        else:
            if arm_angle < self.straight_arm_min:
                msg = "Straighten your arms"
            elif lift_dist < self.chest_lift_threshold:
                msg = "Lift your chest higher"
            else:
                msg = "Drop your shoulders away from ears"
            
        return frame_perfect, msg

@dataclass
class RaisedArmPose:
    # Configuration Fields
    # In MediaPipe, y=0 is the top of the screen. 
    # A wrist 'higher' than a shoulder means its y-coordinate is smaller.
    height_threshold_offset: float = 0.1 
    hold_target_seconds: int = 3
    
    # State Persistence Fields
    is_holding: bool = False
    start_hold_time: float = 0.0
    current_hold_duration: float = 0.0
    start_msg = "Raise your right arm to test"

    def evaluate(self, landmarks):
        """
        Processes landmarks to check if the Right Arm is raised.
        Index 12: Right Shoulder
        Index 16: Right Wrist
        """
        # 1. Extract Points
        r_shldr_y = landmarks[12].y
        r_wrist_y = landmarks[16].y

        # 2. Check Geometry
        # Is the wrist above the shoulder by at least the offset?
        frame_perfect = r_wrist_y < (r_shldr_y - self.height_threshold_offset)

        # 3. Persistence Logic (Identical to WarriorII for consistency)
        if frame_perfect:
            if not self.is_holding:
                self.is_holding = True
                self.start_hold_time = time.time()
            else:
                self.current_hold_duration = time.time() - self.start_hold_time
        else:
            self.is_holding = False
            self.start_hold_time = 0.0
            self.current_hold_duration = 0.0

        # 4. Return status and feedback
        if frame_perfect:
            if self.current_hold_duration >= self.hold_target_seconds:
                message = f"TEST PASSED! ({int(self.current_hold_duration)}s)"
            else:
                message = f"RAISED: {int(self.current_hold_duration)}s / {self.hold_target_seconds}s"
        else:
            message = self.start_msg

        return frame_perfect, message
    



@dataclass
class WarriorIIPose:
    # Configuration Fields
    straight_arm_min: float = 165.0
    bent_knee_min: float = 80.0
    bent_knee_max: float = 110.0
    hold_target_seconds: int = 5  # Goal for the user
    
    # State Persistence Fields
    is_holding: bool = False
    start_hold_time: float = 0.0
    current_hold_duration: float = 0.0
    start_msg = "Adjust pose to start timer"

    def _calculate_angle(self, a, b, c) -> float:
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180 else angle

    def evaluate(self, landmarks):
        # 1. Extract Points (Standard Warrior II points)
        l_shldr = [landmarks[11].x, landmarks[11].y]
        l_elb   = [landmarks[13].x, landmarks[13].y]
        l_wri   = [landmarks[15].x, landmarks[15].y]
        l_hip   = [landmarks[23].x, landmarks[23].y]
        l_knee  = [landmarks[25].x, landmarks[25].y]
        l_ank   = [landmarks[27].x, landmarks[27].y]

        # 2. Check Geometry
        arm_angle = self._calculate_angle(l_shldr, l_elb, l_wri)
        knee_angle = self._calculate_angle(l_hip, l_knee, l_ank)

        # 3. Determine if current frame is "Perfect"
        frame_perfect = (arm_angle >= self.straight_arm_min and 
                         self.bent_knee_min <= knee_angle <= self.bent_knee_max)

        # 4. Persistence Logic
        if frame_perfect:
            if not self.is_holding:
                # User just entered perfect form
                self.is_holding = True
                self.start_hold_time = time.time()
            else:
                # User is continuing to hold perfect form
                self.current_hold_duration = time.time() - self.start_hold_time
        else:
            # Form broken, reset the timer
            self.is_holding = False
            self.start_hold_time = 0.0
            self.current_hold_duration = 0.0

        # 5. Return status and feedback
        if frame_perfect:
            if self.current_hold_duration >= self.hold_target_seconds:
                message = f"GOAL REACHED! ({int(self.current_hold_duration)}s)"
            else:
                message = f"HOLDING: {int(self.current_hold_duration)}s / {self.hold_target_seconds}s"
        else:
            message = self.start_msg

        return frame_perfect, message


class PoseSmoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.previous_landmarks = None

    def smooth(self, current_landmarks):
        """
        Applies a simplified Bayesian update to smooth landmark coordinates.
        """
        if self.previous_landmarks is None:
            self.previous_landmarks = current_landmarks
            return current_landmarks

        smoothed_landmarks = []
        for curr, prev in zip(current_landmarks, self.previous_landmarks):
            # Create a new landmark-like object with smoothed x, y, z
            # We treat x and y as independent Bayesian updates
            smooth_x = self.alpha * curr.x + (1 - self.alpha) * prev.x
            smooth_y = self.alpha * curr.y + (1 - self.alpha) * prev.y
            smooth_z = self.alpha * curr.z + (1 - self.alpha) * prev.z
            
            # Simple mock-up of the landmark structure for compatibility
            class SmoothedPoint:
                def __init__(self, x, y, z):
                    self.x, self.y, self.z = x, y, z
            
            smoothed_landmarks.append(SmoothedPoint(smooth_x, smooth_y, smooth_z))
        
        self.previous_landmarks = smoothed_landmarks
        return smoothed_landmarks
    


import numpy as np
# Helper class for downstream compatibility
class Point:
    def __init__(self, x, y): self.x, self.y = x, y


class BayesianPoseRefiner:
    def __init__(self, num_landmarks=33, process_noise=0.005):
        # Initialize state as NumPy arrays for vectorized math
        self.mu = np.zeros((num_landmarks, 2))      # [x, y] coordinates
        self.sigma = np.ones((num_landmarks, 2))    # Uncertainty
        self.Q = process_noise                      # Prediction noise
        self.initialized = False

    def refine(self, landmarks_list):
        """
        landmarks_list: The raw list of landmark objects from MediaPipe
        """
        # 1. Convert MediaPipe landmarks to NumPy arrays
        # Shape: (33, 2) for coordinates, (33, 1) for confidence
        coords = np.array([[lm.x, lm.y] for lm in landmarks_list])
        
        # Extract presence/visibility scores as our 'Confidence'
        # We use a default of 0.5 if the attribute is missing
        conf = np.array([[getattr(lm, 'presence', 0.5)] for lm in landmarks_list])
        
        # 2. Prediction Step (Prior)
        self.sigma += self.Q

        # 3. Measurement Uncertainty (Likelihood)
        # R is high when confidence is low
        R = 1.0 - conf + 1e-6

        # 4. Bayesian Update (The Kalman Gain)
        # K = Prior_Uncertainty / (Prior_Uncertainty + Measurement_Uncertainty)
        K = self.sigma / (self.sigma + R)

        if not self.initialized:
            self.mu = coords
            self.initialized = True
        else:
            # Posterior Mean = Prior Mean + Gain * (Measurement - Prior Mean)
            self.mu = self.mu + K * (coords - self.mu)

        # 5. Update Uncertainty (Posterior)
        self.sigma = (1 - K) * self.sigma
        
        refined_points = [Point(p[0], p[1]) for p in self.mu]
        return refined_points, self.sigma.copy()
    




import numpy as np

class AdaptiveBayesianRefiner:
    def __init__(self, num_landmarks=33, q_base=0.001, k_velocity=0.05):
        self.mu = np.zeros((num_landmarks, 2))      
        self.sigma = np.ones((num_landmarks, 2))    
        self.q_base = q_base                        # Noise when still
        self.k = k_velocity                         # How much movement boosts Q
        self.initialized = False

    def refine(self, landmarks_list):
        coords = np.array([[lm.x, lm.y] for lm in landmarks_list])
        conf = np.array([[getattr(lm, 'presence', 0.5)] for lm in landmarks_list])
        
        if not self.initialized:
            self.mu = coords
            self.initialized = True
            return [Point(p[0], p[1]) for p in self.mu], self.sigma

        # 1. Calculate Velocity (Distance moved since last refined state)
        # We use the Euclidean distance for each landmark
        velocity = np.linalg.norm(coords - self.mu, axis=1, keepdims=True)

        # 2. Adaptive Q: More velocity = More Process Noise
        # This makes the filter 'looser' during fast moves
        Q_adaptive = self.q_base + (self.k * (velocity**2))

        # 3. Prediction Step (Prior)
        self.sigma += Q_adaptive

        # 4. Measurement Uncertainty (Likelihood)
        R = 1.0 - conf + 1e-6

        # 5. Bayesian Update (Kalman Gain)
        K = self.sigma / (self.sigma + R)

        # 6. Update Mean and Sigma
        self.mu = self.mu + K * (coords - self.mu)
        self.sigma = (1 - K) * self.sigma

                
        return [Point(p[0], p[1]) for p in self.mu], self.sigma