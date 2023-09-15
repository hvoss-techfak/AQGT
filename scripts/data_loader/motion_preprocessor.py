import numpy as np


class MotionPreprocessor:
    def __init__(self, skeletons, mean_pose):
        self.skeletons = np.array(skeletons)
        self.mean_pose = np.array(mean_pose).reshape(-1, 3)
        self.filtering_message = "PASS"

    def get(self):
        assert (self.skeletons is not None)

        # filtering
        if self.skeletons != []:
            if self.check_static_motion(verbose=False):
                self.skeletons = []
                self.filtering_message = "motion"

        if self.skeletons != []:
            self.skeletons = self.skeletons.tolist()
            for i, frame in enumerate(self.skeletons):
                assert not np.isnan(self.skeletons[i]).any()  # missing joints

        return self.skeletons, self.filtering_message

    def check_static_motion(self, verbose=False):
        def get_variance(skeleton):
            variance = np.median(np.var(skeleton.flatten()))
            return variance

        left_arm_var = get_variance(self.skeletons)
        right_arm_var = get_variance(self.skeletons)

        th = 0.002  # exclude 16905
        if left_arm_var < th and right_arm_var < th:
            if verbose:
                print('skip - check_static_motion left var {}, right var {}'.format(left_arm_var, right_arm_var))
            return True
        else:
            if verbose:
                print('pass - check_static_motion left var {}, right var {}'.format(left_arm_var, right_arm_var))
            return False

    def check_pose_diff(self, verbose=False):
        diff = np.abs(self.skeletons - self.mean_pose)
        diff = np.mean(diff)

        th = 0.02
        if diff < th:
            if verbose:
                print('skip - check_pose_diff {:.5f}'.format(diff))
            return True
        else:
            if verbose:
                print('pass - check_pose_diff {:.5f}'.format(diff))
            return False

    def check_spine_angle(self, verbose=False):
        def angle_between(v1, v2):
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        angles = []
        for i in range(self.skeletons.shape[0]):
            spine_vec = self.skeletons[i, 1] - self.skeletons[i, 0]
            angle = angle_between(spine_vec, [0, -1, 0])
            angles.append(angle)
        max_d = np.rad2deg(max(angles))
        mean_d = np.rad2deg(np.mean(angles))
        if max_d > 30 or mean_d > 20:  # exclude 4495
            # if np.rad2deg(max(angles)) > 20:  # exclude 8270
            if verbose:
                print('skip - check_spine_angle {:.5f}, {:.5f}'.format(max_d, mean_d))
            return True
        else:
            if verbose:
                print('pass - check_spine_angle {:.5f}, {:.5f}'.format(max_d, mean_d))
            return False
