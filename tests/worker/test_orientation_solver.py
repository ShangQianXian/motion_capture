import math
import unittest
import cv2
import numpy as np
from core import orientations as ori, retarget_math as rm
from backend_worker.orientation_solver import solve_head, solve_foot, FACE_IDS, FACE_MODEL, CAMERA_TO_STANDARD


def angular_error(a,b):
    return math.degrees(rm.quat_angle(rm.quat_mul(rm.quat_conjugate(a),b)))


class OrientationSolver(unittest.TestCase):
    def test_head_yaw_pitch_roll_with_known_camera(self):
        for axis in ((1,0,0),(0,1,0),(0,0,1),(1,2,3)):
            for angle in (-.6,0,.6):
                q=rm.quat_from_axis_angle(axis,angle)
                rotation=np.asarray(CAMERA_TO_STANDARD).T @ np.asarray(rm.matrix3_from_quat(q))
                rvec=cv2.Rodrigues(rotation)[0]
                camera=np.array(((1024.,0,384),(0,1024.,512),(0,0,1)))
                pixels=cv2.projectPoints(np.asarray(FACE_MODEL),rvec,np.array((0.,0.,2.)),camera,None)[0].reshape(-1,2)
                points=np.zeros((133,3))
                for i,p in zip(FACE_IDS,pixels):
                    points[23+i]=(* (p/(768,1024)),.95)
                actual,info=solve_head(points,(768,1024))
                self.assertIsNotNone(actual, info)
                self.assertLess(angular_error(q,actual),5)

    def test_feet_side_back_and_roll(self):
        for side in ('L','R'):
            for yaw,pitch in ((0,0),(.8,.3),(1.57,0),(3.14,.2),(-.7,-.4)):
                q=rm.quat_mul(rm.quat_from_axis_angle((0,0,1),yaw),rm.quat_from_axis_angle((1,0,0),pitch))
                points=np.zeros((133,3))
                ids=(15,17,18,19) if side=='L' else (16,20,21,22)
                for index,p in zip(ids,ori.foot_template(side)):
                    rotated=rm.quat_rotate_vector(q,p)
                    points[index]=((384+rotated[0]*500)/768,(700-rotated[2]*500)/1024,.95)
                prior=rm.quat_from_axis_angle((0,0,1),yaw)
                actual,info=solve_foot(points,side,(768,1024),500,.215,prior)
                self.assertIsNotNone(actual,info)
                self.assertLess(angular_error(q,actual),5,(side,yaw,pitch,info))

    def test_missing_observations_do_not_claim_a_solution(self):
        points=np.zeros((133,3))
        self.assertIsNone(solve_head(points,(768,1024))[0])
        self.assertIsNone(solve_foot(points,'L',(768,1024),500,.215,rm.IDENTITY_QUAT)[0])


if __name__=='__main__':
    unittest.main()
