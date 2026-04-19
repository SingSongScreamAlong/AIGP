"""
Phase 2+3: Gate Corner Keypoint Detection + PnP Pose Estimation
================================================================
Combined inference class that takes a camera frame and returns
gate 3D poses. This is the perception module that feeds into
the state estimator (Phase 4) and controller (Phase 5).

Pipeline:
    Camera Frame -> YOLOv8-pose -> 4 Corner Keypoints -> PnP -> 3D Gate Pose

Usage:
    from detect_keypoints import GateKeypointDetector
    detector = GateKeypointDetector('models/gate_corners_v1.pt')
    results = detector.detect(frame)

Author: Conrad Weeden
Date: April 2026
"""
import time,sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
sys.path.insert(0, str(Path(__file__).parent.parent))
from pnp_pose import GatePoseEstimator

class GateKeypointDetector:
    CORNER_NAMES=['TL','TR','BR','BL']
    CORNER_COLORS=[(0,255,0),(255,0,0),(0,0,255),(255,255,0)]
    def __init__(self,model_path='models/gate_corners_v1.pt',camera_matrix=None,dist_coeffs=None,img_w=640,img_h=480,fov_deg=90.0,conf_threshold=0.5,keypoint_conf_threshold=0.3):
        self.model=YOLO(model_path)
        self.conf_threshold=conf_threshold
        self.kp_conf_threshold=keypoint_conf_threshold
        self.img_w=img_w;self.img_h=img_h
        self.pose_estimator=GatePoseEstimator(camera_matrix=camera_matrix,dist_coeffs=dist_coeffs,img_w=img_w,img_h=img_h,fov_deg=fov_deg)
        print(f'GateKeypointDetector: model={model_path}, {img_w}x{img_h}, FOV={fov_deg}')
    def detect(self,frame,estimate_pose=True):
        results=self.model(frame,conf=self.conf_threshold,verbose=False)
        detections=[]
        for result in results:
            if result.boxes is None or result.keypoints is None: continue
            boxes=result.boxes;kpts_data=result.keypoints.data.cpu().numpy()
            for i in range(len(boxes)):
                x1,y1,x2,y2=boxes.xyxy[i].cpu().numpy().astype(int)
                conf=float(boxes.conf[i])
                gate_kpts=kpts_data[i]
                keypoints_xy=gate_kpts[:,:2];keypoint_confs=gate_kpts[:,2]
                visibility=np.zeros(4,dtype=int)
                for k in range(4):
                    if keypoint_confs[k]>=self.kp_conf_threshold:
                        kx,ky=keypoints_xy[k]
                        visibility[k]=2 if (0<=kx<=self.img_w and 0<=ky<=self.img_h) else 1
                det={'bbox':(int(x1),int(y1),int(x2),int(y2)),'confidence':conf,'keypoints':keypoints_xy,'keypoint_confs':keypoint_confs,'visibility':visibility,'pose':None,'distance':float('inf')}
                if estimate_pose and np.sum(visibility>=1)>=3:
                    pose=self.pose_estimator.estimate_pose(keypoints_xy,visibility=visibility)
                    if pose['success']: det['pose']=pose;det['distance']=pose['distance']
                detections.append(det)
        detections.sort(key=lambda d:d['distance'])
        return detections
    def get_nearest_gate(self,frame):
        detections=self.detect(frame,estimate_pose=True)
        for d in detections:
            if d['pose'] and d['pose']['success']: return d
        return None
    def get_next_gates(self,frame,n=2):
        detections=self.detect(frame,estimate_pose=True)
        return [d for d in detections if d['pose'] and d['pose']['success']][:n]
