"""
PnP Pose Estimation - Gate 3D Position from Corner Keypoints
"""
import numpy as np
import cv2

GATE_OPENING_HALF_SIZE=0.2
GATE_CORNERS_3D=np.array([[-0.2,0.2,0.0],[0.2,0.2,0.0],[0.2,-0.2,0.0],[-0.2,-0.2,0.0]],dtype=np.float64)

class GatePoseEstimator:
    def __init__(self,camera_matrix=None,dist_coeffs=None,img_w=640,img_h=480,fov_deg=90.0,gate_size=None):
        if camera_matrix is not None:
            self.camera_matrix=np.array(camera_matrix,dtype=np.float64)
        else:
            fy=(img_h/2.0)/np.tan(np.radians(fov_deg/2.0));fx=fy
            self.camera_matrix=np.array([[fx,0,img_w/2.0],[0,fy,img_h/2.0],[0,0,1]],dtype=np.float64)
        self.dist_coeffs=np.array(dist_coeffs,dtype=np.float64) if dist_coeffs is not None else np.zeros(5,dtype=np.float64)
        if gate_size is not None:
            h=gate_size
            self.object_points=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]],dtype=np.float64)
        else:
            self.object_points=GATE_CORNERS_3D.copy()
        self.img_w=img_w;self.img_h=img_h
    def estimate_pose(self,keypoints_2d,visibility=None,method='default'):
        keypoints_2d=np.array(keypoints_2d,dtype=np.float64)
        if visibility is not None:
            visibility=np.array(visibility);mask=visibility>=1
            if np.sum(mask)<3: return {'success':False,'reason':'Not enough points'}
            obj_pts=self.object_points[mask];img_pts=keypoints_2d[mask]
        else:
            obj_pts=self.object_points;img_pts=keypoints_2d
        n=len(obj_pts)
        flags=cv2.SOLVEPNP_IPPE_SQUARE if n==4 else cv2.SOLVEPNP_ITERATIVE
        try:
            ok,rvec,tvec=cv2.solvePnP(obj_pts.reshape(-1,1,3),img_pts.reshape(-1,1,2),self.camera_matrix,self.dist_coeffs,flags=flags)
        except cv2.error as e: return {'success':False,'reason':str(e)}
        if not ok: return {'success':False,'reason':'solvePnP failed'}
        rvec=rvec.flatten();tvec=tvec.flatten()
        R,_=cv2.Rodrigues(rvec)
        distance=float(np.linalg.norm(tvec))
        proj,_=cv2.projectPoints(obj_pts.reshape(-1,1,3), rvec,tvec,self.camera_matrix,self.dist_coeffs)
        proj=proj.reshape(-1,2)
        err=float(np.mean(np.linalg.norm(proj-img_pts,axis=1)))
        sy=np.sqrt(R[0,0]**2+R[1,0]**2)
        if sy>1e-6: roll=np.arctan2(R[2,1],R[2,2]);pitch=np.arctan2(-R[2,0],sy);yaw=np.arctan2(R[1,0],R[0,0])
        else: roll=np.arctan2(-R[1,2],R[1,1]);pitch=np.arctan2(-R[2,0],sy);yaw=0
        return {'success':True,'rvec':rvec,'tvec':tvec,'distance':distance,'rotation_matrix':R,'euler_angles':np.degrees(np.array([roll,pitch,yaw])),'gate_normal':R[:,2],'reprojection_error':err,'num_points_used':n,'projected_points':proj}
    def draw_pose(self,frame,pose,label=None):
        if not pose['success']: return frame
        ann=frame.copy();rvec=pose['rvec'];tvec=pose['tvec']
        al=0.1;axpts=np.array([[0,0,0],[al,0,0],[0,al,0],[0,0,al]],dtype=np.float64)
        p,_=cv2.projectPoints(axpts,rvec,tvec,self.camera_matrix,self.dist_coeffs)
        p=p.reshape(-1,2).astype(int);o=tuple(p[0])
        cv2.arrowedLine(ann,o,tuple(p[1]),(0,0,255),2)
        cv2.arrowedLine(ann,o,tuple(p[2]),(0,255,0),2)
        cv2.arrowedLine(ann,o,tuple(p[3]),(255,0,0),2)
        cp2,_=cv2.projectPoints(self.object_points,rvec,tvec,self.camera_matrix,self.dist_coeffs)
        c=cp2.reshape(-1,2).astype(int)
        for i in range(4): cv2.line(ann,tuple(c[i]),tuple(c[(i+1)%4]),(0,255,255),2)
        if label is None: label=f'd={pose["distance"]:.2f}m'
        cv2.putText(ann,label,(o[0]-40,o[1]-15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
        return ann
    def update_intrinsics(self,camera_matrix=None,dist_coeffs=None,fov_deg=None):
        if camera_matrix is not None: self.camera_matrix=np.array(camera_matrix,dtype=np.float64)
        elif fov_deg is not None:
            fy=(self.img_h/2.0)/np.tan(np.radians(fov_deg/2.0));fx=fy
            self.camera_matrix=np.array([[fx,0,self.img_w/2.0],[0,fy,self.img_h/2.0],[0,0,1]],dtype=np.float64)
        if dist_coeffs is not None: self.dist_coeffs=np.array(dist_coeffs,dtype=np.float64)

def compute_gate_waypoint(pose,offset_distance=0.0):
    if not pose['success']: return None,None
    return pose['tvec']-pose['gate_normal']*offset_distance,-pose['gate_normal']
