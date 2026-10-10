import torch


@torch.no_grad()
def anchor_in_frustum(anchors, intrinsics, pose, h, w, cam_center = None, distance_lis = None, levels = None, level_index:int = None):
    '''
        input
            anchors: torch.tensor([N, 3], device = 'cuda')  
                    A tensor containing N anchors' position in 3D space
            intrinsics: torch.tensor([3, 3], device = 'cuda') 
                or torch.tensor([4], device = 'cuda') [fx, fy, cx, cy]  
                    Camera intrinsic parameters as a matrix or a vector
            pose: torch.tensor([4, 4], device = 'cuda') 
                    Camera pose as a transformation matrix or a vector
            level_index: int
            levels: torch.tensor([N], device = 'cuda') 
            h: int   Image height
            w: int   Image width
            depth_require: bool  Whether return depth map or not 

        return
            mask: torch.tensor([N], device = 'cuda')  
                    A mask indicating whether each point is within the camera frustum
    '''

    anchors = anchors.float()
    intrinsics = intrinsics.float()
    pose = pose.float()

    if len(intrinsics.shape) == 1:
        fx, fy, cx, cy = intrinsics
        intrinsics = torch.tensor([[fx, .0, cx],
                        [.0, fy, cy],
                        [.0, .0, 1.0]], device=intrinsics.device)
    
    points_cam = torch.mm(pose[:3, :3], anchors.t()) + pose[:3, 3].unsqueeze(1)
    
    points_img = torch.mm(intrinsics, points_cam)

    # 1e-6 is to prevent division by zero
    points_img = points_img / (points_img[2, :] + 1e-6)

    zs = points_cam[2, :]

    visible_mask = torch.bitwise_and(points_img[0, :] >= 0, points_img[0, :] <= w)
    visible_mask.bitwise_and_(points_img[1, :] >= 0)
    visible_mask.bitwise_and_(points_img[1, :] <= h)
    visible_mask.bitwise_and_(zs > 0)

    mask = torch.zeros([anchors.shape[0],], dtype=torch.bool, device='cuda')

    if distance_lis is not None:
        max_level = len(distance_lis)

        point_dist = torch.norm(anchors - cam_center, dim=-1)


        for level in range(max_level+1):
            if level_index is not None:
                level = level_index

            level_mask = torch.ones([anchors.shape[0],], dtype=torch.bool, device='cuda')

            if level == 0:
                level_mask.bitwise_and_(point_dist < distance_lis[0])
            elif level == max_level:
                level_mask.bitwise_and_(point_dist >= distance_lis[max_level-1])
            else:
                level_mask.bitwise_and_(point_dist < distance_lis[level])
                level_mask.bitwise_and_(point_dist >= distance_lis[level-1])

            level_mask.bitwise_and_(levels == level)

            level_mask.bitwise_and_(visible_mask)

            mask.bitwise_or_(level_mask)

            if level_index is not None:
                break
    else:
        mask.bitwise_or_(visible_mask)

    return mask