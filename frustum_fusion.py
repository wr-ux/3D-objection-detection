import numpy as np
import pandas as pd
import os

def mask_guided_frustum_association():
    print("启动 Mask 引导截锥体 3D 关联 (Mask-Guided Frustum Association)...\n")

    # ================= 1. 定义路径 =================
    # 请确保这两个文件在当前目录下，或修改为绝对路径
    mask_path = "/home/waas/Project4/Achelous-main/export_results/46221_ship_mask.npy"  
    csv_path = "/home/waas/Project4/radar/46221.csv" 
    
    if not os.path.exists(mask_path) or not os.path.exists(csv_path):
        print(f"找不到文件，请检查路径: \n{mask_path}\n{csv_path}")
        return

    # ================= 2. 加载数据 =================
    ship_mask = np.load(mask_path)
    height, width = ship_mask.shape # (1080, 1920)
    print(f"成功加载 Mask，分辨率: {width}x{height}")

    radar_df = pd.read_csv(csv_path)
    total_points = len(radar_df)
    print(f"成功加载原始雷达点云，总点数: {total_points}")

    # ================= 3. 剔除视野外的点 =================
    # 只保留投射在相机图像分辨率 (1920x1080) 以内的点
    valid_cam_mask = (
        (radar_df['u'] >= 0) & (radar_df['u'] < width) &
        (radar_df['v'] >= 0) & (radar_df['v'] < height) &
        (radar_df['z'] > 0) # 深度必须大于 0（在相机前方）
    )
    cam_points = radar_df[valid_cam_mask].copy()
    print(f"过滤视野外点云，剩余点数: {len(cam_points)}")

    # ================= 4. Mask 视锥体过滤 =================
    # 获取视野内雷达点对应的像素坐标
    u_indices = cam_points['u'].astype(int).values
    v_indices = cam_points['v'].astype(int).values
    
    # 使用 Numpy 索引提取坐标对应的 Mask 值
    mask_values = ship_mask[v_indices, u_indices]
    
    # 提取落在目标区域内（mask 值为 1）的雷达点
    target_points = cam_points[mask_values == 1]
    
    print(f"经过 Mask 过滤，命中目标点数: {len(target_points)}")

    # ================= 5. 3D 距离聚合 =================
    if len(target_points) > 0:
        # 获取目标点的深度值 Z
        z_values = target_points['z'].values
        
        # 计算中位数，用于排除异常离群点
        median_z = np.median(z_values)
        min_z = np.min(z_values)
        
        print("\n【最终 3D 测距结果】")
        print(f"该目标的预估绝对距离为: {median_z:.2f} 米 (最近反射点: {min_z:.2f} 米)")
    else:
        print("\n没有任何雷达点落入 Mask 区域。")

if __name__ == "__main__":
    mask_guided_frustum_association()
