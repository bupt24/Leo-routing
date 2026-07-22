"""
下载和解析 Starlink TLE 数据

TLE (Two-Line Element) 是描述卫星轨道的标准格式，
由 NORAD 维护，可从 CelesTrak 免费获取。
"""

import os
import math
import requests
from datetime import datetime
from pathlib import Path

import torch

# 地球常数
EARTH_RADIUS_KM = 6371.0
MU = 398600.4418  # km³/s² 地球引力常数


def download_starlink_tle(save_dir: str = "data/tle") -> str:
    """
    从 CelesTrak 下载最新的 Starlink TLE 数据
    
    :param save_dir: 保存目录
    :return: 保存的文件路径
    """
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"正在从 CelesTrak 下载 Starlink TLE 数据...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    # 保存文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"starlink_tle_{timestamp}.txt"
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, "w") as f:
        f.write(response.text)
    
    # 统计卫星数量
    lines = response.text.strip().split("\n")
    num_sats = len(lines) // 3
    print(f"下载完成！共 {num_sats} 颗卫星，保存至: {filepath}")
    
    # 同时保存一份 latest 版本
    latest_path = os.path.join(save_dir, "starlink_tle_latest.txt")
    with open(latest_path, "w") as f:
        f.write(response.text)
    
    return filepath


def parse_tle_file(filepath: str) -> list[dict]:
    """
    解析 TLE 文件
    
    :param filepath: TLE 文件路径
    :return: 卫星轨道参数列表
    """
    with open(filepath, "r") as f:
        content = f.read()
    
    # 移除空行，重新组织
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    
    satellites = []
    i = 0
    
    while i < len(lines) - 2:
        # 找到以数字开头的 TLE 第一行
        if not lines[i].startswith("1 "):
            # 这是卫星名称
            name = lines[i]
            i += 1
            if i >= len(lines) - 1:
                break
        else:
            name = "UNKNOWN"
        
        line1 = lines[i]
        line2 = lines[i + 1] if i + 1 < len(lines) else ""
        
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            i += 1
            continue
        
        i += 2  # 跳过已处理的两行
        
        try:
            # 解析 TLE 第一行
            norad_id = int(line1[2:7])
            epoch_year = int(line1[18:20])
            epoch_day = float(line1[20:32])
            
            # 解析 TLE 第二行
            inclination = float(line2[8:16])  # 轨道倾角 (度)
            raan = float(line2[17:25])  # 升交点赤经 (度)
            eccentricity = float("0." + line2[26:33])  # 偏心率
            arg_perigee = float(line2[34:42])  # 近地点幅角 (度)
            mean_anomaly = float(line2[43:51])  # 平均近点角 (度)
            mean_motion = float(line2[52:63])  # 平均运动 (圈/天)
            
            # 计算轨道半长轴
            n = mean_motion * 2 * math.pi / 86400  # 角速度 (rad/s)
            semi_major_axis = (MU / (n ** 2)) ** (1/3)  # km
            altitude = semi_major_axis - EARTH_RADIUS_KM  # km
            
            satellites.append({
                "name": name,
                "norad_id": norad_id,
                "inclination_deg": inclination,
                "raan_deg": raan,
                "eccentricity": eccentricity,
                "arg_perigee_deg": arg_perigee,
                "mean_anomaly_deg": mean_anomaly,
                "mean_motion": mean_motion,
                "semi_major_axis_km": semi_major_axis,
                "altitude_km": altitude,
            })
        except (ValueError, IndexError) as e:
            # print(f"解析失败: {name}, 错误: {e}")
            continue
    
    return satellites


def tle_to_position(sat: dict, time_since_epoch_sec: float = 0.0) -> tuple[float, float, float]:
    """
    将 TLE 轨道参数转换为 ECI 坐标
    
    :param sat: 卫星轨道参数字典
    :param time_since_epoch_sec: 从 TLE epoch 开始的时间 (秒)
    :return: (x, y, z) in km
    """
    # 轨道参数
    a = sat["semi_major_axis_km"]  # 半长轴
    e = sat["eccentricity"]  # 偏心率
    i = math.radians(sat["inclination_deg"])  # 倾角
    raan = math.radians(sat["raan_deg"])  # 升交点赤经
    omega = math.radians(sat["arg_perigee_deg"])  # 近地点幅角
    M0 = math.radians(sat["mean_anomaly_deg"])  # 初始平均近点角
    
    # 计算平均运动 (rad/s)
    n = sat["mean_motion"] * 2 * math.pi / 86400
    
    # 当前平均近点角
    M = M0 + n * time_since_epoch_sec
    
    # 求解开普勒方程 M = E - e*sin(E) (牛顿迭代)
    E = M
    for _ in range(10):
        E = M + e * math.sin(E)
    
    # 真近点角
    nu = 2 * math.atan2(
        math.sqrt(1 + e) * math.sin(E / 2),
        math.sqrt(1 - e) * math.cos(E / 2)
    )
    
    # 轨道平面内的距离
    r = a * (1 - e * math.cos(E))
    
    # 轨道平面内坐标
    x_orbital = r * math.cos(nu)
    y_orbital = r * math.sin(nu)
    
    # 转换到 ECI 坐标系
    cos_raan, sin_raan = math.cos(raan), math.sin(raan)
    cos_i, sin_i = math.cos(i), math.sin(i)
    cos_omega, sin_omega = math.cos(omega), math.sin(omega)
    
    # 旋转矩阵
    x = (cos_raan * cos_omega - sin_raan * sin_omega * cos_i) * x_orbital + \
        (-cos_raan * sin_omega - sin_raan * cos_omega * cos_i) * y_orbital
    y = (sin_raan * cos_omega + cos_raan * sin_omega * cos_i) * x_orbital + \
        (-sin_raan * sin_omega + cos_raan * cos_omega * cos_i) * y_orbital
    z = (sin_omega * sin_i) * x_orbital + (cos_omega * sin_i) * y_orbital
    
    return x, y, z


def generate_positions_from_tle(
    satellites: list[dict],
    time_sec: float = 0.0,
    max_sats: int = None,
    altitude_range: tuple[float, float] = (500, 600),
) -> torch.Tensor:
    """
    从 TLE 数据生成卫星位置张量
    
    :param satellites: TLE 解析后的卫星列表
    :param time_sec: 仿真时间 (秒)
    :param max_sats: 最大卫星数量 (None 表示全部)
    :param altitude_range: 高度筛选范围 (km)
    :return: 位置张量 [num_sats, 3]
    """
    positions = []
    
    for sat in satellites:
        # 筛选特定高度范围的卫星
        if altitude_range:
            if not (altitude_range[0] <= sat["altitude_km"] <= altitude_range[1]):
                continue
        
        x, y, z = tle_to_position(sat, time_sec)
        positions.append([x, y, z])
        
        if max_sats and len(positions) >= max_sats:
            break
    
    return torch.tensor(positions, dtype=torch.float32)


def filter_starlink_shell(
    satellites: list[dict],
    shell: str = "shell1"
) -> list[dict]:
    """
    筛选特定 Starlink 壳层的卫星
    
    Starlink 壳层参数:
    - Shell 1: 550 km, 53° 倾角, 22 轨道面 × 72 卫星
    - Shell 2: 540 km, 53.2° 倾角
    - Shell 3: 570 km, 70° 倾角
    - Shell 4: 560 km, 97.6° 倾角 (极轨)
    
    :param satellites: 全部卫星列表
    :param shell: 壳层名称
    :return: 筛选后的卫星列表
    """
    shell_params = {
        "shell1": {"alt": (540, 560), "inc": (52, 54)},
        "shell2": {"alt": (530, 550), "inc": (52, 54)},
        "shell3": {"alt": (560, 580), "inc": (69, 71)},
        "shell4": {"alt": (550, 570), "inc": (96, 99)},
    }
    
    if shell not in shell_params:
        return satellites
    
    params = shell_params[shell]
    filtered = []
    
    for sat in satellites:
        alt = sat["altitude_km"]
        inc = sat["inclination_deg"]
        
        if params["alt"][0] <= alt <= params["alt"][1] and \
           params["inc"][0] <= inc <= params["inc"][1]:
            filtered.append(sat)
    
    return filtered


if __name__ == "__main__":
    # 测试下载和解析
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "data", "tle")
    
    filepath = download_starlink_tle(data_dir)
    satellites = parse_tle_file(filepath)
    
    print(f"\n解析完成，共 {len(satellites)} 颗卫星")
    
    # 筛选 Shell 1 (550km, 53°)
    shell1_sats = filter_starlink_shell(satellites, "shell1")
    print(f"Shell 1 卫星数: {len(shell1_sats)}")
    
    # 生成位置
    positions = generate_positions_from_tle(shell1_sats, time_sec=0, max_sats=100)
    print(f"位置张量形状: {positions.shape}")
    
    # 显示前几颗卫星
    print("\n前 5 颗卫星:")
    for sat in shell1_sats[:5]:
        print(f"  {sat['name']}: 高度={sat['altitude_km']:.1f}km, 倾角={sat['inclination_deg']:.1f}°")
