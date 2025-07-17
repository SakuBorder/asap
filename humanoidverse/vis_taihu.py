import os
import sys
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import os,sys
cur_work_path = os.getcwd()
sys.path.append(cur_work_path)

import torch
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot

color_manager = {
    'get_color': lambda name: (0.0, 1.0, 0.0),  
    'get_random_color': lambda: (1.0, 0.0, 0.0) 
}
args = gymutil.parse_arguments(description="Robot Motion Visualizer")

gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.contact_offset = 0.02
    sim_params.physx.rest_offset = 0.001

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
plane_params.restitution = 0.0
gym.add_ground(sim, plane_params)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

cam_pos = gymapi.Vec3(3.0, 3.0, 2.0)
cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

asset_root = "./"
asset_file = "humanoidverse/data/robots/tai5/tai5.urdf"
tai5_xml = "humanoidverse/data/robots/tai5/tai5.xml"

print(f"Loading asset: {asset_file}")

if not os.path.exists(asset_file):
    print(f"*** Asset file not found: {asset_file}")
    quit()

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.use_mesh_materials = True
asset_options.vhacd_enabled = False
asset_options.disable_gravity = False
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset_options.collapse_fixed_joints = True

try:
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    print("Asset loaded successfully")
except Exception as e:
    print(f"*** Failed to load asset: {e}")
    quit()

num_dofs = gym.get_asset_dof_count(asset)
num_bodies = gym.get_asset_rigid_body_count(asset)
print(f"Asset info: {num_dofs} DOFs, {num_bodies} bodies")

env_lower = gymapi.Vec3(-2.0, -2.0, 0.0)  
env_upper = gymapi.Vec3(2.0, 2.0, 2.0)
env = gym.create_env(sim, env_lower, env_upper, 1)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

actor_handle = gym.create_actor(env, asset, pose, "robot", 0, 1)

dof_props = gym.get_actor_dof_properties(env, actor_handle)
for i in range(num_dofs):
    dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
    dof_props['stiffness'][i] = 300.0
    dof_props['damping'][i] = 10.0

dof_pos_limits = torch.zeros(num_dofs, 2, dtype=torch.float, device=args.sim_device, requires_grad=False)
hard_dof_pos_limits = torch.zeros(num_dofs, 2, dtype=torch.float, device=args.sim_device, requires_grad=False)
dof_vel_limits = torch.zeros(num_dofs, dtype=torch.float, device=args.sim_device, requires_grad=False)
torque_limits = torch.zeros(num_dofs, dtype=torch.float, device=args.sim_device, requires_grad=False)
for i in range(len(dof_props)):
    dof_pos_limits[i, 0] = dof_props["lower"][i].item()
    dof_pos_limits[i, 1] = dof_props["upper"][i].item()
    dof_pos_limits[i, 0] = dof_props["lower"][i].item()
    dof_vel_limits[i] = dof_props["velocity"][i].item()
    torque_limits[i] = dof_props["effort"][i].item()

gym.set_actor_dof_properties(env, actor_handle, dof_props)

def get_standing_pose():
    """定义一个基本的站立姿态"""
    pose = np.zeros(num_dofs)
    
    joint_names = []
    for i in range(num_dofs):
        joint_names.append(gym.get_asset_dof_name(asset, i))
    
    for i, name in enumerate(joint_names):
        if 'HIP_P' in name:
            pose[i] = 0.0
        elif 'HIP_R' in name:
            pose[i] = 0.0
        elif 'HIP_Y' in name:
            pose[i] = 0.0
        elif 'KNEE_P' in name:
            pose[i] = 0.3
        elif 'ANKLE_P' in name:
            pose[i] = -0.15
        elif 'ANKLE_R' in name:
            pose[i] = 0.0
        elif 'SHOULDER_R' in name: 
            if 'L_' in name:
                pose[i] = -1.0 
            else:
                pose[i] = 1.0 
        else:
            pose[i] = 0.0
    
    return pose

standing_pose = get_standing_pose()

print("Loading motion library...")
# 🔥 修改：使用整个动作库而不是单个文件
motion_file = "./humanoidverse/data/motions/tai5/walk.pkl"

if not os.path.exists(motion_file):
    print(f"*** Motion file not found: {motion_file}")
    print("Will use standing pose only")
    motion_lib = None
    sk_tree = None
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = "./humanoidverse/config/robot/tai5/tai5.yaml"
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)
    motion_lib = MotionLibRobot(config.robot.motion, num_envs=17, device=args.sim_device)
    
    motion_lib.load_motions(random_sample=False)
    num_motions = motion_lib.num_motions()
    motion_keys = motion_lib.curr_motion_keys
    print(f"Motion library loaded successfully with {num_motions} motions")
    print(f"Motion keys: {motion_keys}")

dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
for i in range(num_dofs):
    dof_states['pos'][i] = standing_pose[i]
    dof_states['vel'][i] = 0.0

gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

print("Preparing simulation...")
gym.prepare_sim(sim)

num_envs = 1
env_ids = torch.arange(num_envs, dtype=torch.int32, device=args.sim_device)

current_pose = torch.tensor(standing_pose, dtype=torch.float32, device=args.sim_device).unsqueeze(0)

dof_states_tensor = gym.acquire_dof_state_tensor(sim)
dof_states_tensor = gymtorch.wrap_tensor(dof_states_tensor)

rigidbody_state_tensor = gym.acquire_rigid_body_state_tensor(sim)
rigidbody_state_tensor = gymtorch.wrap_tensor(rigidbody_state_tensor)
rigidbody_state_tensor = rigidbody_state_tensor.reshape(num_envs, -1, 13)

actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
actor_root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor)

try:
    contact_force_tensor = gym.acquire_net_contact_force_tensor(sim)
    contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
    contact_force_tensor = contact_force_tensor.view(num_envs, -1, 3)
    contact_available = True
    print("Contact force tensor acquired successfully")
except Exception as e:
    print(f"Warning: Could not acquire contact force tensor: {e}")
    contact_available = False

import mujoco
model = mujoco.MjModel.from_xml_path(tai5_xml)
body_names = []
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    body_names.append(name)
print(f"Mujoco Rigid bodies: {body_names}")

body_names = []
for i in range(num_bodies):
    body_names.append(gym.get_asset_rigid_body_name(asset, i))
print(f"Isaac Rigid bodies: {body_names}")

def create_joint_mapping_fix(dof_pos_raw):
    """创建关节映射修正 - 基于输出分析优化"""
    dof_pos_fixed = dof_pos_raw.clone()
    
    joint_names = []
    for i in range(num_dofs):
        joint_names.append(gym.get_asset_dof_name(asset, i))
    
    print(f"Applying joint fix at time {motion_time:.2f}")
    
    for i, name in enumerate(joint_names):
        original_value = dof_pos_raw[0][i].item()
        
        if name == 'R_SHOULDER_R':
            dof_pos_fixed[0][i] = -abs(original_value) - 0.5 
            print(f"  {name}: {original_value:.3f} -> {dof_pos_fixed[0][i].item():.3f}")
            
        elif name == 'R_SHOULDER_Y':
            dof_pos_fixed[0][i] = -original_value
            print(f"  {name}: {original_value:.3f} -> {dof_pos_fixed[0][i].item():.3f}")
            
        elif name == 'R_SHOULDER_P':
            dof_pos_fixed[0][i] = -original_value * 0.8
            print(f"  {name}: {original_value:.3f} -> {dof_pos_fixed[0][i].item():.3f}")
            
        elif name == 'R_ELBOW_Y':
            dof_pos_fixed[0][i] = -torch.clamp(torch.abs(dof_pos_raw[0][i]), 0.1, 2.0)
            print(f"  {name}: {original_value:.3f} -> {dof_pos_fixed[0][i].item():.3f}")
            
        elif name == 'R_WRIST_P':
            dof_pos_fixed[0][i] = original_value * 0.3
            print(f"  {name}: {original_value:.3f} -> {dof_pos_fixed[0][i].item():.3f}")
            
        dof_pos_fixed[0][i] = torch.clamp(dof_pos_fixed[0][i], -3.14, 3.14)
    
    return dof_pos_fixed

# 🔥 修改：切换动作函数
def switch_to_motion(new_motion_id):
    """切换到指定的动作"""
    global motion_id, motion_time, motion_lib
    
    if motion_lib is None:
        print("No motion library available")
        return
    
    if 0 <= new_motion_id < num_motions:
        motion_id = new_motion_id
        motion_time = 0.0
        
        # 重新加载动作数据
        motion_lib.load_motions(random_sample=False, start_idx=motion_id)
        
        # 获取当前动作信息
        current_motion_key = motion_lib.curr_motion_keys[0] if len(motion_lib.curr_motion_keys) > 0 else "Unknown"
        print(f"✅ 切换到动作 {motion_id}: {current_motion_key}")
    else:
        print(f"❌ 无效的动作ID: {new_motion_id}, 有效范围: 0-{num_motions-1}")

# 🔥 修改：订阅所有需要的键盘事件
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "toggle_pause")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_M, "toggle_motion")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_N, "next_motion")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "prev_motion")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset_motion")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "quit")

# 🔥 修改：全局变量
paused = False
use_motion = motion_lib is not None
motion_time = 0.0
motion_id = 0
dt = sim_params.dt
show_collision_vis = True
collision_count = 0
debug_mode = True 
apply_joint_fix = False

# 🔥 修改：打印控制说明
def print_control_instructions():
    print("\n" + "="*60)
    print("🎮 键盘控制说明:")
    print("="*60)
    print("空格键 (SPACE)  : 暂停/播放")
    print("M键            : 启用/禁用动作播放")
    print("N键            : 下一个动作")
    print("P键            : 上一个动作")
    print("R键            : 重置动作时间")
    print("Q键            : 退出程序")
    print("="*60)
    if motion_lib is not None:
        print(f"📊 当前状态: 共{num_motions}个动作")
        print(f"🎬 当前动作: {motion_id}")
    print("="*60)
    print("💡 请确保点击Isaac Gym窗口获得焦点!")
    print("="*60)

print_control_instructions()

print("Starting visualization...")
if motion_lib is not None:
    print("Motion playback enabled")
else:
    print("Standing pose only (no motion file)")

# 🔥 修改：主循环
while not gym.query_viewer_has_closed(viewer):
    # 处理事件
    for evt in gym.query_viewer_action_events(viewer):
        # 🔥 修改：添加事件调试信息
        if evt.value > 0:
            print(f"🔍 检测到事件: {evt.action}")
        
        if evt.action == "toggle_pause" and evt.value > 0:
            paused = not paused
            print(f"🎮 模拟 {'暂停' if paused else '继续'}")
        elif evt.action == "toggle_motion" and evt.value > 0 and motion_lib is not None:
            use_motion = not use_motion
            motion_time = 0.0
            print(f"🎬 动作播放 {'启用' if use_motion else '禁用'}")
        # 🔥 新增：处理动作切换事件
        elif evt.action == "next_motion" and evt.value > 0 and motion_lib is not None:
            next_motion_id = (motion_id + 1) % num_motions
            switch_to_motion(next_motion_id)
        elif evt.action == "prev_motion" and evt.value > 0 and motion_lib is not None:
            prev_motion_id = (motion_id - 1) % num_motions
            switch_to_motion(prev_motion_id)
        elif evt.action == "reset_motion" and evt.value > 0 and motion_lib is not None:
            motion_time = 0.0
            print("🔄 重置动作时间")
        elif evt.action == "quit" and evt.value > 0:
            print("👋 程序退出")
            break
    
    # 检查是否需要退出
    if gym.query_viewer_has_closed(viewer):
        break
    
    if not paused:
        if motion_lib is not None and use_motion:
            # 🔥 修改：使用第一个slot获取动作长度
            motion_len = motion_lib.get_motion_length(torch.tensor([0]).to(args.sim_device))[0].item()
            motion_time_wrapped = motion_time % motion_len
            
            # 🔥 修改：使用第一个slot获取动作状态
            motion_res = motion_lib.get_motion_state(
                torch.tensor([0]).to(args.sim_device), 
                torch.tensor([motion_time_wrapped]).to(args.sim_device)
            )
            
            root_pos = motion_res["root_pos"]
            root_rot = motion_res["root_rot"] 
            dof_pos = motion_res["dof_pos"]
            root_vel = motion_res["root_vel"]
            root_ang_vel = motion_res["root_ang_vel"]
            
            if apply_joint_fix:
                dof_pos_clamped = create_joint_mapping_fix(dof_pos)
            else:
                dof_pos_clamped = dof_pos.clone()
                for i in range(num_dofs):
                    dof_pos_clamped[0][i] = torch.clamp(dof_pos_clamped[0][i], -3.14, 3.14)
            
            if "rg_pos" in motion_res:
                rb_pos = motion_res["rg_pos"]
                gym.clear_lines(viewer)
                gym.refresh_rigid_body_state_tensor(sim)
                
                for i, pos_joint in enumerate(rb_pos[0, 0:1]):
                    red_strength = 1.0 - i / max(rb_pos.shape[1] - 1, 1)
                    color = (red_strength, 0.0, 0.0)
                    sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=color)
                    sphere_pose = gymapi.Transform(gymapi.Vec3(pos_joint[0], pos_joint[1], pos_joint[2]), r=None)
                    gymutil.draw_lines(sphere_geom, gym, viewer, env, sphere_pose)

            root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1).repeat(num_envs, 1)
            gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root_states), 
                                                    gymtorch.unwrap_tensor(env_ids), len(env_ids))
            
            dof_state = torch.stack([dof_pos_clamped, torch.zeros_like(dof_pos_clamped)], dim=-1).squeeze().repeat(num_envs, 1)
            gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(dof_state), 
                                            gymtorch.unwrap_tensor(env_ids), len(env_ids))
            
            motion_time += dt
        else:
            # 使用站立姿态
            standing_dof_state = torch.tensor(standing_pose, dtype=torch.float32, device=args.sim_device)
            dof_state = torch.stack([standing_dof_state, torch.zeros_like(standing_dof_state)], dim=-1).unsqueeze(0).repeat(num_envs, 1)
            gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(dof_state), 
                                            gymtorch.unwrap_tensor(env_ids), len(env_ids))
        
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_actor_root_state_tensor(sim)

        rb_state_tensor = gym.acquire_rigid_body_state_tensor(sim)
        rb_state = gymtorch.wrap_tensor(rb_state_tensor)
        rb_state = rb_state.view(num_envs, num_bodies, 13)
        positions = rb_state[:, :, 0:3]
    
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

print("Cleaning up...")
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("Done!")