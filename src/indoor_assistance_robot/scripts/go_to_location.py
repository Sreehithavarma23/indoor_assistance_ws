#!/usr/bin/env python3
import sys
import os
import yaml
import math
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from ament_index_python.packages import get_package_share_directory

# Workspace path setup for indoor_robot_ai
workspace_src = os.path.expanduser("~/indoor_assistance_ws/src/indoor_assistance_robot/scripts/indoor_robot_ai")
if workspace_src not in sys.path:
    sys.path.insert(0, workspace_src)

NLU_ENGINE = None
ASR_INSTANCE = None
LOAD_AUDIO = None

# Load NLU Engine
try:
    from src.nlu import parse as nlu_parse
    NLU_ENGINE = nlu_parse
    print("[INFO] NLU parse function loaded successfully!")
except Exception as e:
    print(f"[WARN] Could not load NLU Engine: {e}")

# Load WhisperASR and load_audio_file
try:
    from src.speech.asr import WhisperASR
    from src.speech.audio_capture import load_audio_file
    ASR_INSTANCE = WhisperASR()
    LOAD_AUDIO = load_audio_file
    print("[INFO] WhisperASR and load_audio_file loaded successfully!")
except Exception as e:
    print(f"[WARN] Could not load ASR Engine: {e}")

def get_pose(navigator, x, y, yaw_rad):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.orientation.z = math.sin(yaw_rad / 2.0)
    pose.pose.orientation.w = math.cos(yaw_rad / 2.0)
    return pose

def resolve_location(user_input, locations):
    locations_lower = {str(k).strip().lower(): k for k in locations.keys()}
    clean_input = user_input.strip().strip('.').lower()

    # Case A: Audio .wav File Input
    if clean_input.endswith('.wav'):
        print(f"[INFO] Processing Audio File: {user_input}")
        if ASR_INSTANCE and LOAD_AUDIO:
            try:
                audio_data, sr = LOAD_AUDIO(user_input)
                result_obj = ASR_INSTANCE.transcribe(audio_data, sr)
                
                # Extract text attribute from Transcript object
                transcript = getattr(result_obj, 'text', str(result_obj))
                print(f"[INFO] Transcribed Text: '{transcript}'")
                
                user_input = transcript
                clean_input = str(transcript).strip().strip('.').lower()
            except Exception as e:
                print(f"[ERROR] ASR Transcription failed: {e}")
                return None
        else:
            print("[ERROR] ASR Engine is unavailable.")
            return None

    # Case B: Direct Match in locations.yaml (Case-Insensitive)
    if clean_input in locations_lower:
        return locations_lower[clean_input]

    # Case C: Speech Text String parsed through NLU
    if NLU_ENGINE:
        result = NLU_ENGINE(user_input) if callable(NLU_ENGINE) else {}
        
        if isinstance(result, dict):
            intent = result.get('intent')
            entities = result.get('entities', {})
        else:
            intent = getattr(result, 'intent', None)
            entities = getattr(result, 'entities', {})

        destination = entities.get('destination') if isinstance(entities, dict) else None
        if destination:
            destination = str(destination).strip().strip('.').lower()

        print(f"[INFO] NLU Parsed Intent: {intent} | Destination: {destination}")

        if destination in locations_lower:
            return locations_lower[destination]

    return None

def main():
    if len(sys.argv) < 2:
        print("[ERROR] Please provide a location, sentence, or audio path.")
        return

    user_input = " ".join(sys.argv[1:])

    rclpy.init()
    navigator = BasicNavigator()
    navigator.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])

    pkg_share = get_package_share_directory('indoor_assistance_robot')
    yaml_path = os.path.join(pkg_share, 'config', 'locations.yaml')

    if not os.path.exists(yaml_path):
        print(f"[ERROR] Could not find locations.yaml at {yaml_path}")
        rclpy.shutdown()
        return

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    locations = data.get('locations', {})
    target_name = resolve_location(user_input, locations)

    if not target_name:
        print(f"[ERROR] Could not resolve target location from input: '{user_input}'")
        print(f"Available locations in config: {list(locations.keys())}")
        rclpy.shutdown()
        return

    target_data = locations[target_name]
    target_pose = get_pose(navigator, target_data['x'], target_data['y'], target_data['yaw'])

    print("[INFO] Waiting for Nav2 to activate...")
    navigator.waitUntilNav2Active()

    print(f"[INFO] Navigating to target location: '{target_name}' at ({target_data['x']}, {target_data['y']})...")
    navigator.goToPose(target_pose)

    while not navigator.isTaskComplete():
        pass

    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        print(f"[SUCCESS] Robot successfully reached '{target_name}'!")
    elif result == TaskResult.CANCELED:
        print(f"[WARN] Navigation to '{target_name}' was canceled.")
    elif result == TaskResult.FAILED:
        print(f"[ERROR] Navigation to '{target_name}' failed.")

    rclpy.shutdown()

if __name__ == '__main__':
    main()