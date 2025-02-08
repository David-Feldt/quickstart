# Adds the lib directory to the Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from drive_controller import RobotController
import time

def main():
    try:
        # Initialize the robot controller
        robot = RobotController()
        
        # Drive forward 1 meter
        print("Driving forward 1 meter...")
        robot.drive_distance(1.0)
        time.sleep(5)  # Brief pause between movements
        
        # Turn 90 degrees right (-90 for right turn)
        print("Turning right 90 degrees...")
        robot.turn_degrees(90)
        time.sleep(5)  # Brief pause between movements
        
        # Drive forward 1 meter
        print("Driving forward 1 meter...")
        robot.drive_distance(1.0)
        
        # Cleanup
        robot.cleanup()
        print("Movement complete!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if 'robot' in locals():
            robot.cleanup()

if __name__ == "__main__":
    main()
