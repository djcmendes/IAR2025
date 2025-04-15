from controller import Robot, DistanceSensor, Motor

# Constants
TIME_STEP = 64  # Time step (should match the Webots world)
MAX_SPEED = 10   # Maximum motor speed (rad/s)

# Create the Robot instance
robot = Robot()

# Get and enable light sensors (front left and front right)
left_sensor = robot.getDevice('light_sensor_left')  # Front left
right_sensor = robot.getDevice('light_sensor_right') # Front right
left_sensor.enable(TIME_STEP)
right_sensor.enable(TIME_STEP)

# Get motors and set them to velocity control mode
left_motor = robot.getDevice('motor.left')
right_motor = robot.getDevice('motor.right')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

while robot.step(TIME_STEP) != -1:
    # Read sensor values
    left_value = left_sensor.getValue()
    right_value = right_sensor.getValue()
    
    #print (left_value)
    #print (right_value)
    
    # Braitenberg Vehicle 2a (phototaxis) configuration:
    # Left sensor controls right motor, right sensor controls left motor
    left_speed = 10  # Adjust the 0.01 scaling factor as needed
    right_speed = 10
    
    # Cap speeds to maximum
    left_speed = min(left_speed, MAX_SPEED)
    right_speed = min(right_speed, MAX_SPEED)
    
    # Set motor speeds
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)