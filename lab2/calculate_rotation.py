import tensorflow as tf
import argparse

@tf.function
def calculate_rotation(x, y, angle):
    point = tf.constant([[x], [y]], dtype=tf.float32)

    rotation_matrix = tf.stack(
        [
            [tf.cos(angle), -tf.sin(angle)],
            [tf.sin(angle), tf.cos(angle)],
        ]
    )
    rotated_point = tf.matmul(rotation_matrix, point)
    return tf.reshape(rotated_point, shape=(2,))

def create_parser():
    new_parser = argparse.ArgumentParser(
        prog="main",
        description="Show the number from inputted image by user",
        epilog="bla bla",
    )
    new_parser.add_argument("-x", "--x_coordinate")
    new_parser.add_argument("-y", "--y_coordinate")
    new_parser.add_argument("-t", "--theta")
    return new_parser

if __name__ == "__main__":
    parser = create_parser()
    command_args = parser.parse_args()
    x, y, theta = float(command_args.x_coordinate), float(command_args.y_coordinate), float(command_args.theta)

    print(calculate_rotation(x, y, theta))

