import argparse
import sys

import tensorflow as tf


@tf.function
def calculate_equations(A, A_shape, V, V_size):
    A = tf.constant(A, shape=A_shape, dtype=tf.float32)
    B = tf.constant(V, shape=[V_size, 1], dtype=tf.float32)
    return tf.linalg.solve(A, B)


def create_parser():
    parser = argparse.ArgumentParser(
        prog="main",
        description="Solve equations using TensorFlow matrix operations.",
        epilog="Example: python main.py -a '2 3 1 -1' -ash '2 2' -v '8 1' -vs 2",
    )
    parser.add_argument(
        "-a", "--a_values", required=True, help="Matrix A values (space-separated)"
    )
    parser.add_argument(
        "-ash", "--a_shape", required=True, help="Shape of A (rows cols)"
    )
    parser.add_argument(
        "-v", "--variables", required=True, help="Vector values (space-separated)"
    )
    parser.add_argument(
        "-vs", "--v_size", required=True, help="Vector size (number of equations)"
    )
    return parser


def validate_inputs(A, A_shape, V, V_size):
    if not A or not A_shape or not V or not V_size:
        sys.exit("Error: All arguments (A, A_shape, V, V_size) are required.")

    try:
        A_shape = [int(x) for x in A_shape]
    except ValueError:
        sys.exit("Error: A_shape must contain integers.")

    rows, cols = A_shape
    if len(A) != rows * cols:
        sys.exit(
            f"Error: A must have {rows * cols} elements for shape {A_shape}, but got {len(A)}."
        )

    if V_size != rows:
        sys.exit(
            f"Error: V_size ({V_size}) must match the number of rows in A ({rows})."
        )

    return A_shape


def is_singular(A, A_shape):
    A = tf.constant(A, shape=A_shape, dtype=tf.float32)
    det = tf.linalg.det(A)
    if tf.abs(det) < 1e-8:
        return True
    return False


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    try:
        A = [float(x) for x in args.a_values.split()]
        A_shape = args.a_shape.split()
        V = [float(x) for x in args.variables.split()]
        V_size = int(args.v_size)
    except ValueError:
        sys.exit("Error: All matrix/vector values must be numeric.")

    A_shape = validate_inputs(A, A_shape, V, V_size)

    if is_singular(A, A_shape):
        sys.exit(
            "Error: Matrix A is singular or nearly singular — cannot solve uniquely."
        )

    try:
        solution = calculate_equations(A, A_shape, V, V_size)
        print("Solution:\n", solution.numpy())
    except Exception as e:
        sys.exit(f"TensorFlow error while solving: {e}")
