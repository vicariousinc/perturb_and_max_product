THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python ${THIS_DIR}/generate_mnist_and_contour_mnist.py
python ${THIS_DIR}/generate_erdos_and_lattice.py
python ${THIS_DIR}/generate_convor.py


