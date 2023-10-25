# Install bazel
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.18.10/bazelisk-linux-arm64
chmod +x bazelisk-linux-arm64
sudo mv bazelisk-linux-arm64 /usr/local/bin/bazel

git clone https://github.com/tensorflow/addons.git
cd addons

# This script links project with TensorFlow dependency
python3 ./configure.py

bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts

pip install artifacts/tensorflow_addons-*.whl