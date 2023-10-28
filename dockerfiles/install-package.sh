#!/bin/bash
set -e

# Split the first argument into an array of words
contribs=("$@")
poetry config virtualenvs.create false
for arg in "${contribs[@]}"
do
    echo "Trying to install autogoal_$arg"
    case $arg in
        core)
            cd /home/coder/autogoal/autogoal && poetry install
            cd /home/coder/autogoal && pip install -e autogoal
        ;;
        remote)
            cd /home/coder/autogoal/autogoal-remote && poetry install
            cd /home/coder/autogoal && pip install -e autogoal-remote
        ;;
        common)
            cd /home/coder/autogoal/autogoal-contrib/autogoal_contrib && poetry install
            cd /home/coder/autogoal/autogoal-contrib && sudo pip install -e autogoal_contrib
        ;;
        keras)
            cd /home/coder/autogoal/autogoal-contrib/autogoal_keras && poetry install
            cd /home/coder/autogoal/autogoal-contrib && sudo pip install -e autogoal_keras
            cd /home/coder && wget https://github.com/bazelbuild/bazelisk/releases/download/v1.18.0/bazelisk-linux-arm64 && chmod +x bazelisk-linux-arm64  && sudo mv bazelisk-linux-arm64 /usr/local/bin/bazel 
            bazel
            cd /home/coder && git clone https://github.com/tensorflow/addons.git
            cd /home/coder/addons && sudo python ./configure.py && sudo bazel build build_pip_pkg && sudo bazel-bin/build_pip_pkg artifacts && pip install artifacts/tensorflow_addons-*.whl
        ;;
        *)
            cd "/home/coder/autogoal/autogoal-contrib/autogoal_$arg" && poetry install
            cd "/home/coder/autogoal/autogoal-contrib" && pip install -e "autogoal_$arg"
        ;;
    esac
done

