#!/bin/bash
arg=$1

case $arg in
    core)
        echo "Installing autogoal-core"
        cd /home/coder/autogoal/ && pip install -e autogoal
    ;;
    remote)
        echo "Installing autogoal-remote"
        cd /home/coder/autogoal/ && pip install -e autogoal-remote
    ;;
    common)
        echo "Installing autogoal-contrib-common"
        cd /home/coder/autogoal/autogoal-contrib/ && pip install -e common
    ;;
    sklearn | nltk)
        echo "Installing autogoal-$arg"
        cd /home/coder/autogoal/autogoal-contrib/ && pip install -e $arg
    ;;
    *)
        echo "Invalid argument"
    ;;
esac