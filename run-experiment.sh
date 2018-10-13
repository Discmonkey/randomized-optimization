#!/usr/bin/env bash

export CLASSPATH=ABAGAIL/ABAGAIL.jar:$CLASSPATH

experiment_name=$1


jython experiments/${experiment_name}.py -Xmx64m