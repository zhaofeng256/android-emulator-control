#!/bin/sh
while true; do
    input touchscreen swipe 200 500 200 480 100
    input touchscreen swipe 200 500 200 520 100
    input tap 80 326
    sleep 1
done
