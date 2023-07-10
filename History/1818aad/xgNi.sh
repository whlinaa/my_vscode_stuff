#!/bin/bash

for d in $@
do
    echo "====" $d
    for f in "$d"/*.png
    do
        fb=$(basename "$f")
        echo "change for $fb"
        mv "$f" "${d}/123_${fb}"
    done
done