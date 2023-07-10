#!/bin/bash


# assume given relative path
for d in $@
do
    echo "====" $d
    for f in "$d"/*.png
    do
        fb=$(basename "$f")
        echo "change for $fb"
        mv "$f" "${d}/${d}_${fb}"
    done
done