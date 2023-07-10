#!/bin/bash


# assume given relative path
# e.g., call by change_fig_names.sh 20220314, where 20220314 is the file name
for d in $@
do
    echo "====" $d
    for f in "$d"/*.png
    do
        fb=$(basename "$f")
        echo "change for $fb"
        echo $f
        mv "$f" "${d}/${d}_${fb}"
    done
done