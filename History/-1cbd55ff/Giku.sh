#!/bin/bash
# find . -name "*20220322_21041*" -exec cp -r {} ~/Desktop/ \;
# pwd
echo $1
find $1 -name "*20220322_21041*" 
echo "hello world"
# find . -name $1 -exec cp -r {} ~/Desktop/ \;