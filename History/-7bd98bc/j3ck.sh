#!/bin/bash

foo=0
test_var(){
    foo=1 # it will modify the global foo...
    echo "foo = $foo"

    local foo
    foo=100
    echo "foo = $foo"
}

# echo "global: foo = $foo"
# test_var
# echo "global: foo = $foo"

print(){
    echo "calling print functions..."
    echo "arguments 0: $0"
    echo "arguments 1: $1"
    echo "arguments 2: $2"
    echo "arguments 3: $3"
}
control_tmp(){
    # after [[, we must have a space
    # after if, we must have a space
    echo "calling control function..."
    if [[ $1 = "0" ]]; then
        echo "arg[1] == 0"
    elif [[ $1 = "1" ]]; then
        echo "arg[1] == 1"
    elif [[ $1 = "2" ]]; then
        echo "arg[1] == 2"
    else
        echo "nothing match the value of arg[1]"
    fi
}

control_tmpp(){
    if [ "$1" -eq 5 ]; then
        echo "x equals 5."
    else
        echo "x doesn't equal 5."
    fi

    echo "x = $1"
}

control_string(){
    ANSWER=$1
    if [ -z "$ANSWER" ]; then
        echo "There is no answer." >&2 # redirect error to stderr
        # echo "There is no answer."
        exit 1
    fi
    if [ "$ANSWER" == "yes" ]; then
        echo "The answer is YES."
    elif [ "$ANSWER" == "no" ]; then
        echo "The answer is NO."
    elif [ "$ANSWER" == "maybe" ]; then
        echo "The answer is MAYBE."
    else
        echo "The answer is UNKNOWN."
    fi
}
# control_string "yes"
# control_string "no"
# control_string "maybe"
# control_string "xx"

control_int(){
    INT=$1
    if [ -z "$INT" ]; then
        echo "INT is empty." >&2
        exit 1
    fi
    # if [ "$INT" -eq 0 ]; then
    if (( INT == 0 )) ; then
        echo "INT is zero."
    else
        # if [ "$INT" -lt 0 ]; then
        if (( INT < 0)); then
            echo "INT is negative."
        else
            echo "INT is positive."
    fi
    # if [ $((INT % 2)) -eq 0 ]; then
    if (( ((INT%2)) == 0 )); then
        echo "INT is even."
    else
        echo "INT is odd."
    fi
    fi
}
# control_int 100
# # echo $'hello\nworld'
# echo 
# control_int -1
# echo 
# control_int 0
# echo 
# control_int 

get_input(){
    echo "calling get_input function..."
    read -p 'enter something (this will be put to $REPLY): '
    echo $REPLY

    read -p "Enter your name: " text
    echo "the name you enter is: $text" # "" needed. Otherwise, if input is "hello     world", then it becomes "hello world"

    read -p "enter multiple values: " a1 a2 a3 a4
    echo "$a1"
    echo "$a2"
    echo "$a3"
    echo "$a4"
}

# get_input

arg_practice(){
    # echo "
    # Number of arugments: $#
    # \$0 = $0
    # \$1 = $1
    # \$2 = $2
    # \$3 = $3
    # \$4 = $4
    # \$5 = $5
    # "
    count=1
    while [[ $#>0 ]]; do
        echo "count = $count"
        echo "Argument $count = $1"
        count=$((count+1))
        shift
    done
}

# arg_practice ~/Desktop/*

file_info(){
PROGNAME="$(basename "$0")"
# echo "hello"
if [[ -e "$1" ]]; then
    echo -e "\nFile Type:"
    file "$1"
    echo -e "\nFile Status:"
    stat "$1"
else
    echo "$PROGNAME: usage: $PROGNAME file" >&2
    exit 1
fi 
}

file_info ~/Desktop/
# file_info ~/Desktop/new.txt
# echo "hello"
# file_info /bin/*


print_params () {
    echo "\$1 = $1"
    echo "\$2 = $2"
    echo "\$3 = $3"
    echo "\$4 = $4"
}
pass_params () {
    echo -e "\n" '$* :'; print_params $*
    echo -e "\n" '"$*" :'; print_params "$*"
    echo -e "\n" '$@ :'; print_params $@
    echo -e "\n" '"$@" :'; print_params "$@"
}
# pass_params "word" "words with spaces"

array_create(){
    # a[-1]=30 # error!

    echo ${a[0]}
    echo ${a[1]}
    echo ${a[2]}
    echo ${a[3]}

    days=(Sun Mon Tue Wed Thu Fri Sat)
    stu=([1]=Mary [0]=tom)
    echo ${days[0]}
    echo ${stu[0]}
}

# array_create

while_syntax(){
    count=1
    while(( $count<5 )); do
        echo $count
        count=$((count+1))
    done

    echo "Finished"
}

# while_syntax 

for_syntax(){
    # for i in I am a boy; do
    #     echo $i
    # done
    # echo
    # for i in {A..E}; do
    #     echo $i
    # done
    
    # # get all .txt files
    # for i in *.txt; do
    #     echo $i
    # done

    # for i in distros*.txt; do
    #     echo "$i" # will print the give name, since no match
    # done

    # this avoid the problem above.
    # for i in distros*.txt; do
    #     if [[ -e "$i" ]]; then
    #         echo "$i"
    #     fi
    # done

    # process positional parameters instead 
    # for i; do
    #     echo $i
    # done

    # C-style for loop
    for (( i=0; i<3; i=i+1 )); do 
        echo $i
    done

    # the above is same as 
    i=0
    while (( $i<3 )); do
        echo $i
        i=$((i+1))
    done

}
# for_syntax "hello" 1 2 "world"


expansion_issue(){
    # a="foo"
    # echo $a_file # output nothing, since there is no variable named `a_file`!
    # echo ${a}_file 

    # b[0]=10
    # echo $b[0] # will output 10[0]
    # echo ${b[0]}

    # echo $#
    # echo $1
    # echo $13
    # echo ${13}

    # tmp=
    # echo $tmp # print nothing
    # echo ${tmp:="does not exist"} # does not exist
    # echo $tmp

    # err=
    # echo ${err:?"does not exist"}

    # text="Hello world!"
    # echo "number of characters in '$text' is ${#text}"

    echo $# # 36
    echo ${#} # 36
    echo ${#@} # 36
    echo ${#*} # 36
}
# expansion_issue $(ls /bin)




# get_input

# print 
# print hello world I am fat! You? 
# print hello world I am fat! You?
# print hello world I am fat! You?
# print hello world I am fat! You?
# print "hello world I am fat! You?"

# control 0
# control 1
# control 2
# control 100

# get_input 100


expansion_ex(){
    # ls -d /[a-z]* # show all directories in / that starts with lower letter
    # ls -d /[A-Z]* # show all directories in / that starts with upper letter

    # s1=XJPG.JPGX
    # s2=JPG.JPG
    # for i in {$s1,$s2}; do
    #     echo ${i/JPG/jpg} # replace 1st match only
    #     echo ${i//JPG/jpg} # replace all matches
    #     echo ${i/#JPG/jpg} # replace only if found at start
    #     echo ${i/%JPG/jpg} # replace only if found at end
    # done

    # foo=10
    # if (( foo=100 )); then
    #     echo "true"
    # else
    #     echo "false"
    # fi
    # echo $foo

    # s="hello world"
    # echo $s
    # if [ "hello world"=="xyz" ]; then
    #     echo "true"
    # else
    #     echo "false"
    # fi
    # echo $s

    s="no"
    echo $s
    if [ $s == "yes" ]; then
        echo "yes"
    else
        echo "no"
    fi
    echo $s

}

# expansion_ex


find_ex(){
    cd ~/Desktop
    # find . -perm 777 -size +5 -exec ls -al {} + 
    open *.pdf 

    cd ~/OneDrive\ -\ HKUST\ Connect
    # for i in $(find ~/Desktop -perm 777 -size +1M); do
    for i in "$(find . -size +100M -exec dirname {} +)"; do
        echo "hello"
        echo "$i"
        # echo "He"
        # open "$i"

    done


    # echo ""
    # for i in "$(find . -size +100M)"; do
    #     echo "$i"
    #     echo "Hello"
    #     echo "$(dirname "$i")"
    #     echo "$(basename "$i")"
    #     open $(dirname $i)
    # done

}

# find_ex


