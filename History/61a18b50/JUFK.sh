#!/bin/bash

add_(){
    echo "hello world"
    return
}


# local-vars: script to demonstrate local variables 

foo=0 # global variable foo

funct_1 () {
  local foo      # variable foo local to funct_1
  foo=1
  echo "funct_1: foo = $foo"
}




echo "calling a function: $(add_)"

