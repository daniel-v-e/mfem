#!/bin/bash

output_file="descriptions.txt"
rm -f $output_file

for file in $(ls examples/ex*.cpp | grep -v 'p\.cpp$' | sort -V); do
    echo -n "$(basename $file .cpp) " >> $output_file
    awk '/Description:/,/^[^\/]/{print}' $file | sed 's/\/\/ //' >> $output_file
    echo "" >> $output_file
done