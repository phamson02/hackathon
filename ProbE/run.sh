#!/bin/bash

# Compile the solution
hipcc --std=c++17 -Wall -Wno-unused-result -O2 -static -pipe -c sol.cpp -o sol.o
hipcc --std=c++17 -Wall -Wno-unused-result -O2 -static -pipe -c main.cpp -o main.o
hipcc --std=c++17 -o main main.o sol.o

passed=0
total_with_ans=0

echo "Testing cases with answer files (validation enabled):"
# Test cases 001-009 (have both .in and .ans files)
for i in {001..009}; do
    if [ -f "data/${i}.in" ] && [ -f "data/${i}.ans" ]; then
        echo -n "Test ${i}: "
        srun --time=00:00:05 ./main < data/${i}.in > submission.txt
        python3 validator.py data/${i}.in data/${i}.ans submission.txt
        result=$?
        if [ $result -eq 0 ]; then
            echo "PASSED"
            ((passed++))
        else
            echo "FAILED"
        fi
        ((total_with_ans++))
    fi
done

echo ""
echo "Testing cases without answer files (output only):"
# Test cases 010-015 (only have .in files)
for i in {010..015}; do
    if [ -f "data/${i}.in" ]; then
        echo -n "Test ${i}: "
        srun --time=00:00:05 ./main < data/${i}.in > submission.txt
        echo "OUTPUT GENERATED"
    fi
done

echo ""
echo "Summary: ${passed}/${total_with_ans} tests passed"

# Clean up
rm -f main.o sol.o main submission.txt