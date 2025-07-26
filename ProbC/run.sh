#!/bin/bash
hipcc --std=c++17 -Wall -Wno-unused-result -O2 -static -pipe -c sol.cpp -o sol.o
hipcc --std=c++17 -Wall -Wno-unused-result -O2 -static -pipe -c main.cpp -o main.o
hipcc --std=c++17 -o main main.o sol.o
srun --time=00:00:05 ./main < data/001.in > submission.txt
python3 validator.py data/001.in data/001.ans submission.txt
echo $?
rm main.o sol.o main