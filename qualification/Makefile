INPUT0=kittens.in # about some number of endpoints (530) per cache
INPUT1=me_at_the_zoo.in # small problem -> DP ?
INPUT2=videos_worth_spreading.in # either one or two endpoint in common for each server
INPUT3=trending_today.in # all caches serve all endpoints

INPUT=$(INPUT1)

main: main.cpp
	g++ -o main main.cpp -O3 -Wall -std=c++0x -Wextra -Wno-sign-compare

run: main 
	./main $(INPUT) 5   # timeout in sec

run-all: main
	./main $(INPUT1) 6
	./main $(INPUT2) 6
	./main $(INPUT3) 6
	./main $(INPUT0) 6

clean:
	rm -f main

clean-all: clean
	rm -f sol-*
