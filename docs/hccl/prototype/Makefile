CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude

.PHONY: test clean

test: /tmp/hccl_algo_proto_test
	/tmp/hccl_algo_proto_test

/tmp/hccl_algo_proto_test: tests/test_compose.cpp include/hccl_algo/*.hpp
	$(CXX) $(CXXFLAGS) tests/test_compose.cpp -o $@

clean:
	rm -f /tmp/hccl_algo_proto_test
