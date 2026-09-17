CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude

TEST_BIN := tests/test_refactor_design

.PHONY: test clean

test: $(TEST_BIN)
	$(TEST_BIN)

$(TEST_BIN): tests/test_refactor_design.cpp include/hccl_refactor.hpp include/hccl_refactor/*.hpp
	$(CXX) $(CXXFLAGS) tests/test_refactor_design.cpp -o $(TEST_BIN)

clean:
	rm -f $(TEST_BIN)
