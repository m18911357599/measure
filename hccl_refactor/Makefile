CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude
BUILD ?= build

.PHONY: all test clean

all: $(BUILD)/test_refactor

$(BUILD)/test_refactor: tests/test_refactor.cc include/hccl_refactor/*.h
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) tests/test_refactor.cc -o $@

test: $(BUILD)/test_refactor
	$(BUILD)/test_refactor

clean:
	rm -rf $(BUILD)
