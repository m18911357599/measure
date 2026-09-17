#pragma once

#include "types.hpp"

#include <algorithm>
#include <cassert>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace hccl_refactor {

// All-rank in-process simulator.  Real HCCL issues TxAsync/RxAsync on Transport
// links; this world only exists so the template composition can be unit-tested
// without CANN.  The engine/policy split matches the intended device path.
struct World {
    int n_ranks = 0;
    int n_chunks = 0;
    int chunk_size = 1;
    std::vector<std::vector<Value>> buf;  // [rank][n_chunks * chunk_size]

    int n_elems() const { return n_chunks * chunk_size; }

    static World make(int n_ranks, int chunk_size = 4)
    {
        World w;
        w.n_ranks = n_ranks;
        w.n_chunks = n_ranks;
        w.chunk_size = chunk_size;
        w.buf.assign(static_cast<size_t>(n_ranks),
            std::vector<Value>(static_cast<size_t>(n_ranks * chunk_size), 0));
        return w;
    }

    Value& at(int rank, int chunk, int e)
    {
        return buf[static_cast<size_t>(rank)][static_cast<size_t>(chunk * chunk_size + e)];
    }

    Value at(int rank, int chunk, int e) const
    {
        return buf[static_cast<size_t>(rank)][static_cast<size_t>(chunk * chunk_size + e)];
    }

    void fill_rank_identity()
    {
        for (int r = 0; r < n_ranks; ++r) {
            for (int c = 0; c < n_chunks; ++c) {
                for (int e = 0; e < chunk_size; ++e) {
                    at(r, c, e) = static_cast<Value>(r + 1) * 1000 + c * 10 + e;
                }
            }
        }
    }

    // AllReduce expected: every rank holds sum_r fill(r, c, e)
    Value expected_allreduce(int src_rank_count, int chunk, int e) const
    {
        Value s = 0;
        for (int r = 0; r < src_rank_count; ++r) {
            s += static_cast<Value>(r + 1) * 1000 + chunk * 10 + e;
        }
        return s;
    }

    std::string dump_rank(int r) const
    {
        std::ostringstream os;
        os << "rank " << r << ":";
        for (int c = 0; c < n_chunks; ++c) {
            os << " [";
            for (int e = 0; e < chunk_size; ++e) {
                if (e) {
                    os << ',';
                }
                os << at(r, c, e);
            }
            os << ']';
        }
        return os.str();
    }
};

// Lock-step exchange used by every topology engine.  Compiles to a single
// non-virtual function; Topo/Op differences are all inlined at the call site.
template <typename Op>
void commit_step(World& w, const std::vector<StepPlan>& plans)
{
    const int n = w.n_ranks;
    const std::vector<std::vector<Value>> snapshot = w.buf;

    auto snap_at = [&](int rank, int chunk, int e) -> Value {
        return snapshot[static_cast<size_t>(rank)]
                       [static_cast<size_t>(chunk * w.chunk_size + e)];
    };

    for (int r = 0; r < n; ++r) {
        const StepPlan& p = plans[static_cast<size_t>(r)];
        if (p.recv_from < 0) {
            continue;
        }
        const StepPlan& src = plans[static_cast<size_t>(p.recv_from)];
        if (src.send_to != r) {
            throw std::runtime_error("topology plan is not a matching permutation");
        }
        const size_t k = std::min(src.tx_chunks.size(), p.rx_chunks.size());
        for (size_t i = 0; i < k; ++i) {
            const int src_chunk = src.tx_chunks[i];
            const int dst_chunk = p.rx_chunks[i];
            for (int e = 0; e < w.chunk_size; ++e) {
                const Value v = snap_at(p.recv_from, src_chunk, e);
                if (Op::kReduce) {
                    w.at(r, dst_chunk, e) += v;
                } else {
                    w.at(r, dst_chunk, e) = v;
                }
            }
        }
    }
}

}  // namespace hccl_refactor
