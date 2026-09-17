#pragma once

#include "status.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace hccl_algo {

struct Plane {
    std::vector<int> members;  // local index → global rank
    int chunk = 1;             // blocks per logical slot
    int base = 0;              // block offset of logical 0 when chunk > 1

    int size() const { return static_cast<int>(members.size()); }
    int gid(int local) const { return members[static_cast<size_t>(local)]; }

    Slice slot(int logical, int extraOff = 0) const
    {
        // Intra-server (chunk==1): logical index maps to a global rank/block id.
        // Inter-server (chunk>1): logical index is the server/pod id, stride=chunk.
        if (chunk == 1) {
            return Slice{members[static_cast<size_t>(logical)] + extraOff, 1};
        }
        return Slice{base + logical * chunk + extraOff, chunk};
    }
};

inline Plane identityPlane(int p, int chunk = 1)
{
    Plane pl;
    pl.chunk = chunk;
    pl.members.resize(static_cast<size_t>(p));
    for (int i = 0; i < p; ++i) {
        pl.members[static_cast<size_t>(i)] = i;
    }
    return pl;
}

class World {
public:
    void reset(int nRanks, int nBlocks, int blockBytes)
    {
        nRanks_ = nRanks;
        nBlocks_ = nBlocks;
        blockBytes_ = blockBytes;
        mem_.assign(static_cast<size_t>(nRanks),
            std::vector<uint8_t>(static_cast<size_t>(nBlocks * blockBytes), 0));
        pending_.clear();
        txCount_ = 0;
        reduceCount_ = 0;
    }

    uint8_t* ptr(int rank) { return mem_[static_cast<size_t>(rank)].data(); }
    const uint8_t* ptr(int rank) const { return mem_[static_cast<size_t>(rank)].data(); }
    int blockBytes() const { return blockBytes_; }
    int nBlocks() const { return nBlocks_; }
    int nRanks() const { return nRanks_; }

    void postCopy(int srcRank, int dstRank, Slice src, Slice dst, bool reduce)
    {
        pending_.push_back(Post{srcRank, dstRank, src, dst, reduce});
        ++txCount_;
    }

    void flush()
    {
        for (const auto& p : pending_) {
            const int n = std::min(p.src.count, p.dst.count) * blockBytes_;
            uint8_t* d = ptr(p.dstRank) + p.dst.offset * blockBytes_;
            const uint8_t* s = ptr(p.srcRank) + p.src.offset * blockBytes_;
            if (!p.reduce) {
                std::memcpy(d, s, static_cast<size_t>(n));
            } else {
                ++reduceCount_;
                for (int i = 0; i < n; ++i) {
                    d[i] = static_cast<uint8_t>(d[i] + s[i]);
                }
            }
        }
        pending_.clear();
    }

    int txCount() const { return txCount_; }
    int reduceCount() const { return reduceCount_; }

private:
    struct Post {
        int srcRank;
        int dstRank;
        Slice src;
        Slice dst;
        bool reduce;
    };
    int nRanks_ = 0;
    int nBlocks_ = 0;
    int blockBytes_ = 1;
    std::vector<std::vector<uint8_t>> mem_;
    std::vector<Post> pending_;
    int txCount_ = 0;
    int reduceCount_ = 0;
};

struct Ctx {
    World* world = nullptr;
    const Plane* plane = nullptr;
    int rank = 0;
    int rankSize = 0;
    int channel = 0;

    int gid(int local) const { return plane->gid(local); }
    Slice slot(int logical) const { return plane->slot(logical); }
};

}  // namespace hccl_algo
