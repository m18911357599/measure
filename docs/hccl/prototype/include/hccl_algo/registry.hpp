#pragma once

#include "hierarchical.hpp"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace hccl_algo {

struct RecipeInfo {
    std::string algName;
    std::string family;  // mesh / clos / clos+mesh / ring / double-ring / nhr / multi-channel / hierarchical
    int (*steps)(int p);
};

class RecipeRegistry {
public:
    static RecipeRegistry& instance()
    {
        static RecipeRegistry r;
        return r;
    }

    void add(const RecipeInfo& info) { map_[info.algName] = info; }

    const RecipeInfo* find(const std::string& name) const
    {
        auto it = map_.find(name);
        return it == map_.end() ? nullptr : &it->second;
    }

    std::vector<RecipeInfo> all() const
    {
        std::vector<RecipeInfo> v;
        v.reserve(map_.size());
        for (const auto& kv : map_) {
            v.push_back(kv.second);
        }
        return v;
    }

private:
    std::unordered_map<std::string, RecipeInfo> map_;
};

inline int stepsMesh(int p) { return Mesh::steps(p); }
inline int stepsRing(int p) { return Ring::steps(p); }
inline int stepsNhr(int p) { return Nhr::steps(p); }
inline int stepsDoubleRing(int p) { return DoubleRing::steps(p); }
inline int stepsClosMesh(int p)
{
    const int n0 = (p % 8 == 0) ? 8 : (p % 4 == 0) ? 4 : 2;
    return Hierarchical<Mesh, ClosNhr>::steps(p, n0);
}

struct RecipeRegistrar {
    RecipeRegistrar(const RecipeInfo& info) { RecipeRegistry::instance().add(info); }
};

#define HCCL_REGISTER_RECIPE(name, family, stepsfn) \
    static ::hccl_algo::RecipeRegistrar g_recipe_##stepsfn { \
        ::hccl_algo::RecipeInfo { name, family, stepsfn } \
    }

}  // namespace hccl_algo
