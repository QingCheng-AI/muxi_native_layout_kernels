#pragma once

#include <map>
#include <string>
#include <tuple>

namespace muxi_layout_kernels {

class LayoutGemmArgSelector {
  public:
    LayoutGemmArgSelector();
    ~LayoutGemmArgSelector() {}

    std::tuple<int, int, int, int> getArgs(std::tuple<int, int, int> key);

  private:
    std::string get_dispatch_file_path();

  private:
    std::map<std::tuple<int, int, int>, std::tuple<int, int, int, int>>
        gemmArgMap;
};

LayoutGemmArgSelector &getGlobalGemmArgSelector();

class ContinuousGemmArgSelector {
  public:
    ContinuousGemmArgSelector();
    ~ContinuousGemmArgSelector() {}

    std::tuple<int, int, int, int> getArgs(std::tuple<int, int, int> key);

  private:
    std::string get_dispatch_file_path();

  private:
    std::map<std::tuple<int, int, int>, std::tuple<int, int, int, int>>
        gemmArgMap;
};

ContinuousGemmArgSelector &getGlobalConCGemmArgSelector();

class LayoutGemvArgSelector {
  public:
    LayoutGemvArgSelector();
    ~LayoutGemvArgSelector() {}

    std::tuple<int, int, int, int> getArgs(std::tuple<int, int> key);

  private:
    std::string get_dispatch_file_path();

  private:
    std::map<std::tuple<int, int>, std::tuple<int, int, int, int>> gemvArgMap;
};

LayoutGemvArgSelector &getGlobalGemvArgSelector();

class LayoutGemmJustLayoutAArgSelector {
  public:
    LayoutGemmJustLayoutAArgSelector();
    ~LayoutGemmJustLayoutAArgSelector() {}

    std::tuple<int, int, int, int> getArgs(std::tuple<int, int, int> key);

  private:
    std::string get_dispatch_file_path();

  private:
    std::map<std::tuple<int, int, int>, std::tuple<int, int, int, int>>
        gemmJustLayoutAArgMap;
};

LayoutGemmJustLayoutAArgSelector &getGlobalGemmJustLayoutAArgSelector();

class LayoutGemmJustLayoutASoftFp8ArgSelector {
  public:
    LayoutGemmJustLayoutASoftFp8ArgSelector();
    ~LayoutGemmJustLayoutASoftFp8ArgSelector() {}

    std::tuple<int, int, int, int> getArgs(std::tuple<int, int, int> key);

  private:
    std::string get_dispatch_file_path();

  private:
    std::map<std::tuple<int, int, int>, std::tuple<int, int, int, int>>
        gemmJustLayoutASoftFp8ArgMap;
};

LayoutGemmJustLayoutASoftFp8ArgSelector &
getGlobalGemmJustLayoutASoftFp8ArgSelector();
class LayoutSoftFp8GemvArgSelector {
  public:
    LayoutSoftFp8GemvArgSelector();
    ~LayoutSoftFp8GemvArgSelector() {}

    std::tuple<int, int, int, int> getArgs(std::tuple<int, int> key);

  private:
    std::string get_dispatch_file_path();

  private:
    std::map<std::tuple<int, int>, std::tuple<int, int, int, int>>
        softFp8GemvArgMap;
};

LayoutSoftFp8GemvArgSelector &getGlobalSoftFp8GemvArgSelector();

class LayoutBf16GemvArgSelector {
  public:
    LayoutBf16GemvArgSelector();
    ~LayoutBf16GemvArgSelector() {}

    std::tuple<int, int, int, int> getArgs(std::tuple<int, int> key);

  private:
    std::string get_dispatch_file_path();

  private:
    std::map<std::tuple<int, int>, std::tuple<int, int, int, int>>
        bf16GemvArgMap;
};

LayoutBf16GemvArgSelector &getGlobalBf16GemvArgSelector();

} // namespace muxi_layout_kernels
