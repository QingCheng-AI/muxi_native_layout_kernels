#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "arg_selector.h"

namespace muxi_layout_kernels {

LayoutGemmArgSelector::LayoutGemmArgSelector() {
    // 打开文件
    std::string filename = get_dispatch_file_path();
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        std::cerr << "All layoutC GEMM shape will fallback to GEMM args 1,1,1,1"
                  << std::endl;
    }

    // 读取文件内容
    std::string line;
    std::getline(file, line); // 跳过表头行
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::vector<int> values;

        // 读取每一行的值
        while (std::getline(ss, token, ',')) {
            values.push_back(std::stoi(token));
        }

        // 确保每行有7个值
        if (values.size() != 7) {
            std::cerr << "Invalid line: " << line << std::endl;
            continue;
        }

        // 前三列作为键，后四列作为值
        std::tuple<int, int, int> key = {values[0], values[1], values[2]};
        std::tuple<int, int, int, int> value = {values[3], values[4], values[5],
                                                values[6]};

        // 插入到map中
        gemmArgMap[key] = value;
    }
}

std::tuple<int, int, int, int>
LayoutGemmArgSelector::getArgs(std::tuple<int, int, int> key) {
    auto args = gemmArgMap.find(key);
    if (args != gemmArgMap.end()) {
        // std::cerr << "DEBUG: GEMM shape: " << std::get<0>(key) << ", "
        //           << std::get<1>(key) << ", " << std::get<2>(key)
        //           << " get GEMM args: " << std::get<0>(args->second) << ", "
        //           << std::get<1>(args->second) << ", "
        //           << std::get<2>(args->second) << std::endl;
        return args->second;
    } else {
        std::cerr
            << " Warning: No matching arguments found for layout GEMM shape: "
            << std::get<0>(key) << ", " << std::get<1>(key) << ", "
            << std::get<2>(key) << ". Fallback GEMM args to 1,1,1,1"
            << std::endl;
        // Insert default args to avoid future warnings
        gemmArgMap[key] = {1, 1, 1, 1};
        return std::make_tuple(1, 1, 1, 1);
    }
}

std::string LayoutGemmArgSelector::get_dispatch_file_path() {
    // Deprecated, but we need to support Python 3.8.
    // importlib.resources is preferred in the future.
    namespace py = pybind11;
    py::module importlib_resources = py::module::import("pkg_resources");
    py::object path = importlib_resources.attr("resource_filename")(
        "muxi_layout_kernels", "layout_gemm_dispatch_arg.csv");
    return path.cast<std::string>();
}

LayoutGemmArgSelector &getGlobalGemmArgSelector() {
    static LayoutGemmArgSelector layoutGemmArgSelector;
    return layoutGemmArgSelector;
}

// ContinuousC GEMM selector
ContinuousGemmArgSelector::ContinuousGemmArgSelector() {
    // 打开文件
    std::string filename = get_dispatch_file_path();
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        std::cerr
            << "All ContinuousC GEMM shape will fallback to GEMM args 1,1,1,1"
            << std::endl;
    }

    // 读取文件内容
    std::string line;
    std::getline(file, line); // 跳过表头行
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::vector<int> values;

        // 读取每一行的值
        while (std::getline(ss, token, ',')) {
            values.push_back(std::stoi(token));
        }

        // 确保每行有7个值
        if (values.size() != 7) {
            std::cerr << "Invalid line: " << line << std::endl;
            continue;
        }

        // 前三列作为键，后四列作为值
        std::tuple<int, int, int> key = {values[0], values[1], values[2]};
        std::tuple<int, int, int, int> value = {values[3], values[4], values[5],
                                                values[6]};

        // 插入到map中
        gemmArgMap[key] = value;
    }
}

std::tuple<int, int, int, int>
ContinuousGemmArgSelector::getArgs(std::tuple<int, int, int> key) {
    auto args = gemmArgMap.find(key);
    if (args != gemmArgMap.end()) {
        // std::cerr << "DEBUG: GEMM shape: " << std::get<0>(key) << ", "
        //           << std::get<1>(key) << ", " << std::get<2>(key)
        //           << " get GEMM args: " << std::get<0>(args->second) << ", "
        //           << std::get<1>(args->second) << ", "
        //           << std::get<2>(args->second) << std::endl;
        return args->second;
    } else {
        std::cerr << " Warning: No matching arguments found for continuousC "
                     "GEMM shape: "
                  << std::get<0>(key) << ", " << std::get<1>(key) << ", "
                  << std::get<2>(key) << ". Fallback GEMM args to 1,1,1"
                  << std::endl;
        // Insert default args to avoid future warnings
        gemmArgMap[key] = {1, 1, 1, 1};
        return std::make_tuple(1, 1, 1, 1);
    }
}

std::string ContinuousGemmArgSelector::get_dispatch_file_path() {
    // Deprecated, but we need to support Python 3.8.
    // importlib.resources is preferred in the future.
    namespace py = pybind11;
    py::module importlib_resources = py::module::import("pkg_resources");
    py::object path = importlib_resources.attr("resource_filename")(
        "muxi_layout_kernels", "continuous_gemm_dispatch_arg.csv");
    return path.cast<std::string>();
}

ContinuousGemmArgSelector &getGlobalConCGemmArgSelector() {
    static ContinuousGemmArgSelector continuousGemmArgSelector;
    return continuousGemmArgSelector;
}

// GEMV selector
LayoutGemvArgSelector::LayoutGemvArgSelector() {
    // 打开文件
    std::string filename = get_dispatch_file_path();
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        std::cerr << "All Fp16 GEMV shape will fallback to GEMV args 256,1,1,1"
                  << std::endl;
    }

    // 读取文件内容
    std::string line;
    std::getline(file, line); // 跳过表头行
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::vector<int> values;

        // 读取每一行的值
        while (std::getline(ss, token, ',')) {
            values.push_back(std::stoi(token));
        }

        // 确保每行有6个值
        if (values.size() != 6) {
            std::cerr << "Invalid line: " << line << std::endl;
            continue;
        }

        // 前两列作为键，后四列作为值
        std::tuple<int, int> key = {values[0], values[1]};
        std::tuple<int, int, int, int> value = {values[2], values[3], values[4],
                                                values[5]};

        // 插入到map中
        gemvArgMap[key] = value;
    }
}

std::tuple<int, int, int, int>
LayoutGemvArgSelector::getArgs(std::tuple<int, int> key) {
    auto args = gemvArgMap.find(key);
    if (args != gemvArgMap.end()) {
        // std::cerr << "DEBUG: GEMV shape: " << std::get<0>(key) << ", "
        //           << std::get<1>(key) << ", " << std::get<2>(key)
        //           << " get GEMV args: " << std::get<0>(args->second) << ", "
        //           << std::get<1>(args->second) << ", "
        //           << std::get<2>(args->second) << ", "
        //           << std::get<3>(args->second) << std::endl;
        return args->second;
    } else {
        std::cerr << " Warning: No matching arguments found for GEMV shape: "
                  << std::get<0>(key) << ", " << std::get<1>(key)
                  << ". Fallback GEMV args to 256,1,1,1" << std::endl;
        // Insert default args to avoid future warnings
        gemvArgMap[key] = {256, 1, 1, 1};
        return std::make_tuple(256, 1, 1, 1);
    }
}

std::string LayoutGemvArgSelector::get_dispatch_file_path() {
    // Deprecated, but we need to support Python 3.8.
    // importlib.resources is preferred in the future.
    namespace py = pybind11;
    py::module importlib_resources = py::module::import("pkg_resources");
    py::object path = importlib_resources.attr("resource_filename")(
        "muxi_layout_kernels", "layout_gemv_dispatch_arg.csv");
    return path.cast<std::string>();
}

LayoutGemvArgSelector &getGlobalGemvArgSelector() {
    static LayoutGemvArgSelector layoutGemvArgSelector;
    return layoutGemvArgSelector;
}

LayoutGemmJustLayoutAArgSelector::LayoutGemmJustLayoutAArgSelector() {
    // 打开文件
    std::string filename = get_dispatch_file_path();
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        std::cerr
            << "All JustLayoutA GEMM shape will fallback to GEMM args 1,1,1,1"
            << std::endl;
    }

    // 读取文件内容
    std::string line;
    std::getline(file, line); // 跳过表头行
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::vector<int> values;

        // 读取每一行的值
        while (std::getline(ss, token, ',')) {
            values.push_back(std::stoi(token));
        }

        // 确保每行有7个值
        if (values.size() != 7) {
            std::cerr << "Invalid line: " << line << std::endl;
            continue;
        }

        // 前三列作为键，后三列作为值
        std::tuple<int, int, int> key = {values[0], values[1], values[2]};
        std::tuple<int, int, int, int> value = {values[3], values[4], values[5],
                                                values[6]};

        // 插入到map中
        gemmJustLayoutAArgMap[key] = value;
    }
}

std::tuple<int, int, int, int>
LayoutGemmJustLayoutAArgSelector::getArgs(std::tuple<int, int, int> key) {
    auto args = gemmJustLayoutAArgMap.find(key);
    if (args != gemmJustLayoutAArgMap.end()) {
        return args->second;
    } else {
        std::cerr << " Warning: No matching arguments found for just LayoutA "
                     "GEMM shape: "
                  << std::get<0>(key) << ", " << std::get<1>(key) << ", "
                  << std::get<2>(key) << ". Fallback GEMM args to 1,1,1,1"
                  << std::endl;
        // Insert default args to avoid future warnings
        gemmJustLayoutAArgMap[key] = {1, 1, 1, 1};
        return std::make_tuple(1, 1, 1, 1);
    }
}

std::string LayoutGemmJustLayoutAArgSelector::get_dispatch_file_path() {
    // Deprecated, but we need to support Python 3.8.
    // importlib.resources is preferred in the future.
    namespace py = pybind11;
    py::module importlib_resources = py::module::import("pkg_resources");
    py::object path = importlib_resources.attr("resource_filename")(
        "muxi_layout_kernels", "layoutA_gemm_dispatch_arg.csv");
    return path.cast<std::string>();
}

LayoutGemmJustLayoutAArgSelector &getGlobalGemmJustLayoutAArgSelector() {
    static LayoutGemmJustLayoutAArgSelector layoutGemmJustLayoutAArgSelector;
    return layoutGemmJustLayoutAArgSelector;
}

LayoutGemmJustLayoutASoftFp8ArgSelector::
    LayoutGemmJustLayoutASoftFp8ArgSelector() {
    // 打开文件
    std::string filename = get_dispatch_file_path();
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        std::cerr << "All JustLayoutASoftFp8 GEMM shape will fallback to GEMM "
                     "args 1,1,1,1"
                  << std::endl;
    }

    // 读取文件内容
    std::string line;
    std::getline(file, line); // 跳过表头行
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::vector<int> values;

        // 读取每一行的值
        while (std::getline(ss, token, ',')) {
            values.push_back(std::stoi(token));
        }

        // 确保每行有7个值
        if (values.size() != 7) {
            std::cerr << "Invalid line: " << line << std::endl;
            continue;
        }

        // 前三列作为键，后三列作为值
        std::tuple<int, int, int> key = {values[0], values[1], values[2]};
        std::tuple<int, int, int, int> value = {values[3], values[4], values[5],
                                                values[6]};

        // 插入到map中
        gemmJustLayoutASoftFp8ArgMap[key] = value;
    }
}

std::tuple<int, int, int, int> LayoutGemmJustLayoutASoftFp8ArgSelector::getArgs(
    std::tuple<int, int, int> key) {
    auto args = gemmJustLayoutASoftFp8ArgMap.find(key);
    if (args != gemmJustLayoutASoftFp8ArgMap.end()) {
        return args->second;
    } else {
        std::cerr << " Warning: No matching arguments found for just "
                     "LayoutASoftFp8 GEMM shape: "
                  << std::get<0>(key) << ", " << std::get<1>(key) << ", "
                  << std::get<2>(key) << ". Fallback GEMM args to 1,1,1,1"
                  << std::endl;
        // Insert default args to avoid future warnings
        gemmJustLayoutASoftFp8ArgMap[key] = {1, 1, 1, 1};
        return std::make_tuple(1, 1, 1, 1);
    }
}

std::string LayoutGemmJustLayoutASoftFp8ArgSelector::get_dispatch_file_path() {
    // Deprecated, but we need to support Python 3.8.
    // importlib.resources is preferred in the future.
    namespace py = pybind11;
    py::module importlib_resources = py::module::import("pkg_resources");
    py::object path = importlib_resources.attr("resource_filename")(
        "muxi_layout_kernels", "layoutA_Soft_Fp8_gemm_dispatch_arg.csv");
    return path.cast<std::string>();
}

LayoutGemmJustLayoutASoftFp8ArgSelector &
getGlobalGemmJustLayoutASoftFp8ArgSelector() {
    static LayoutGemmJustLayoutASoftFp8ArgSelector
        layoutGemmJustLayoutASoftFp8ArgSelector;
    return layoutGemmJustLayoutASoftFp8ArgSelector;
}

// GEMV soft fp8 selector
LayoutSoftFp8GemvArgSelector::LayoutSoftFp8GemvArgSelector() {
    // 打开文件
    std::string filename = get_dispatch_file_path();
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        std::cerr
            << "All Soft Fp8 GEMV shape will fallback to GEMV args 256,1,1,1"
            << std::endl;
    }

    // 读取文件内容
    std::string line;
    std::getline(file, line); // 跳过表头行
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::vector<int> values;

        // 读取每一行的值
        while (std::getline(ss, token, ',')) {
            values.push_back(std::stoi(token));
        }

        // 确保每行有6个值
        if (values.size() != 6) {
            std::cerr << "Invalid line: " << line << std::endl;
            continue;
        }

        // 前两列作为键，后四列作为值
        std::tuple<int, int> key = {values[0], values[1]};
        std::tuple<int, int, int, int> value = {values[2], values[3], values[4],
                                                values[5]};

        // 插入到map中
        softFp8GemvArgMap[key] = value;
    }
}

std::tuple<int, int, int, int>
LayoutSoftFp8GemvArgSelector::getArgs(std::tuple<int, int> key) {
    auto args = softFp8GemvArgMap.find(key);
    if (args != softFp8GemvArgMap.end()) {
        // std::cerr << "DEBUG: GEMV shape: " << std::get<0>(key) << ", "
        //           << std::get<1>(key) << ", " << std::get<2>(key)
        //           << " get GEMV args: " << std::get<0>(args->second) << ", "
        //           << std::get<1>(args->second) << ", "
        //           << std::get<2>(args->second) << ", "
        //           << std::get<3>(args->second) << std::endl;
        return args->second;
    } else {
        std::cerr
            << " Warning: No matching arguments found for SoftFp8 GEMV shape: "
            << std::get<0>(key) << ", " << std::get<1>(key)
            << ". Fallback SoftFp8 GEMV args to 256,1,1,1" << std::endl;
        // Insert default args to avoid future warnings
        softFp8GemvArgMap[key] = {256, 1, 1, 1};
        return std::make_tuple(256, 1, 1, 1);
    }
}

std::string LayoutSoftFp8GemvArgSelector::get_dispatch_file_path() {
    // Deprecated, but we need to support Python 3.8.
    // importlib.resources is preferred in the future.
    namespace py = pybind11;
    py::module importlib_resources = py::module::import("pkg_resources");
    py::object path = importlib_resources.attr("resource_filename")(
        "muxi_layout_kernels", "layout_gemv_soft_fp8_dispatch_arg.csv");
    return path.cast<std::string>();
}

LayoutSoftFp8GemvArgSelector &getGlobalSoftFp8GemvArgSelector() {
    static LayoutSoftFp8GemvArgSelector layoutSoftFp8GemvArgSelector;
    return layoutSoftFp8GemvArgSelector;
}

// GEMV soft bf16 selector  // for muxi do not have really bfloat16 compute
// instuctions, so fp16 and bf16 maybe have different best params.
LayoutBf16GemvArgSelector::LayoutBf16GemvArgSelector() {
    // 打开文件
    std::string filename = get_dispatch_file_path();
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        std::cerr << "All Bf16 GEMV shape will fallback to GEMV args 256,1,1,1"
                  << std::endl;
    }

    // 读取文件内容
    std::string line;
    std::getline(file, line); // 跳过表头行
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::vector<int> values;

        // 读取每一行的值
        while (std::getline(ss, token, ',')) {
            values.push_back(std::stoi(token));
        }

        // 确保每行有6个值
        if (values.size() != 6) {
            std::cerr << "Invalid line: " << line << std::endl;
            continue;
        }

        // 前两列作为键，后四列作为值
        std::tuple<int, int> key = {values[0], values[1]};
        std::tuple<int, int, int, int> value = {values[2], values[3], values[4],
                                                values[5]};

        // 插入到map中
        bf16GemvArgMap[key] = value;
    }
}

std::tuple<int, int, int, int>
LayoutBf16GemvArgSelector::getArgs(std::tuple<int, int> key) {
    auto args = bf16GemvArgMap.find(key);
    if (args != bf16GemvArgMap.end()) {
        // std::cerr << "DEBUG: GEMV shape: " << std::get<0>(key) << ", "
        //           << std::get<1>(key) << ", " << std::get<2>(key)
        //           << " get GEMV args: " << std::get<0>(args->second) << ", "
        //           << std::get<1>(args->second) << ", "
        //           << std::get<2>(args->second) << ", "
        //           << std::get<3>(args->second) << std::endl;
        return args->second;
    } else {
        std::cerr
            << " Warning: No matching arguments found for Bf16 GEMV shape: "
            << std::get<0>(key) << ", " << std::get<1>(key)
            << ". Fallback Bf16 GEMV args to 256,1,1,1" << std::endl;
        // Insert default args to avoid future warnings
        bf16GemvArgMap[key] = {256, 1, 1, 1};
        return std::make_tuple(256, 1, 1, 1);
    }
}

std::string LayoutBf16GemvArgSelector::get_dispatch_file_path() {
    // Deprecated, but we need to support Python 3.8.
    // importlib.resources is preferred in the future.
    namespace py = pybind11;
    py::module importlib_resources = py::module::import("pkg_resources");
    py::object path = importlib_resources.attr("resource_filename")(
        "muxi_layout_kernels", "layout_gemv_bf16_dispatch_arg.csv");
    return path.cast<std::string>();
}

LayoutBf16GemvArgSelector &getGlobalBf16GemvArgSelector() {
    static LayoutBf16GemvArgSelector layoutBf16GemvArgSelector;
    return layoutBf16GemvArgSelector;
}

} // namespace muxi_layout_kernels
