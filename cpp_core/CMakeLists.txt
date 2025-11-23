cmake_minimum_required(VERSION 3.15)
project(cpp_core LANGUAGES CXX)

# =============== 1. C++ 标准配置 ===============
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /EHsc")
    add_compile_options(/W3 /wd4267 /wd4244 /wd4146)
    add_compile_options(/source-charset:utf-8 /execution-charset:utf-8)
endif()

# =============== 2. 优先使用 backtest-env 的 Python（保留你的核心需求） ===============
set(PYTHON_ENV "C:/Users/Lenovo/anaconda3/envs/backtest-env")
set(Python3_ROOT_DIR ${PYTHON_ENV})
set(Python3_FIND_VIRTUALENV ONLY)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message("===== Python 配置 =====")
message("Python 可执行文件: ${Python3_EXECUTABLE}")
message("Python Include: ${Python3_INCLUDE_DIRS}")

# =============== 3. 自动搜索 pybind11（不依赖固定路径） ===============
# 优先从 Conda 环境找，找不到则自动搜索系统路径
find_package(pybind11 REQUIRED HINTS 
    "${PYTHON_ENV}/Lib/site-packages/pybind11/share/cmake/pybind11"
)
message("===== pybind11 配置 =====")
message("pybind11 找到: ${pybind11_FOUND}")
message("pybind11 Include: ${pybind11_INCLUDE_DIRS}")

# =============== 4. 自动搜索 Arrow + Arrow Python（关键：多路径自动查找） ===============
# 定义可能存放 Arrow 头文件的所有路径（覆盖 Conda、vcpkg、系统安装等场景）
set(ARROW_POSSIBLE_INCLUDE_DIRS
    "${PYTHON_ENV}/Lib/site-packages/pyarrow/include"  # Conda pyarrow 自带
    "${PYTHON_ENV}/Library/include"                   # Conda arrow-cpp
)

# 定义可能存放 Arrow 库文件的所有路径
set(ARROW_POSSIBLE_LIB_DIRS
    "${PYTHON_ENV}/Library/lib"                       # Conda arrow-cpp
    "${PYTHON_ENV}/Lib/site-packages/pyarrow"         # Conda pyarrow 自带库
)

# 自动查找 Arrow 头文件路径（找到第一个存在的路径）
find_path(ARROW_INCLUDE_DIR 
    NAMES arrow/table.h arrow/python/pyarrow.h  # 用关键头文件验证
    PATHS ${ARROW_POSSIBLE_INCLUDE_DIRS}
    REQUIRED  # 找不到则报错，提示安装
)

# 自动查找 Arrow 库文件路径
find_path(ARROW_LIB_DIR
    NAMES arrow.lib arrow_python.lib  # 用关键库文件验证
    PATHS ${ARROW_POSSIBLE_LIB_DIRS}
    REQUIRED
)

# 打印找到的 Arrow 路径（验证是否正确）
message("===== Arrow 配置 =====")
message("Arrow Include: ${ARROW_INCLUDE_DIR}")
message("Arrow Lib Dir: ${ARROW_LIB_DIR}")
message("Arrow include dirs found: ${ARROW_POSSIBLE_INCLUDE_DIRS}")
message("Arrow libraries found: ${ARROW_POSSIBLE_LIB_DIRS}")


# =============== 5. 源文件配置 ===============
set(SOURCES
    src/backtest.cpp
    src/utils.cpp
    src/bindings.cpp
)

# 链接库目录（自动指向找到的库路径）
link_directories(${ARROW_POSSIBLE_LIB_DIRS})

# =============== 6. Python 扩展模块编译 ===============
pybind11_add_module(cpp_core MODULE ${SOURCES})

# 头文件目录（自动包含找到的路径）
target_include_directories(cpp_core PRIVATE
    ${ARROW_POSSIBLE_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # 你自己的头文件
)

# 链接依赖（顺序不变，CMake 会自动在 ARROW_LIB_DIR 中找库）
target_link_libraries(cpp_core PRIVATE
    pybind11::pybind11
    Python3::Python
    arrow
    arrow_python
    parquet
)

# 输出 .pyd 路径（保留你的原配置）
set_target_properties(cpp_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../longgang_trader"
)