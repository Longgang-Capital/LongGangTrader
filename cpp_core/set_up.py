#这里为cpp_core对应的python拓展配置
# cpp_core/setup.py
import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# 自定义编译命令：调用 CMake 编译
class CMakeBuild(build_ext):
    def run(self):
        # 1. 检查 CMake 是否安装
        try:
            self.spawn(["cmake", "--version"])
        except OSError:
            raise RuntimeError("请先安装 CMake（https://cmake.org/download/）")

        # 2. 创建编译目录（build）
        build_dir = os.path.join(os.path.dirname(__file__), "build")
        os.makedirs(build_dir, exist_ok=True)

        # 3. 执行 CMake 配置（Release 模式）
        cmake_cmd = [
            "cmake",
            "-S", os.path.dirname(__file__),  # 源文件目录（cpp_core）
            "-B", build_dir,                  # 编译输出目录
            f"-DCMAKE_BUILD_TYPE=Release",    # 编译模式（必须 Release，匹配 pyarrow）
            f"-DCMAKE_INSTALL_PREFIX={self.build_lib}",  # 安装目录
        ]
        self.spawn(cmake_cmd)

        # 4. 执行编译
        build_cmd = [
            "cmake",
            "--build", build_dir,
            "--config", "Release",
            "--target", "cpp_core",
            "--parallel", "4"  # 4 线程编译（可调整）
        ]
        self.spawn(build_cmd)

        # 5. 复制编译后的 .pyd 文件到目标目录
        src_pyd = os.path.join(build_dir, "Release", "cpp_core.pyd")
        dst_pyd = os.path.join(self.build_lib, "cpp_core.pyd")
        self.copy_file(src_pyd, dst_pyd)

# 定义扩展模块
ext_modules = [
    Extension(
        name="cpp_core",  # 模块名（Python 导入时用 import cpp_core）
        sources=[],  # 源文件由 CMake 管理，此处留空
        language="C++"
    )
]

#  setup 配置
setup(
    name="cpp_core",
    version="0.1.0",
    description="LongGangTrader C++ 高性能回测核心",
    author="Your Name",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},  # 关联自定义编译命令
    zip_safe=False,
    python_requires=">=3.9",  # 匹配你的 Conda 环境 Python 版本
)