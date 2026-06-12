cmake_minimum_required(VERSION 3.5)
project(sim LANGUAGES C CXX CUDA)

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=gnu++11>)
add_compile_options(-Wdeprecated-declarations)
SET(CMAKE_BUILD_TYPE Release CACHE STRING
	"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
#	FORCE
	)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

#find_package(TinyXML REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Boost #[[1.66.0]] COMPONENTS filesystem python3 numpy3)

add_compile_options(-DHAVE_CSTDDEF)
add_definitions(-DOCTOCON_DIR="${CMAKE_HOME_DIRECTORY}")
include_directories(${EIGEN3_INCLUDE_DIR})
#include_directories(${TinyXML_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})

link_directories(./fem/)

include(FindOpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(GLOB srcs "*.h" "*.cpp")
# 过滤测试文件
list(FILTER srcs EXCLUDE REGEX ".*test\\.cpp$")
file(GLOB fluid_srcs "fluid/*.h" "fluid/*.cu")
file(GLOB util_srcs "util/*.h" "util/*.cpp" "util/json/*.h" "util/json/*.cpp")
# find_package(DART REQUIRED COMPONENTS utils-urdf gui optimizer-ipopt CONFIG)
# add_compile_options(-DHAVE_CSTDDEF)
# include_directories(${DART_INCLUDE_DIRS})

add_library(sim ${srcs} ${fluid_srcs} ${util_srcs})
target_link_libraries(sim ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} CUDA::cudart #[[${DART_LIBRARIES}]] #[[${TinyXML_LIBRARIES}]] fem)

add_subdirectory(fem)

# ============ Fluid Coupling: C ABI 共享库 ============
# 编译 WormCInterface.cpp 为 fish.dll / fish.so，供 Julia ccall 调用
# fluid_coupling 位于 BAAIWorm/fluid_coupling
get_filename_component(WORKSPACE_ROOT "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)
set(FLUID_COUPLING_DIR "${WORKSPACE_ROOT}/BAAIWorm/fluid_coupling")
add_library(fish SHARED
    ${srcs} ${util_srcs}
    ${FLUID_COUPLING_DIR}/WormCInterface.cpp
)
target_include_directories(fish PRIVATE
    ${PROJECT_SOURCE_DIR}          # Metaworm_zfver/sim — 与 Worm.cpp 等源文件一致
    ${CMAKE_SOURCE_DIR}            # Metaworm_zfver 顶层（部分 util 头文件可能需要）
    ${EIGEN3_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${FLUID_COUPLING_DIR}
)
target_link_libraries(fish fem ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
set_target_properties(fish PROPERTIES
    PREFIX ""          # 去掉 lib 前缀
    OUTPUT_NAME "fish"
    # 固定输出到 sim/，与 FishCoupler.jl 的查找路径一致
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
# ============ Fluid Coupling: C ABI 共享库 ============

# ============ PeridynoBridge 单元测试 ============
add_executable(test_peridyno_bridge test_peridyno_bridge.test.cpp fluid/PeridynoBridge.cu)
target_include_directories(test_peridyno_bridge PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)
target_link_libraries(test_peridyno_bridge CUDA::cudart ${Boost_LIBRARIES})
set_target_properties(test_peridyno_bridge PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Headless 集成测试
add_executable(headless_test headless_test.test.cpp)

# 力和位移验证测试
add_executable(force_disp_test force_disp_test.cpp)
target_include_directories(force_disp_test PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)
target_link_libraries(force_disp_test sim CUDA::cudart ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
set_target_properties(force_disp_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# 力反馈验证测试
add_executable(force_check force_check.test.cpp)
target_include_directories(force_check PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)
target_link_libraries(force_check sim CUDA::cudart ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
target_include_directories(headless_test PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)
target_link_libraries(headless_test sim CUDA::cudart ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
set_target_properties(headless_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# FSI 稳定性测试（sin 肌肉 + 体积/翻转指标）
add_executable(fsi_stability fsi_stability.test.cpp)
target_include_directories(fsi_stability PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)
target_link_libraries(fsi_stability sim CUDA::cudart ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
set_target_properties(fsi_stability PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# FSI 顶点力快照（5 个时刻导出 + 供 Python 可视化）
add_executable(fsi_force_snapshot fsi_force_snapshot.test.cpp)
target_include_directories(fsi_force_snapshot PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)
target_link_libraries(fsi_force_snapshot sim CUDA::cudart ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
set_target_properties(fsi_force_snapshot PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# 无偏位移对比（contact+hydro vs hydro-only）
add_executable(fsi_unbiased_disp fsi_unbiased_disp.test.cpp)
target_include_directories(fsi_unbiased_disp PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)
target_link_libraries(fsi_unbiased_disp sim CUDA::cudart ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
set_target_properties(fsi_unbiased_disp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
# ============ PeridynoBridge 单元测试 ============

set(WORM_SIM_INCLUDES 
    ${PROJECT_SOURCE_DIR} 
    ${EIGEN3_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS} PARENT_SCOPE)
