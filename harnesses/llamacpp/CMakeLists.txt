# canitoolcall-llamacpp: offline replay of raw model outputs through llama.cpp's
# REAL chat parser (common/chat.h), driven over JSON lines by
# src/canitoolcall/adapters/llamacpp.py.
#
# Builds only ggml (CPU) + llama + llama-common + this tool. No model weights and
# no GPU backend are needed: the parser is built from the chat template and text
# is detokenized with a vocab-only GGUF.
#
#   cmake -S harnesses/llamacpp -B .engines/llamacpp/build -DCMAKE_BUILD_TYPE=Release \
#         -DLLAMA_CPP_DIR=.engines/llamacpp/llama.cpp
#   cmake --build .engines/llamacpp/build --target canitoolcall-llamacpp
#
# (scripts/engines/llamacpp.sh does exactly this at the pinned commit.)
cmake_minimum_required(VERSION 3.14)
project(canitoolcall_llamacpp C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.engines/llamacpp/llama.cpp"
    CACHE PATH "llama.cpp source checkout (pinned by scripts/engines/llamacpp.sh)")
if(NOT EXISTS "${LLAMA_CPP_DIR}/common/chat.h")
    message(FATAL_ERROR "llama.cpp not found at ${LLAMA_CPP_DIR}; run scripts/engines/llamacpp.sh")
endif()

set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)
set(GGML_METAL           OFF CACHE BOOL "" FORCE)
set(GGML_BLAS            OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE          OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_CPP_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp")

add_executable(canitoolcall-llamacpp replay.cpp)
target_link_libraries(canitoolcall-llamacpp PRIVATE llama-common llama)
target_include_directories(canitoolcall-llamacpp PRIVATE "${LLAMA_CPP_DIR}/common" "${LLAMA_CPP_DIR}/vendor")
