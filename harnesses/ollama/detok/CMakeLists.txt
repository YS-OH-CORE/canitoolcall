# ctc-detok: llama-server token rendering from a vocab-only GGUF, built against
# the llama.cpp tag Ollama pins in LLAMA_CPP_VERSION. Only ggml (CPU) + llama +
# llama-common are compiled; no server, no model weights.
#
# Configure with -DLLAMA_CPP_SOURCE=<llama.cpp checkout> (scripts/engines/ollama.sh).
#
# Ollama's llama/compat patch is not applied: it only adds metadata/tensor
# translation hooks to llama-model-loader.cpp and clip.cpp for Ollama-published
# GGUFs; tokenization and common_token_to_piece are unpatched upstream code.
cmake_minimum_required(VERSION 3.14)
project(ctc_ollama_detok C CXX)
set(CMAKE_CXX_STANDARD 17)

set(LLAMA_CPP_SOURCE "" CACHE PATH "llama.cpp checkout at Ollama's LLAMA_CPP_VERSION")
if(NOT LLAMA_CPP_SOURCE)
    message(FATAL_ERROR "set -DLLAMA_CPP_SOURCE=<llama.cpp checkout>")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_CPP_SOURCE} ${CMAKE_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)
add_executable(ctc-detok detok.cpp)
target_link_libraries(ctc-detok PRIVATE llama-common llama)
target_include_directories(ctc-detok PRIVATE ${LLAMA_CPP_SOURCE}/vendor ${LLAMA_CPP_SOURCE}/common)
