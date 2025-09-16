#
# simple-ctx

#
# simple-backend

set(TEST_TARGET simple-backend2)
add_executable(${TEST_TARGET} simple-backend.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

target_compile_options(${TEST_TARGET} PRIVATE /utf-8)

if (GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()
