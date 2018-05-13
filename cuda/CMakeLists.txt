find_package(CUDA REQUIRED)

add_library(cudamat STATIC
    src/matrix.cu
    include/matrix.cuh
)

target_include_directories(cudamat PUBLIC include)

set_target_properties(cudamat PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON # workaround for windows builds
)

target_link_libraries(cudamat ${CUDA_LIBRARIES})