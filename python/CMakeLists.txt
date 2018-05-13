find_package(PythonLibs REQUIRED)
find_package(PythonInterp REQUIRED)

add_library(pycudamat SHARED
	src/module.c
    src/matrix.c
)

if(WIN32)
	set_target_properties(pycudamat PROPERTIES
		SUFFIX ".pyd"
	)
endif()

include_directories(
	${PYTHON_INCLUDE_DIRS}
)

target_link_libraries(pycudamat
	${PYTHON_LIBRARIES}
	cudamat
)

install(
	TARGETS pycudamat
	DESTINATION ${CMAKE_SOURCE_DIR}/dist/build/lib
)
