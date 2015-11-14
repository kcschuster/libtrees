# The name of this project is "Trees".  CMakeLists files in this project can
# refer to the root source directory of the project as ${TREES_SOURCE_DIR} and
# to the root binary directory of the project as ${TREES_BINARY_DIR}.

cmake_minimum_required(VERSION 2.4)
if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

PROJECT(TREES)

# Compiler and path settings:
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTIMIZATION_FLAGS}" CACHE INTERNAL "Compiler optimization flags.")
SET(EXECUTABLE_OUTPUT_PATH ${TREES_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for executables.")
SET(LIBRARY_OUTPUT_PATH ${TREES_BINARY_DIR}/lib CACHE INTERNAL "Single output directory for libraries.")
SET(OPTIMIZATION_FLAGS "-O3 -funroll-loops -Wall" CACHE STRING "Compiler optimization flags.")
SET(SWIG_DIR /usr/local CACHE STRING "directory containing SWIG.")


# Advanced options in the ccmake gui:

#Set up definitions and libraries:

#################################

#write a configure file 
FILE(WRITE ${TREES_SOURCE_DIR}/src/config.h "//configure file for trees\n")
FILE(APPEND ${TREES_SOURCE_DIR}/src/config.h "#define _TREES_SRC_PATH \"${TREES_SOURCE_DIR}\"\n")

#################################

#Set up source files:
AUX_SOURCE_DIRECTORY(src TREES_SRC)

#################################

#make trees.h header file:

FILE(GLOB HEADER_FILES "${TREES_SOURCE_DIR}/src/*.h")
STRING(REPLACE "${TREES_SOURCE_DIR}/src/trees.h;" "" HEADER_FILES "${HEADER_FILES}")
STRING(REPLACE "${TREES_SOURCE_DIR}/src/" "" HEADER_FILES "${HEADER_FILES}")
FILE(WRITE ${TREES_SOURCE_DIR}/src/trees.h "#ifndef _TREES_H\n#define _TREES_H\n\n")
FOREACH(_FILENAME ${HEADER_FILES})
	FILE(APPEND ${TREES_SOURCE_DIR}/src/trees.h "#include \"${_FILENAME}\"\n")
ENDFOREACH(_FILENAME ${HEADER_FILES})
FILE(APPEND ${TREES_SOURCE_DIR}/src/trees.h "\n#endif\n\n") 

#install headers:

FILE(GLOB files "${TREES_SOURCE_DIR}/src/*.h*")
INSTALL(FILES ${files} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/trees)

################################

# Create a library called "trees" 

ADD_LIBRARY(trees 
	${TREES_SRC} ${THIRD_PARTY_SRC}
)

TARGET_LINK_LIBRARIES(trees ${LIBRARIES_USED})

INSTALL(TARGETS trees
			RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
			LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
            ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)


#################################
#ADD_SUBDIRECTORY(unit_tests)
#################################