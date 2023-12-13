cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(minishogi)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -MP -fpermissive -DUNICODE -DUSE_MAKEFILE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_SSE")

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif ()

list(APPEND CMAKE_PREFIX_PATH ../libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Threads REQUIRED)
set(LIBS ${LIBS} Threads::Threads)

set(SOURCES
    main.cpp
    network.cpp
    args.cpp
    types.cpp
    bitboard.cpp
    misc.cpp
    movegen.cpp
    position.cpp
    usi.cpp
    evaluate.cpp
    search.cpp
    extra/rp_cmd.cpp
    extra/user_test.cpp
)

add_executable(minishogi ${SOURCES})
target_link_libraries(minishogi "${TORCH_LIBRARIES}")
target_link_libraries(minishogi ${LIBS})
set_property(TARGET minishogi PROPERTY CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -DNDEBUG")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif ()

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET minishogi
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:minishogi>)
endif (MSVC)
