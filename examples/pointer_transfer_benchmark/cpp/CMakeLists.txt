add_executable(pointer_transfer_benchmark
  pointer_transfer_benchmark.cpp
)
target_link_libraries(pointer_transfer_benchmark
  PUBLIC
  holoscan::core
)

file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
  install(FILES pointer_transfer_benchmark.cpp CMakeLists.min.txt
    DESTINATION "${app_relative_dest_path}" COMPONENT holoscan-examples)
endif()

install(TARGETS pointer_transfer_benchmark
  DESTINATION "${app_relative_dest_path}" COMPONENT holoscan-examples)
