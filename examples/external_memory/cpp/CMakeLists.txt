add_executable(external_memory
  external_memory.cpp
)

# Link against Holoscan core library
target_link_libraries(external_memory
  PRIVATE
  holoscan::core
)

file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(external_memory PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
  install(FILES external_memory.cpp
          DESTINATION "${app_relative_dest_path}"
          COMPONENT holoscan-examples)
  install(FILES CMakeLists.min.txt
          RENAME "CMakeLists.txt"
          DESTINATION "${app_relative_dest_path}"
          COMPONENT holoscan-examples)
endif()

install(TARGETS external_memory
        DESTINATION "${app_relative_dest_path}"
        COMPONENT holoscan-examples)

if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_CPP_EXTERNAL_MEMORY_TEST
           COMMAND external_memory
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          )
endif()
