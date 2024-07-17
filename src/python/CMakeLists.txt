pybind11_add_module(wos_ext main.cpp
                            scene.cu
                            wos.cu
                            test.cpp)
target_link_libraries(wos_ext PRIVATE drjit-core drjit-autodiff)

if(MSVC)
  # https://github.com/microsoft/vcpkg/issues/30272#issuecomment-1817929528
  set_target_properties(wos_ext PROPERTIES COMPILE_OPTIONS -Xcompiler)
  set_target_properties(wos_ext
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_RELEASE        ${CMAKE_BINARY_DIR}/python/wos_ext
    LIBRARY_OUTPUT_DIRECTORY_DEBUG          ${CMAKE_BINARY_DIR}/python/wos_ext
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/python/wos_ext
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL     ${CMAKE_BINARY_DIR}/python/wos_ext)
else()
  set_target_properties(wos_ext
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/wos_ext
    FOLDER python)
endif()

################# copy data ####################
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/wos_ext)
set(WOS_PYTHON_FILES
  __init__.py)
add_custom_target(copy_wos_ext_data ALL
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/${WOS_PYTHON_FILES}
  ${CMAKE_BINARY_DIR}/python/wos_ext
  COMMENT "Copying wos ext file to build directory" 
)
add_dependencies(wos_ext copy_wos_ext_data)