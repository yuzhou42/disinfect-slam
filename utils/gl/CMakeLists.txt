add_library(gl_utils renderer_base.cc shader.cc image.cc)
target_link_libraries(gl_utils PUBLIC spdlog imgui ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${CUDA_LIBRARIES}
                            )
target_include_directories(gl_utils PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
