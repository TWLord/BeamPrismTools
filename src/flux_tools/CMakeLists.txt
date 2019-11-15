add_library(DUNE-PRISM_FluxTools
  dk2nu_TreeReader.cxx
  OscillationHelper.cxx
  FluxLikelihood.cxx
  FluxFitter.cxx
  FluxCombiner.cxx
  EffectiveFluxUncertaintyHelper.cxx
  EtrueErecoSolver.cxx
)

if(DEFINED USE_FHICL AND USE_FHICL)
  add_dependencies(DUNE-PRISM_FluxTools fhiclcpp-simple)
endif()

include_directories(${INCDIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Requried for fhiclcpp
include_directories(${CMAKE_INSTALL_PREFIX}/include)

install(TARGETS DUNE-PRISM_FluxTools DESTINATION lib)

target_link_libraries(DUNE-PRISM_FluxTools ${ROOT_LIBS})

if(NOT "${CMAKE_LINK_FLAGS}" STREQUAL "")
  set_target_properties(DUNE-PRISM_FluxTools PROPERTIES LINK_FLAGS ${CMAKE_LINK_FLAGS})
endif()

if(NOT "${CMAKE_DEPENDLIB_FLAGS}" STREQUAL "")
  target_link_libraries(DUNE-PRISM_FluxTools ${CMAKE_DEPENDLIB_FLAGS})
endif()

SET(MODULETargets ${MODULETargets};DUNE-PRISM_FluxTools PARENT_SCOPE)
