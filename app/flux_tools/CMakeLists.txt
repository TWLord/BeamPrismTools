SET(FLUX_APPS
  BuildFluxes
  FitFluxes
  OscillateFlux
  # PredictRunPlanEventRates -- Deprecated
  CombineBuiltFluxes
  BuildUncertaintyMatrix
  MakeLitedk2nu
  FluxSmusher
  FluxLinearSolver_Standalone
  EtrueErecoSolver
)

include_directories(${INCDIRS})
# Requried for fhiclcpp
include_directories(${CMAKE_INSTALL_PREFIX}/include)

foreach(appname ${FLUX_APPS})

  add_executable(dp_${appname} ${appname}.cxx)

  target_link_libraries(dp_${appname} ${MODULETargets})
  target_link_libraries(dp_${appname} ${ROOT_LIBS})

  add_dependencies(dp_${appname} spectra)

  if(NOT "${CMAKE_LINK_FLAGS}" STREQUAL "")
    set_target_properties(dp_${appname} PROPERTIES LINK_FLAGS ${CMAKE_LINK_FLAGS})
  endif()

  if(NOT "${CMAKE_DEPENDLIB_FLAGS}" STREQUAL "")
    target_link_libraries(dp_${appname} ${CMAKE_DEPENDLIB_FLAGS})
  endif()

  install(TARGETS dp_${appname} DESTINATION bin)

endforeach()
