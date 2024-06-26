# Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

add_mfem_miniapp(nurbs_ex1
  MAIN nurbs_ex1.cpp
  LIBRARIES mfem)

add_mfem_miniapp(nurbs_curveint
  MAIN nurbs_curveint.cpp
  LIBRARIES mfem)

add_mfem_miniapp(nurbs_patch_ex1
  MAIN nurbs_patch_ex1.cpp
  LIBRARIES mfem)

if (MFEM_ENABLE_TESTING)
  add_test(NAME nurbs_ex1_1d_r1_o2_ser
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/segment-nurbs.mesh -r 1 -o 2 -lod 3)

  add_test(NAME nurbs_ex1_1d_r1_o2_wbc_ser
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/segment-nurbs.mesh -r 1 -o 2 -wbc -lod 3)

  add_test(NAME nurbs_ex1_r0_o4_ser
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis -r 0 -o 4)

  add_test(NAME nurbs_ex1_r2_ser
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis -r 2)

  add_test(NAME nurbs_ex1_per_ser
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/beam-hex-nurbs.mesh -pm 1 -ps 2)

  add_test(NAME nurbs_ex1_lap_r0_ser
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/pipe-nurbs-2d.mesh -o 2 -no-ibp -r 0)

  add_test(NAME nurbs_ex1_lap_r2_ser
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/pipe-nurbs-2d.mesh -o 2 -no-ibp -r 2)

  add_test(NAME nurbs_ex1_weak_r0_ser
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/pipe-nurbs-2d.mesh -o 2 --weak-bc -r 0)

  add_test(NAME nurbs_ex1_weak_r2_ser
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/pipe-nurbs-2d.mesh -o 2 --weak-bc -r 2)

  add_test(NAME nurbs_ex1_weak_mp_ser
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/ball-nurbs.mesh -o 2 --weak-bc -r 0)

  add_test(NAME nurbs_ex1_weak_patch_format_ser
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/square-disc-nurbs-patch.mesh -o 2 --weak-bc -r 0)

  add_test(NAME nurbs_ex1_weak_patch_format_r1_ser
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/square-disc-nurbs-patch.mesh -o 2 --weak-bc -r 1)

  if (MFEM_USE_LAPACK)
    add_test(NAME nurbs_patch_ex1_o4_r2_iro8_patcha_ser
      COMMAND $<TARGET_FILE:nurbs_patch_ex1> -incdeg 3 -ref 2 -iro 8 -patcha)
  endif()

  add_test(NAME nurbs_patch_ex1_o4_r2_iro8_patcha_pa_ser
    COMMAND $<TARGET_FILE:nurbs_patch_ex1> -incdeg 3 -ref 2 -iro 8 -patcha -pa)

  add_test(NAME nurbs_patch_ex1_o4_r2_iro8_patcha_fint_ser
    COMMAND $<TARGET_FILE:nurbs_patch_ex1> -incdeg 3 -ref 2 -iro 8 -patcha -fint)

  add_test(NAME nurbs_curveint_unit_weight_ser
    COMMAND $<TARGET_FILE:nurbs_curveint> -no-vis -no-visit
    -uw -n 9)

  add_test(NAME nurbs_curveint_non_unit_weight_ser
    COMMAND $<TARGET_FILE:nurbs_curveint> -no-vis -no-visit
    -nw -n 9)

  add_test(NAME nurbs_ex1_two_squares_knot_insert
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-squares-nurbs.mesh -o 1 -rf ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-squares.ref)

  add_test(NAME nurbs_ex1_two_squares_rot_knot_insert
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-squares-nurbs-rot.mesh -o 1 -rf ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-squares.ref)

  add_test(NAME nurbs_ex1_two_squares_autoedge_knot_insert
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-squares-nurbs-autoedge.mesh -o 1 -rf ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-squares.ref)

  add_test(NAME nurbs_ex1_two_cubes_knot_insert
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-cubes-nurbs.mesh -o 1 -r 3 -rf ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-cubes.ref)

  add_test(NAME nurbs_ex1_two_cubes_rot_knot_insert
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-cubes-nurbs-rot.mesh -o 1 -r 3 -rf ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-cubes.ref)

  add_test(NAME nurbs_ex1_two_cubes_autoedge_knot_insert
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-cubes-nurbs-autoedge.mesh -o 1 -r 3 -rf ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-cubes.ref)
endif()

if (MFEM_USE_MPI)
  add_mfem_miniapp(nurbs_ex1p
    MAIN nurbs_ex1p.cpp
    LIBRARIES mfem)

  if (MFEM_ENABLE_TESTING)
    add_test(NAME nurbs_ex1p_np=4
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:nurbs_ex1p> -no-vis
      ${MPIEXEC_POSTFLAGS})

    add_test(NAME nurbs_ex1p_lap_np=4
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:nurbs_ex1p> -no-vis -m
      ${PROJECT_SOURCE_DIR}/data/pipe-nurbs-2d.mesh -o 2 -no-ibp
      ${MPIEXEC_POSTFLAGS})

    add_test(NAME nurbs_ex1p_weak_mp_np=4
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:nurbs_ex1p> -no-vis
      -m ${PROJECT_SOURCE_DIR}/data/ball-nurbs.mesh -o 2 --weak-bc -r 0
      ${MPIEXEC_POSTFLAGS})

    add_test(NAME nurbs_ex1_weak_patch_format_np=4
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS}  $<TARGET_FILE:nurbs_ex1p> -no-vis
        -m ${PROJECT_SOURCE_DIR}/data/square-disc-nurbs-patch.mesh -o 2 --weak-bc -r 0)

    add_test(NAME nurbs_ex1_weak_patch_format_r1_np=4
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS}  $<TARGET_FILE:nurbs_ex1p> -no-vis
        -m ${PROJECT_SOURCE_DIR}/data/square-disc-nurbs-patch.mesh -o 2 --weak-bc -r 1)
  endif()

  add_mfem_miniapp(nurbs_ex11p
    MAIN nurbs_ex11p.cpp
    LIBRARIES mfem)

  if (MFEM_ENABLE_TESTING)
    add_test(NAME nurbs_ex11p_np=4
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:nurbs_ex11p> -no-vis
      ${MPIEXEC_POSTFLAGS})
  endif()
endif()
