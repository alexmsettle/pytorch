def define_targets(rules):
    rules.cc_library(
        name = "macros",
        hdrs = [
            "Macros.h",
            # Despite the documentation in Macros.h, Export.h is included
            # directly by many downstream files. Thus, we declare it as a
            # public header in this file.
            "Export.h",
        ],
        srcs = [":cmake_macros"],
        visibility = ["//visibility:public"],
    )

    rules.cmake_configure_file(
        name = "cmake_macros",
        src = "cmake_macros.h.in",
        out = "cmake_macros.h",
        definitions = {
            "C10_BUILD_SHARED_LIBS": True,
            "C10_USE_GFLAGS": "config_setting://c10:using_gflags",
            "C10_USE_GLOG": "config_setting://c10:using_glog",
            "C10_USE_MSVC_STATIC_RUNTIME": True,
            "C10_USE_NUMA": False,
        },
    )
