load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

# maybe() doesn't load a 3rd party repository if it's already loaded.

def bazel_skylib():
    maybe(
        http_archive,
        name = "bazel_skylib",
        urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz"],
        sha256 = "f7be3474d42aae265405a592bb7da8e171919d74c16f082a5457840f06054728",
    )

# com_google_absl: Abseil.
def com_google_absl():
    maybe(
        http_archive,
        name = "com_google_absl",
        sha256 = "a3e25655bf03555bfd702ed9a1001af9d0aa44eb387a75e85757b15e72a05d44",
        strip_prefix = "abseil-cpp-6a876051b118c86e8bfa51961270055da5948813",
        urls = [
            "https://github.com/abseil/abseil-cpp/archive/6a876051b118c86e8bfa51961270055da5948813.zip",
        ],
    )

def com_google_googletest():
    maybe(
        http_archive,
        name = "com_google_googletest",
        sha256 = "353571c2440176ded91c2de6d6cd88ddd41401d14692ec1f99e35d013feda55a",
        strip_prefix = "googletest-release-1.11.0",
        urls = [
            "https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip",
        ],
    )
