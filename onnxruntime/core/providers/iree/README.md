# IREE Execution Provider

This directory contains the prototype IREE execution provider, implementing a full JIT-based flow. Until development
is further along, this README will contain roadmap and build instructions geared at the project developers.

## Development Roadmap

The EP is sufficient for functional testing but needs further development in the following ways:

* Execution is currently synchronous and "copy-ful". We need to switch to the async calling convention, implement
  memory transfer plumbing and switch to async scheduling. Based on prior experience, this is a day or two of
  work and the best time to do it is once functional testing has gotten underway with the simpler implementation.
* The EP was laid out to eventually support a "minimum" build that excludes the JIT compiler by having a
  "pre-compilation" mechanism. For now, we just try to keep the compiler support infra isolated from the runtime
  with the intersect point being VMFB binaries.
* Basic error handling and crash dump/reproducer handling is in place but should be extended to use env vars to
  log intermediates and control where reproducers are written (currently they just go got TMP). What is there should
  be relatively informative but needs more ergonomic support for shipping.
* Various snarky/TODO comments should be resolved and/or removed from the code.
* We need to plumb through an "--iree_home" style flag to point at a development-ready tree. For now, this has been
  hand assembled and hard-coded. See below.
* Device topology and corresponding compilation options have been hard-coded (to CPU/generic). This needs to be both
  controllable by options and better defaulted.
* FP8 dtypes have not been mapped.

With the above, this EP should be built out to relative completion and be suitable for upstreaming.

## Building

### IREE Dev Tree

This is the first time outside of a monorepo that we have used all of the IREE compiler, MLIR C API and IREE runtime
API together as part of a large external project. In order to do it, we have crafted an IREE "dev package" that is
currently manually assembled but should be automated. This lets us take a static/source dep on the runtime and
a shared library dep on the compiler (built separately) and avoid the common woes that come from building a large,
LLVM based combination.

This is currently done manually with the following commands (given an IREE source and build tree):

```
# Change paths here to correspond to your setup.
IREE_SRC_DIR=/home/stella/src/iree
IREE_BUILD_DIR=/home/stella/src/iree-build
DEV_PKG_DIR=/home/stella/src/onnxruntime-iree-bits
mkdir -p $DEV_PKG_DIR/include $DEV_PKG_DIR/lib
# Compiler lib and headers.
ln -s $IREE_BUILD_DIR/lib/libIREECompiler.so lib/
cp -R $IREE_SRC_DIR/compiler/bindings/c/iree $DEV_PKG_DIR/include
# TODO: MLIRInterop.h is in the private source tree. Make it public upstream to match here.
cp $IREE_SRC_DIR/compiler/src/iree/compiler/API/MLIRInterop.h $DEV_PKG_DIR/include/iree/compiler
cp -R $IREE_SRC_DIR/third_party/llvm-project/mlir/include/mlir-c $DEV_PKG_DIR/include
# Runtime lib and headers.
ln -s $IREE_BUILD_DIR/runtime/src/iree/runtime/libiree_runtime_unified.a $DEV_PKG_DIR/lib
ln -s $IREE_BUILD_DIR/build_tools/third_party/flatcc/libflatcc_runtime.a $DEV_PKG_DIR/lib
ln -s $IREE_BUILD_DIR/build_tools/third_party/flatcc/libflatcc_parsing.a $DEV_PKG_DIR/lib
for i in $(cd $IREE_SRC_DIR/runtime/src && find . -name '*.h'); do mkdir -p $DEV_PKG_DIR/include/$(dirname $i)/; cp $IREE_SRC_DIR/runtime/src/$i $DEV_PKG_DIR/include/$(dirname $i)/; done
```

NOTE: To use the above, IREE must have been built with `-DIREE_ENABLE_THIN_ARCHIVES=OFF` or else the
archives cannot just be copied around like this (the getting started guide recommends to have it on to aid
development). If you get it wrong, you'll get strange linker failures complaining about missing object files.

This is ripe for some upstream packaging work but gets us going for now.

## Dev Builds

The following produces a reasonable development setup on Linux:

```
CC=clang CXX=clang++ LDFLAGS="-fuse-ld=lld" \
./build.sh --config=RelWithDebInfo --cmake_generator=Ninja \
    --use_iree --cmake_extra_defines "ONNXRUNTIME_IREE_HOME=/home/stella/src/onnxruntime-iree-bits" \
    --use_cache --use_full_protobuf \
    --enable_symbolic_shape_infer_tests \
    --update --build
```

Explanations:

* `--cmake_generator` because it is 2024
* `--cmake_extra_defines`: Injects the IREE dev package directory. This should be plumbed through eventually as
  `--iree_home` to correspond with the others
* `--use_cache` enable ccache
* `--use_full_protobuf` enables protobuf metadata to make debug printing nicer
* `--enable_symbolic_shape_infer_tests` the EP works on ONNX models which have had symbolic shape inference run
  on them. This sets up the build to do this automatically.

Once the build has been initially done, you can work in the `build/Linux/RelWithDebInfo` directory if that is more
comfortable.

## Manual Testing

While the ORT test runner has some automatic abilities to run symbolic shape inference on `.onnx` files prior to
execution, if manually iterating, this may need to be done by hand. Here is the basic steps, taking
`testdata/transform/gemm_activation_fusion` as an example.

This was done in a Python venv with the following:

```
mpmath==1.3.0
numpy==1.26.2
onnx==1.15.0
packaging==23.2
protobuf==4.25.1
sympy==1.12
```

```
# From build dir.
mkdir -p symbolic_testdata
cp -R testdata/transform/gemm_activation_fusion symbolic_testdata

python ../../../onnxruntime/python/tools/symbolic_shape_infer.py --verbose 1 \
    --input symbolic_testdata/gemm_activation_fusion/gemm_activation_fusion.onnx \
    --output symbolic_testdata/gemm_activation_fusion/gemm_activation_fusion.onnx
```

It should then be possible to run the test with our EP:

```
./onnx_test_runner -e iree -v symbolic_testdata/gemm_activation_fusion
```
