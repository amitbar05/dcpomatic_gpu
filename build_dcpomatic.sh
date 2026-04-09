#!/bin/bash
set -e

DEPS=/home/amit/dcp-o-matic-gpu/deps/install

export PKG_CONFIG_PATH=$DEPS/lib64/pkgconfig:$DEPS/lib/pkgconfig:$PKG_CONFIG_PATH
export CPATH=$DEPS/include
export LIBRARY_PATH=$DEPS/lib:$DEPS/lib64:$LIBRARY_PATH
export LDFLAGS="-L$DEPS/lib -L$DEPS/lib64 -Wl,-rpath,$DEPS/lib -Wl,-rpath,$DEPS/lib64"

cd /home/amit/dcp-o-matic-gpu
python3 ./waf build -j$(nproc) 2>&1
echo "Exit code: $?"
