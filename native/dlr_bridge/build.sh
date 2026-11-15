#!/bin/sh
# Builds dlr_bridge.dll (x64 Windows, no CRT) with clang + lld-link; output in build/.
set -e
cd "$(dirname "$0")"
mkdir -p build
for d in kernel32 ws2_32; do llvm-dlltool -m i386:x86-64 -d $d.def -l build/$d.lib; done
clang --target=x86_64-pc-windows-msvc -O2 -ffreestanding -fno-builtin -fno-stack-protector -mno-stack-arg-probe -Wall -Wextra -c bridge.c -o build/bridge.obj
lld-link /dll /nodefaultlib /entry:DllMain /machine:x64 /out:build/dlr_bridge.dll build/bridge.obj build/kernel32.lib build/ws2_32.lib
