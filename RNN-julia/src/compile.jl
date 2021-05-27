#!/usr/bin/env julia --project --startup-file=no
using PackageCompiler
packages = [:Flux, :Plots]
sysimage_path = string("sys-", Sys.ARCH, "-", ENV["USER"], ".so")
snoopfile = tempname() * ".jl"
write(snoopfile, "using Plots\nplotly()\nplot(rand(5), rand(5))\nsavefig(tempname() * \".html\")\n")
create_sysimage(packages, sysimage_path = sysimage_path, precompile_execution_file = snoopfile)
str = raw"""
#!/bin/bash
PATH=$(echo "$PATH" | tr ":" "\n" | grep -v "$JULIA_PROJECT" | tr "\n" ":") \
julia -J $(dirname ${BASH_SOURCE[0]})/sys-$(arch)-${USER-$(whoami)}.so "$@"
"""
write("julia", str)
try chmod("julia", 0o775) catch end