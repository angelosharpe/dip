#!/usr/bin/env bash

output="documentation"

exec epydoc --html classifier/ -o $output -v --graph=all --show-imports --show-sourcecode
