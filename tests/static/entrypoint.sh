#!/bin/bash
set -e

# Явно используем bash для запуска, чтобы он понял наши команды
exec bash tests/static/run.sh "$@"
