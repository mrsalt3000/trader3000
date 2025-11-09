#!/bin/bash
set -e

# Сначала устанавливаем зависимости
pip install -r tests/static/requirements.txt > /dev/null

# Явно используем bash для запуска, чтобы он понял наши команды
exec bash tests/static/run.sh "$@"
