#!/bin/bash
set -e

# Переходим в корень проекта
cd "$(dirname "$0")/../../"

# Очищаем первый аргумент от возможного \r с Windows
ARG1=${1%$'\r'}

if [ "$ARG1" == "fix" ]; then
  echo "Running autoflake (surgeon)..."
  # Удаляем неиспользуемые импорты и переменные
  autoflake --in-place --remove-all-unused-imports --remove-unused-variables --recursive src/

  echo "Running formatters (black, isort)..."
  black src/
  isort src/
  echo "Formatting complete."
else
  echo "Running black..."
  black --check --diff src/

  echo "Running isort..."
  isort --check-only --diff src/

  echo "Running flake8..."
  flake8 src/

  echo "Running mypy..."
  mypy src/
fi
