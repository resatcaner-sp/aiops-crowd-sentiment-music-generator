# Variables
targets := "src/ tests/"
app_env := env_var_or_default("APP_ENVIRONMENT", "local")
app_config_path := env_var_or_default("APP_CONFIG_PATH", justfile_directory())

# Uncomment following line if you use Windows environment. Make sure git bash is installed (assuming git is installed already if you are here:)
# set shell := ["C:\\Program Files\\Git\\bin\\bash.exe", "-cu"]

# Default recipe
_default:
    just fix "all"
    just check "all"
    just test

# Show all available recipes
help:
    @just --list --list-prefix "···· "

# Runs the linter and formatter (fix enabled). Optionally use the argument "all" to run all fixable pre-commit hooks.
fix *arg: (lint "--fix") format
    @if [ '{{ arg }}' = 'all' ]; then \
        just _pre-commit "poetry-lock" "sync_with_poetry"; \
    fi

# Runs the linter, formatter, and mypy. Optionally use the argument "all" to run all checkable pre-commit hooks.
check *arg: lint (format "--check") mypy
    @if [ '{{ arg }}' = 'all' ]; then \
        just _pre-commit "poetry-check"; \
    fi

# Lint the code
lint *args:
    poetry run ruff check {{ targets }} {{ args }}
# Format the code
format *args:
    poetry run ruff format {{ targets }} {{ args }}
# Run mypy to check for type errors
mypy *args:
    poetry run mypy {{ targets }} {{ args }}
# Run all pre-commit hooks on all files
pre-commit-all:
    poetry run pre-commit run --all-files

_pre-commit +hooks:
    @for hook in {{ hooks }}; do \
        poetry run pre-commit run $hook --all-files; \
    done;

# Run tests with given markers. All tests are run if no marker given.
test *markers:
    @if [ -z {{ markers }} ]; then \
        just _test; \
    else \
        for marker in {{ markers }}; do just _test "-m $marker"; done; \
        \
    fi

_test *marker:
    poetry run pytest --random-order --cov {{ marker }}
# Build the package
build:
    poetry lock --no-update  # Update the lock file
    poetry install           # Install the project dependencies
    poetry build             # Build the package
