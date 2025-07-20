# Code Standards & Conventions

## Python Version
- Python 3.12+ is required for all code
- Type annotations are mandatory for all functions and methods

## Programming Paradigms
- **Functional Approach**: Prefer functional, declarative programming over object-oriented where possible
- **Modularization**: Break down complex logic into smaller, reusable functions
- **Iteration**: Favor iteration over code duplication
- **RORO Pattern**: Receive an Object, Return an Object for consistent function interfaces

## Code Style & Formatting
- **Line Length**: 120 characters maximum
- **Quote Style**: Double quotes for strings (enforced by Ruff)
- **Indentation**: 4 spaces (no tabs)
- **Docstring Format**: Google style (enforced by Ruff)
- **Import Order**: Sorted by type with known first-party modules defined
- **Naming Conventions**: 
  - Use descriptive variable names with auxiliary verbs (e.g., `is_active`, `has_permission`)
  - Use lowercase with underscores for directories and files (e.g., `routers/user_routes.py`)
  - Favor named exports for routes and utility functions

## Type Checking
- **MyPy Configuration**: Strict mode enabled
- All function definitions must have type annotations (`disallow_untyped_defs = true`)
- Type checking is performed on the interior of functions without annotations
- Warnings for functions that return `Any` when another return type is specified
- Warnings for unused `# type: ignore` comments

## Linting Rules
- **Ruff Checks**: Multiple rule sets enabled including:
  - A: Prevent using keywords that clobber Python builtins
  - B: Bugbear security warnings
  - C: McCabe complexity (max complexity = 5)
  - D: Pydocstyle (Google convention)
  - E/F/W: Standard PyCodeStyle/PyFlakes/Warnings
  - I: Import sorting
  - UP: Python version-specific syntax improvements
  - And many more specialized rules (ARG, PIE, RET, SIM, etc.)

## Docstrings
- **Required For**: All public modules, classes, methods, and functions
- **Format**: Google docstring style
  ```python
  def function(param1: str, param2: int) -> bool:
      """Short description of function.
      
      Longer description if needed.
      
      Args:
          param1: Description of param1
          param2: Description of param2
          
      Returns:
          Description of return value
          
      Raises:
          ExceptionType: When and why this exception is raised
      """
  ```
- **Exceptions**: Tests are exempt from docstring requirements

## Testing Standards
- Tests must use pytest fixtures and markers
- Random test order is enforced to catch interdependencies
- Coverage minimum: 80% (branch coverage enabled)
- Environment variables for tests: `APP_ENVIRONMENT=local`

## Error Handling & Control Flow
- **Early Returns**: Handle errors at the beginning of functions with early returns
- **Guard Clauses**: Use guard clauses to handle preconditions and invalid states early
- **Happy Path Last**: Place the main function logic (happy path) last for improved readability
- **Avoid Nesting**: Minimize deeply nested if statements
- **Concise Conditionals**: Use one-line syntax for simple conditional statements
- **Avoid Else**: Use if-return pattern instead of if-else when possible
- **Error Types**: Use custom error types or error factories for consistent error handling

## FastAPI Best Practices
- **Function Types**: Use `def` for pure functions and `async def` for I/O-bound operations
- **Functional Components**: Use plain functions and Pydantic models for routes
- **Declarative Routes**: Define routes with clear return type annotations
- **Lifespan Management**: Prefer lifespan context managers over `@app.on_event` decorators
- **Middleware**: Use middleware for logging, error monitoring, and performance optimization
- **Dependency Injection**: Rely on FastAPI's dependency injection for managing state and resources
- **HTTP Exceptions**: Use `HTTPException` for expected errors with appropriate status codes
- **Response Models**: Always define response models for API endpoints

## Performance Optimization
- **Async I/O**: Minimize blocking I/O operations; use async for database and API calls
- **Caching**: Implement caching for static and frequently accessed data
- **Lazy Loading**: Use lazy loading techniques for large datasets
- **Serialization**: Optimize data serialization/deserialization with Pydantic
- **Metrics**: Monitor API performance metrics (response time, latency, throughput)

## Git Commit Standards
- Commits must follow gitlint rules
- Pre-commit hooks must pass before commits are accepted