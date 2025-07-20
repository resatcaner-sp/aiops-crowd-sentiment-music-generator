# Python & FastAPI Development Guidelines

## Programming Paradigms
- **Functional Approach**: Prefer functional, declarative programming over object-oriented where possible
- **Modularization**: Break down complex logic into smaller, reusable functions
- **Iteration**: Favor iteration over code duplication
- **RORO Pattern**: Receive an Object, Return an Object for consistent function interfaces

## Naming Conventions
- **Variables**: Use descriptive names with auxiliary verbs (e.g., `is_active`, `has_permission`)
- **Files & Directories**: Use lowercase with underscores (e.g., `routers/user_routes.py`)
- **Exports**: Favor named exports for routes and utility functions

## Function Design
- **Pure Functions**: Use `def` for pure functions without side effects
- **Async Operations**: Use `async def` for I/O-bound operations (database, external APIs)
- **Type Hints**: Always include type annotations for function parameters and return values
- **Validation**: Prefer Pydantic models over raw dictionaries for input validation

## Error Handling & Control Flow
- **Early Returns**: Handle errors at the beginning of functions with early returns
- **Guard Clauses**: Use guard clauses to handle preconditions and invalid states early
- **Happy Path Last**: Place the main function logic (happy path) last for improved readability
- **Avoid Nesting**: Minimize deeply nested if statements
- **Concise Conditionals**: Use one-line syntax for simple conditional statements
- **Avoid Else**: Use if-return pattern instead of if-else when possible
- **Error Types**: Use custom error types or error factories for consistent error handling

## FastAPI Best Practices
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

## File Structure
```
src/
├── module_name/
│   ├── routers/
│   │   ├── router_a.py  # Exported router, sub-routes
│   │   └── router_b.py
│   ├── models/
│   │   ├── model_a.py   # Pydantic models
│   │   └── model_b.py
│   ├── services/
│   │   └── service_a.py # Business logic
│   ├── utils/
│   │   └── helpers.py   # Utility functions
│   └── main.py          # Application entry point
```

## Code Examples

### Route Definition
```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/items", tags=["items"])

class ItemResponse(BaseModel):
    id: int
    name: str
    description: str | None = None

@router.get("/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int) -> ItemResponse:
    """Get item by ID."""
    # Early return for error case
    if item_id <= 0:
        raise HTTPException(status_code=400, detail="Item ID must be positive")
        
    # Happy path
    return ItemResponse(id=item_id, name="Sample Item")
```

### Error Handling
```python
async def process_data(data: dict) -> dict:
    """Process input data with proper error handling."""
    # Guard clauses
    if not data:
        raise HTTPException(status_code=400, detail="Empty data provided")
        
    if "required_field" not in data:
        raise HTTPException(status_code=400, detail="Missing required field")
    
    # Happy path last
    result = await perform_processing(data)
    return result
```