# Contributing to StockSense-AI

First off, thank you for considering contributing to StockSense-AI! It's people like you that make this project better for everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

---

## Code of Conduct

This project and everyone participating in it is governed by our commitment to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

---

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please open an issue with:

1. **Clear title** describing the problem
2. **Steps to reproduce** the issue
3. **Expected behavior** vs **actual behavior**
4. **Environment details** (OS, Python version, package versions)
5. **Screenshots** if applicable

### Suggesting Features

We welcome feature suggestions! Please open an issue with:

1. **Clear description** of the feature
2. **Use case** - why is this feature needed?
3. **Proposed implementation** (optional)

### Code Contributions

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write/update tests
5. Submit a pull request

---

## Development Setup

### Prerequisites

- Python 3.9+
- pip
- git

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/yourusername/StockSense-AI.git
cd StockSense-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests to verify setup
python tests/test_mvp.py
```

---

## Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our style guidelines

3. **Test your changes**:
   ```bash
   python tests/test_mvp.py
   python compare_models.py
   ```

4. **Commit with clear messages**:
   ```bash
   git commit -m "Add: brief description of feature"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** with:
   - Clear title and description
   - Reference any related issues
   - Screenshots for UI changes

---

## Style Guidelines

### Python Code

- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small

### Example Function

```python
def calculate_forecast(
    data: pd.DataFrame,
    horizon: int = 30,
    model_type: str = 'xgboost'
) -> pd.DataFrame:
    """
    Generate sales forecast using the specified model.
    
    Args:
        data: Historical sales data with 'ds' and 'y' columns
        horizon: Number of days to forecast (default: 30)
        model_type: Model to use ('xgboost', 'arima', 'prophet')
        
    Returns:
        DataFrame with forecast results and confidence intervals
        
    Raises:
        ValueError: If data is empty or model_type is invalid
    """
    # Implementation here
    pass
```

### Commit Messages

Use clear, descriptive commit messages:

- `Add: new feature description`
- `Fix: bug description`
- `Update: what was changed`
- `Refactor: what was refactored`
- `Docs: documentation changes`
- `Test: test additions/changes`

---

## Reporting Bugs

### Bug Report Template

```markdown
## Bug Description
A clear description of what the bug is.

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Windows 11]
- Python: [e.g., 3.11]
- Package versions: [run `pip freeze`]

## Screenshots
If applicable, add screenshots.

## Additional Context
Any other context about the problem.
```

---

## Suggesting Features

### Feature Request Template

```markdown
## Feature Description
A clear description of the feature.

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed Solution
How would you implement this? (optional)

## Alternatives Considered
Other solutions you've considered. (optional)

## Additional Context
Any other context or screenshots.
```

---

## Questions?

If you have questions, feel free to:

1. Open an issue with the `question` label
2. Check existing issues for similar questions

Thank you for contributing to StockSense-AI!
