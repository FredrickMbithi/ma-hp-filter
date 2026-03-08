# Environment Setup Guide

This document provides exact steps to reproduce the development environment for the MA-HP Filter project.

## Prerequisites

- **Python Version:** 3.11 or higher (recommended: 3.12+)
- **Operating System:** Linux, macOS, or Windows with WSL
- **Git:** For version control
- **pip:** Python package installer (included with Python)

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ma-hp-filter
```

### 2. Create Virtual Environment

**On Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Install Dependencies

All dependencies are pinned to specific versions for reproducibility:

```bash
pip install -r requirements.txt
```

This will install:

- **Data Processing:** pandas (3.0.1), numpy (2.4.2)
- **Statistical Analysis:** statsmodels (0.14.6), scipy (1.17.1)
- **Machine Learning:** scikit-learn (1.8.0)
- **Visualization:** matplotlib (3.10.8), seaborn (0.13.2)
- **Configuration:** PyYAML (6.0.3)
- **Development Tools:** Jupyter, pytest, black, flake8, mypy

### 5. Verify Installation

Check that all dependencies are correctly installed:

```bash
make env-info
```

Or run directly:

```bash
python -m src.utils.environment
```

Expected output will show:

- Python version
- System information
- All critical library versions

## Detailed Setup Steps

### Python Version Management

We recommend using **pyenv** (Linux/macOS) or **pyenv-win** (Windows) to manage Python versions:

**Install pyenv (Linux/macOS):**

```bash
curl https://pyenv.run | bash
```

**Install specific Python version:**

```bash
pyenv install 3.12.0
pyenv local 3.12.0
```

### Virtual Environment Best Practices

1. **Always activate** the virtual environment before working on the project
2. **Never commit** the `venv/` directory (it's in `.gitignore`)
3. **Update requirements.txt** if you add new dependencies:
   ```bash
   pip freeze > requirements.txt
   ```

## Development Workflow

### Initial Setup

```bash
# Clone and set up
git clone <repository-url>
cd ma-hp-filter
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt

# Verify environment
make env-info
```

### Daily Workflow

```bash
# Activate environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run tests
make test

# Format code
make format

# Lint code
make lint

# Run notebooks
make notebook
```

## Makefile Commands

The project includes a Makefile with useful shortcuts:

- `make install` — Install all dependencies
- `make test` — Run tests with coverage
- `make lint` — Check code quality (flake8 + mypy)
- `make format` — Auto-format code (black)
- `make backtest` — Run default backtest
- `make notebook` — Launch Jupyter notebook server
- `make env-info` — Display environment information
- `make clean` — Remove cache and log files

## Troubleshooting

### Issue: Import errors after installation

**Solution:**

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### Issue: Version conflicts

**Solution:**

```bash
# Clean install
deactivate  # if venv is active
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: Jupyter kernel not found

**Solution:**

```bash
# Add the virtual environment as a Jupyter kernel
python -m ipykernel install --user --name=ma-hp-filter --display-name="MA-HP Filter"
```

## Critical Dependencies

The following dependencies are critical for the project:

| Package      | Version | Purpose                       |
| ------------ | ------- | ----------------------------- |
| pandas       | 3.0.1   | Time series data manipulation |
| numpy        | 2.4.2   | Numerical computations        |
| statsmodels  | 0.14.6  | HP filter implementation      |
| matplotlib   | 3.10.8  | Plotting and visualization    |
| scipy        | 1.17.1  | Scientific computing          |
| scikit-learn | 1.8.0   | Machine learning utilities    |
| PyYAML       | 6.0.3   | Configuration file parsing    |

## Environment Validation

To ensure your environment matches the expected configuration:

1. **Check Python version:**

   ```bash
   python --version
   ```

   Expected: Python 3.11+ (3.12+ recommended)

2. **Check installed packages:**

   ```bash
   pip list
   ```

   Compare with `requirements.txt`

3. **Run environment logger:**

   ```bash
   python -m src.utils.environment
   ```

   Verify all versions match

4. **Run tests:**
   ```bash
   make test
   ```
   All tests should pass

## Docker Alternative (Optional)

For maximum reproducibility, consider using Docker:

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "-m", "src.backtest.engine"]
```

Build and run:

```bash
docker build -t ma-hp-filter .
docker run -v $(pwd)/data:/app/data ma-hp-filter
```

## Additional Resources

- **Python Virtual Environments:** https://docs.python.org/3/tutorial/venv.html
- **pip Documentation:** https://pip.pypa.io/
- **pyenv:** https://github.com/pyenv/pyenv
- **pandas Documentation:** https://pandas.pydata.org/docs/
- **statsmodels HP Filter:** https://www.statsmodels.org/stable/generated/statsmodels.tsa.filters.hp_filter.hpfilter.html

## Support

If you encounter issues not covered in this guide:

1. Check that you're using Python 3.11+
2. Verify all dependencies are installed: `pip list`
3. Run environment diagnostics: `make env-info`
4. Check the project's issue tracker or contact the maintainer

---

**Last Updated:** March 8, 2026  
**Python Version:** 3.12+  
**Maintainer:** Project Team
