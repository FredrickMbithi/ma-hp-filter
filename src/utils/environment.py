"""
Environment logging utilities for reproducibility.
"""
import sys
import platform


def log_environment() -> dict:
    """
    Log Python version and key library versions.
    
    Returns:
        dict: Environment information including Python version and all critical dependencies.
    """
    environment_info = {
        'python': sys.version,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'system': platform.system(),
        'machine': platform.machine(),
        'processor': platform.processor(),
    }
    
    # Import libraries with graceful fallback for missing packages
    try:
        import pandas as pd
        environment_info['pandas'] = pd.__version__
    except ImportError:
        environment_info['pandas'] = 'NOT INSTALLED'
    
    try:
        import numpy as np
        environment_info['numpy'] = np.__version__
    except ImportError:
        environment_info['numpy'] = 'NOT INSTALLED'
    
    try:
        import statsmodels
        environment_info['statsmodels'] = statsmodels.__version__
    except ImportError:
        environment_info['statsmodels'] = 'NOT INSTALLED'
    
    try:
        import matplotlib
        environment_info['matplotlib'] = matplotlib.__version__
    except ImportError:
        environment_info['matplotlib'] = 'NOT INSTALLED'
    
    try:
        import scipy
        environment_info['scipy'] = scipy.__version__
    except ImportError:
        environment_info['scipy'] = 'NOT INSTALLED'
    
    try:
        import sklearn
        environment_info['scikit-learn'] = sklearn.__version__
    except ImportError:
        environment_info['scikit-learn'] = 'NOT INSTALLED'
    
    try:
        import yaml
        environment_info['pyyaml'] = yaml.__version__
    except ImportError:
        environment_info['pyyaml'] = 'NOT INSTALLED'
    
    return environment_info


def print_environment():
    """Print environment information in a formatted way."""
    env = log_environment()
    
    print("=" * 60)
    print("ENVIRONMENT INFORMATION")
    print("=" * 60)
    print(f"\nPython:")
    print(f"  Version: {env['python_version']}")
    print(f"  Full:    {env['python']}")
    
    print(f"\nSystem:")
    print(f"  Platform:  {env['platform']}")
    print(f"  System:    {env['system']}")
    print(f"  Machine:   {env['machine']}")
    print(f"  Processor: {env['processor']}")
    
    print(f"\nCritical Dependencies:")
    print(f"  pandas:       {env['pandas']}")
    print(f"  numpy:        {env['numpy']}")
    print(f"  statsmodels:  {env['statsmodels']}")
    print(f"  matplotlib:   {env['matplotlib']}")
    print(f"  scipy:        {env['scipy']}")
    print(f"  scikit-learn: {env['scikit-learn']}")
    print(f"  pyyaml:       {env['pyyaml']}")
    print("=" * 60)


if __name__ == '__main__':
    print_environment()
