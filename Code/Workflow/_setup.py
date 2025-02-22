import subprocess
import sys
import importlib.util
import os

### Install missing packages from requirements.txt
##############################################################

def is_package_installed(package_name):
    """Check if a package is installed."""
    return importlib.util.find_spec(package_name) is not None

def install_and_import_requirements(requirements_file="requirements.txt"):
    """
    Reads the requirements.txt file, installs missing packages only if not installed, and imports them.
    """
    try:
        with open(requirements_file, "r") as file:
            packages = [line.strip() for line in file if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        print(f"Error: {requirements_file} not found.")
        return

    for package in packages:
        package_name = package.split("==")[0] if "==" in package else package
        if not is_package_installed(package_name):
            print(f"Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        try:
            __import__(package_name)
        except ImportError:
            print(f"Warning: Failed to import {package_name} after installation.")

install_and_import_requirements()

### Load environment variables from .env file
##############################################################

from dotenv import load_dotenv
from pathlib import Path

# Get the absolute path to the current script (_setup.py)
script_dir = Path(__file__).resolve().parent

# Get the project root (assuming Notebooks is at the same level as data)
project_root = script_dir.parent.parent.parent

load_dotenv(project_root / ".env")

# imports for the graphs
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from typing import Annotated
import operator

import numpy as np
import pandas as pd

from typing import Annotated
from typing_extensions import TypedDict

from polygon import RESTClient

from IPython.display import Image, display

import matplotlib.pyplot as plt

import json