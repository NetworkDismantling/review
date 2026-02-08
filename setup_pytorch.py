#!/usr/bin/env python3
"""
Dynamic PyTorch and PyTorch Geometric installer.
Detects CUDA version and installs the appropriate wheels.
"""

import subprocess
import sys
import platform
import re
import argparse
import logging
from pathlib import Path
from typing import Optional


"""
Pytorch installation instructions:

    Linux and Windows support:

    OSX

    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

    Linux and Windows

    # CUDA 11.8
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

    # ROCM 6.4 (Linux only)
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4
    # CUDA 12.6
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
    # CUDA 12.8
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    # CUDA 12.9
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
    # CPU only
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu



Torch geometric installation instructions:
    - OSX
        pip install torch_geometric

        # Optional dependencies:
        pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html


    - Linux and Windows support:
        pip install torch_geometric (UP TO: torch 2.8; cuda: 11.8, 12.1, 12.4, 12.6, 12.8, 12.9)

        # Optional dependencies:
        pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html


"""

# ============================================================================
# PyTorch CUDA Support Matrix
# ============================================================================
# Maps CUDA major.minor to PyTorch wheel suffix
# Based on: https://pytorch.org/get-started/locally/
PYTORCH_CUDA_SUPPORT = {
    # CUDA 11.x
    (11, 8): "cu118",
    
    # CUDA 12.x
    (12, 1): "cu121",
    (12, 4): "cu124",
    (12, 6): "cu126",
    (12, 8): "cu128",
    (12, 9): "cu129",
    
    # CUDA 13.x (future-proofing)
    (13, 0): "cu130",
}

# Fallback mapping for unsupported CUDA versions
PYTORCH_CUDA_FALLBACK = {
    11: "cu118",  # For CUDA 11.x not explicitly listed
    12: "cu129",  # For CUDA 12.x not explicitly listed (use latest 12.x)
    13: "cu130",  # For CUDA 13.x not explicitly listed
}


# ============================================================================
# PyTorch Geometric CUDA Support Matrix
# ============================================================================
# Maps CUDA major.minor to PyG wheel suffix
# Based on: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
PYG_CUDA_SUPPORT = {
    # CUDA 11.x
    (11, 8): "cu118",
    
    # CUDA 12.x
    (12, 1): "cu121",
    (12, 4): "cu124",
    (12, 6): "cu126",
    (12, 8): "cu128",
    (12, 9): "cu129",
}

# Fallback mapping for unsupported CUDA versions
PYG_CUDA_FALLBACK = {
    11: "cu118",  # For CUDA 11.x not explicitly listed
    12: "cu129",  # For CUDA 12.x not explicitly listed (use latest 12.x)
}


# ============================================================================
# PyTorch and PyG Versions
# ============================================================================
# Supported PyTorch versions per CUDA version
# These versions are guaranteed to be supported by PyG (checked against PYG_TORCH_SUPPORT)
PYTORCH_VERSIONS = {
    "cu118": "2.7.1",  # CUDA 11.8
    "cu121": "2.5.1",  # CUDA 12.1
    "cu124": "2.6.0",  # CUDA 12.4
    "cu126": "2.8.0",  # CUDA 12.6
    "cu128": "2.8.0",  # CUDA 12.8
    "cu129": "2.8.0",  # CUDA 12.9
    "cu130": "2.8.0",  # CUDA 13.0
    "cpu": "2.8.0",    # CPU
}

# PyG support matrix: (torch_major, torch_minor) -> list of supported CUDA versions
# Based on: https://data.pyg.org/whl/
PYG_TORCH_SUPPORT = {
    (2, 8): ["cpu", "cu118", "cu121", "cu124", "cu126", "cu128", "cu129"],
    (2, 7): ["cpu", "cu118"],
    (2, 6): ["cpu", "cu118", "cu121", "cu124"],
    (2, 5): ["cpu", "cu118", "cu121"],
    (2, 4): ["cpu", "cu118", "cu121"],
    (2, 3): ["cpu", "cu118", "cu121"],
    (2, 2): ["cpu", "cu118", "cu121"],
    (2, 1): ["cpu", "cu118", "cu121"],
    (2, 0): ["cpu", "cu118", "cu121"],
}

# ============================================================================
# PyTorch version command
# ===========================================================================
# For some reason, on MacOS torch import is failing with an OpenMP error.
#   This could be a conflicg between numpy and torch's OpenMP versions.
#   To work around this, it seems that importing numpy before does the trick.
TORCH_VERSION_CMD = f"python -c \"import numpy; import torch; print(torch.__version__)\"" 

# ============================================================================
# Verification command
# ============================================================================
INSTALLATION_VERIFY_CMD = """
python -c "
import numpy
import torch
import torch_geometric
print(f'PyTorch version installed: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'PyG version installed: {torch_geometric.__version__}')
"
"""


def run_command(cmd: str, check: bool = True, silent: bool = False) -> tuple[str, int]:
    """
    Run shell command and return output.
    
    Args:
        cmd: Shell command to execute
        check: If True, exit on non-zero return code
        silent: If True, don't log debug information
        
    Returns:
        Tuple of (stdout, returncode)
    """
    if not silent:
        logger.debug(f"Executing: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after 300s: {cmd}")
        if check:
            sys.exit(1)
        return "", -1
    
    if check and result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}: {cmd}")
        if result.stderr:
            logger.error(f"stderr: {result.stderr}")
        sys.exit(1)
    
    return result.stdout.strip(), result.returncode


def detect_cuda_version() -> Optional[str]:
    """
    Detect CUDA version from nvidia-smi or nvcc.
    
    Returns:
        CUDA version string (e.g., "11.8") or None if not found
    """
    # Try nvidia-smi first
    logger.debug("Attempting to detect CUDA via nvidia-smi...")
    stdout, returncode = run_command("nvidia-smi", check=False, silent=True)
    if returncode == 0:
        # Extract CUDA version from nvidia-smi output
        match = re.search(r"CUDA Version:\s+(\d+)\.(\d+)", stdout)
        if match:
            major, minor = match.groups()
            cuda_version = f"{major}.{minor}"
            logger.debug(f"Detected CUDA {cuda_version} via nvidia-smi")
            return cuda_version
    
    # Try nvcc
    logger.debug("Attempting to detect CUDA via nvcc...")
    stdout, returncode = run_command("nvcc --version", check=False, silent=True)
    if returncode == 0:
        match = re.search(r"release (\d+)\.(\d+)", stdout)
        if match:
            major, minor = match.groups()
            cuda_version = f"{major}.{minor}"
            logger.debug(f"Detected CUDA {cuda_version} via nvcc")
            return cuda_version
    
    logger.debug("No CUDA installation detected")
    return None


def get_cuda_suffix(cuda_version: Optional[str], support_matrix: dict, fallback_matrix: dict) -> str:
    """
    Get CUDA suffix (e.g., 'cu118', 'cu121') for the given CUDA version.
    
    Args:
        cuda_version: CUDA version string (e.g., "11.8", "12.6") or None for CPU
        support_matrix: Dict mapping (major, minor) tuples to CUDA suffixes
        fallback_matrix: Dict mapping major version to fallback CUDA suffix
        
    Returns:
        CUDA suffix string (e.g., "cu118") or "cpu" if cuda_version is None
    """
    if cuda_version is None:
        return "cpu"
    
    cuda_major, cuda_minor = map(int, cuda_version.split('.'))
    cuda_tuple = (cuda_major, cuda_minor)
    
    # Try exact match first
    if cuda_tuple in support_matrix:
        logger.debug(f"Using CUDA suffix {support_matrix[cuda_tuple]} for CUDA {cuda_version}")
        return support_matrix[cuda_tuple]
    
    # Fall back to major version mapping
    if cuda_major in fallback_matrix:
        fallback_suffix = fallback_matrix[cuda_major]
        logger.warning(f"CUDA {cuda_version} not explicitly supported, using fallback {fallback_suffix} for CUDA {cuda_major}.x")
        return fallback_suffix
    
    # Ultimate fallback to CPU
    logger.warning(f"CUDA {cuda_version} not supported, falling back to CPU version")
    return "cpu"


def get_safe_pytorch_version(cuda_version: Optional[str]) -> str:
    """
    Get a safe PyTorch version that is compatible with PyG.
    
    Args:
        cuda_version: CUDA version string or None for CPU
        
    Returns:
        PyTorch version string that is guaranteed to be in PYG_TORCH_SUPPORT
    """
    cuda_suffix = get_cuda_suffix(cuda_version, PYTORCH_CUDA_SUPPORT, PYTORCH_CUDA_FALLBACK)
    
    # Get the recommended version for this CUDA suffix
    if cuda_suffix in PYTORCH_VERSIONS:
        recommended_version = PYTORCH_VERSIONS[cuda_suffix]
        # Parse version
        parts = recommended_version.split('.')
        torch_major_minor = (int(parts[0]), int(parts[1]))
        
        # Check if this version is supported by PyG
        if torch_major_minor in PYG_TORCH_SUPPORT:
            logger.debug(f"Using recommended PyTorch {recommended_version} for {cuda_suffix}")
            return recommended_version
        else:
            # Find the highest PyG-supported version
            logger.warning(f"Recommended PyTorch {recommended_version} not supported by PyG")
    
    # Fallback: use the highest version supported by PyG
    supported_versions = sorted(PYG_TORCH_SUPPORT.keys(), reverse=True)
    if supported_versions:
        torch_major_minor = supported_versions[0]
        # Construct version string (use .0 as patch version)
        safe_version = f"{torch_major_minor[0]}.{torch_major_minor[1]}.0"
        logger.info(f"Using safe PyTorch version {safe_version} (highest PyG-compatible)")
        return safe_version
    
    # Ultimate fallback
    # TODO Hardcoded fallback, I don't like it. Plus, it should be updated if PyG drops support for 2.8
    logger.warning("No safe PyTorch version found, using 2.8.0")
    return "2.8.0"


def get_pytorch_index_url(cuda_version: Optional[str]) -> str:
    """
    Get the PyTorch index URL for the given CUDA version.
    
    Args:
        cuda_version: CUDA version string (e.g., "11.8", "12.6") or None for CPU
        
    Returns:
        PyTorch wheel index URL
    """
    cuda_suffix = get_cuda_suffix(cuda_version, PYTORCH_CUDA_SUPPORT, PYTORCH_CUDA_FALLBACK)
    return f"https://download.pytorch.org/whl/{cuda_suffix}"


def get_pyg_index_url(cuda_version: Optional[str], torch_version: str) -> Optional[str]:
    """
    Get the PyG index URL for the given CUDA and PyTorch versions.
    
    Args:
        cuda_version: CUDA version string (e.g., "11.8", "12.6") or None for CPU
        torch_version: PyTorch version string (e.g., "2.8.0")
        
    Returns:
        PyG wheel index URL or None if extensions not available
    """
    # Extract torch major.minor
    torch_parts = torch_version.split('.')
    torch_major_minor = (int(torch_parts[0]), int(torch_parts[1]))
    
    # Check if this PyTorch version is supported by PyG
    if torch_major_minor not in PYG_TORCH_SUPPORT:
        logger.warning(f"PyTorch {torch_version} not explicitly supported by PyG")
        # Try to use the closest equal or lower supported version
        supported_versions = sorted(PYG_TORCH_SUPPORT.keys(), reverse=True)
        closest_version = None
        for supported in supported_versions:
            if supported <= torch_major_minor:
                closest_version = supported
                break
        
        if closest_version:
            torch_major_minor = closest_version
            logger.info(f"Using PyG wheels for PyTorch {torch_major_minor[0]}.{torch_major_minor[1]} (closest match)")
        else:
            # If no lower version exists, use the highest available (experimental)
            if supported_versions:
                torch_major_minor = max(supported_versions)
                logger.warning(f"No lower PyG version found, trying PyTorch {torch_major_minor[0]}.{torch_major_minor[1]} wheels (experimental)")
            else:
                logger.error("No compatible PyG version found")
                return None
    
    # Get CUDA suffix
    cuda_suffix = get_cuda_suffix(cuda_version, PYG_CUDA_SUPPORT, PYG_CUDA_FALLBACK)
    
    # Check if this CUDA version is supported for this PyTorch version
    supported_cuda = PYG_TORCH_SUPPORT[torch_major_minor]
    if cuda_suffix not in supported_cuda:
        logger.warning(f"CUDA suffix {cuda_suffix} not supported for PyTorch {torch_major_minor[0]}.{torch_major_minor[1]}")
        logger.info(f"Supported CUDA versions: {', '.join(supported_cuda)}")
        # Use CPU if available, otherwise skip extensions
        if "cpu" in supported_cuda:
            logger.info("Falling back to CPU version for PyG extensions")
            cuda_suffix = "cpu"
        else:
            logger.warning("No compatible PyG extensions available")
            return None
    
    # Build URL
    torch_version_str = f"{torch_major_minor[0]}.{torch_major_minor[1]}.0"
    return f"https://data.pyg.org/whl/torch-{torch_version_str}+{cuda_suffix}.html"


def uninstall_torch(dry_run: bool = False) -> None:
    """
    Uninstall PyTorch and related packages.
    
    Args:
        dry_run: If True, show what would be done without executing
    """
    packages = ["torch", "torchvision", "torchaudio"]
    
    for pkg in packages:
        if dry_run:
            logger.info(f"[DRY-RUN] Would uninstall {pkg}")
        else:
            logger.info(f"Uninstalling {pkg}...")
            try:
                run_command(f"pip uninstall -y {pkg}", check=False, silent=True)
                logger.info(f"  ✓ {pkg} uninstalled")
            except Exception as e:
                logger.warning(f"  ⚠ Failed to uninstall {pkg}: {e}")


def uninstall_pyg(dry_run: bool = False) -> None:
    """
    Uninstall PyTorch Geometric and extensions.
    
    Args:
        dry_run: If True, show what would be done without executing
    """
    packages = [
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
        "pyg-lib",
    ]
    
    for pkg in packages:
        if dry_run:
            logger.info(f"[DRY-RUN] Would uninstall {pkg}")
        else:
            logger.debug(f"Attempting to uninstall {pkg}...")
            try:
                run_command(f"pip uninstall -y {pkg}", check=False, silent=True)
                logger.debug(f"  ✓ {pkg} uninstalled")
            except Exception as e:
                logger.debug(f"  {pkg} not installed or failed: {e}")


def install_pytorch(cuda_version: Optional[str], dry_run: bool = False) -> None:
    """
    Install PyTorch with the appropriate CUDA support.
    Uses a safe version that is guaranteed to be compatible with PyG.
    
    Args:
        cuda_version: CUDA version string or None for CPU
        dry_run: If True, show what would be done without executing
    """
    index_url = get_pytorch_index_url(cuda_version)
    safe_version = get_safe_pytorch_version(cuda_version)
    
    # Build command with explicit version for safety
    cmd = f"pip install torch=={safe_version} torchvision --index-url {index_url}"
    
    if dry_run:
        logger.info(f"[DRY-RUN] Would install PyTorch {safe_version} from {index_url}")
        logger.debug(f"[DRY-RUN] Command: {cmd}")
        return
    
    logger.info(f"Installing PyTorch {safe_version} from {index_url}...")
    
    try:
        run_command(cmd)
        logger.info(f"✓ PyTorch {safe_version} installed successfully")
    except Exception as e:
        logger.error(f"Failed to install PyTorch: {e}")
        raise


def get_torch_version() -> Optional[str]:
    """
    Get the installed PyTorch version.
    
    Returns:
        PyTorch version string (e.g., "2.8.0") or None if not installed
    """
    logger.debug("Checking for installed PyTorch...")
    stdout, returncode = run_command(
        TORCH_VERSION_CMD,
        check=False, 
        silent=True
    )
    
    if returncode == 0 and stdout:
        # Remove +cu* suffix if present
        version = stdout.split('+')[0]
        logger.debug(f"Found PyTorch version: {version}")
        return version
    
    logger.debug("PyTorch not installed")
    return None


def get_pyg_version() -> Optional[str]:
    """
    Get the installed PyTorch Geometric version.
    
    Returns:
        PyG version string or None if not installed
    """
    logger.debug("Checking for installed PyTorch Geometric...")
    stdout, returncode = run_command(
        "python -c \"import torch_geometric; print(torch_geometric.__version__)\"",
        check=False,
        silent=True
    )
    
    if returncode == 0 and stdout:
        logger.debug(f"Found PyTorch Geometric version: {stdout}")
        return stdout
    
    logger.debug("PyTorch Geometric not installed")
    return None


def check_torch_cuda_compatibility(torch_version: str, cuda_version: Optional[str]) -> bool:
    """
    Check if installed PyTorch version is compatible with detected CUDA.
    
    Args:
        torch_version: Installed PyTorch version
        cuda_version: Detected CUDA version or None for CPU
        
    Returns:
        True if compatible, False otherwise
    """
    if cuda_version is None:
        # CPU installation - any PyTorch version is compatible
        return True
    
    # Get CUDA suffix for the detected CUDA version
    cuda_suffix = get_cuda_suffix(cuda_version, PYTORCH_CUDA_SUPPORT, PYTORCH_CUDA_FALLBACK)
    
    # Check installed PyTorch CUDA version
    stdout, returncode = run_command(
        "python -c \"import torch; print(torch.version.cuda if torch.cuda.is_available() else 'cpu')\"",
        check=False,
        silent=True
    )
    
    if returncode != 0:
        logger.warning("Could not detect PyTorch CUDA version")
        return False
    
    installed_cuda = stdout.strip()
    
    if installed_cuda == "cpu" and cuda_suffix != "cpu":
        logger.warning(f"PyTorch is CPU-only but CUDA {cuda_version} is available")
        return False
    
    if installed_cuda != "cpu" and cuda_suffix == "cpu":
        logger.info("PyTorch has CUDA support but no CUDA detected (compatible)")
        return True
    
    if installed_cuda == cuda_suffix.replace("cu", ""):
        logger.debug(f"PyTorch CUDA version matches: {installed_cuda}")
        return True
    
    # Check if versions are close enough (e.g., cu118 vs 11.8)
    if installed_cuda != "cpu":
        installed_major = installed_cuda.split(".")[0]
        target_major = cuda_version.split(".")[0]
        if installed_major == target_major:
            logger.info(f"PyTorch CUDA {installed_cuda} is compatible with CUDA {cuda_version}")
            return True
    
    logger.warning(f"PyTorch CUDA version mismatch: installed={installed_cuda}, detected={cuda_version}")
    return False


def install_pyg(cuda_version: Optional[str], torch_version: str, dry_run: bool = False) -> None:
    """
    Install PyTorch Geometric and its extensions.
    
    Args:
        cuda_version: CUDA version string or None for CPU
        torch_version: PyTorch version string
        dry_run: If True, show what would be done without executing
    """
    logger.info("Installing PyTorch Geometric...")
    
    if dry_run:
        logger.info(f"[DRY-RUN] Would install PyTorch Geometric for PyTorch {torch_version}")
        logger.debug(f"[DRY-RUN] CUDA version: {cuda_version or 'CPU'}")
    else:
        # Install base PyG
        try:
            run_command("pip install torch-geometric")
            logger.info("✓ Base PyTorch Geometric installed")
        except Exception as e:
            logger.error(f"Failed to install PyTorch Geometric: {e}")
            raise
    
    # Get PyG extensions index URL
    pyg_index_url = get_pyg_index_url(cuda_version, torch_version)
    
    if pyg_index_url:
        if dry_run:
            logger.info(f"[DRY-RUN] Would install PyG extensions from {pyg_index_url}")
        else:
            logger.info(f"Installing PyG optional extensions from {pyg_index_url}...")
        
        extensions = [
            "torch-scatter",
            "torch-sparse",
            "torch-cluster",
            "torch-spline-conv",
        ]
        
        installed_count = 0
        for ext in extensions:
            try:
                cmd = f"pip install {ext} --find-links {pyg_index_url}"
                if dry_run:
                    logger.info(f"  [DRY-RUN] Would install {ext}")
                    logger.debug(f"  [DRY-RUN] Command: {cmd}")
                else:
                    stdout, returncode = run_command(cmd, check=False, silent=True)
                    if returncode == 0:
                        logger.info(f"  ✓ {ext} installed")
                        installed_count += 1
                    else:
                        logger.warning(f"  ⚠ {ext} not available (optional)")
            except Exception as e:
                logger.warning(f"  ⚠ {ext} failed: {e} (optional)")
        
        if not dry_run:
            logger.info(f"✓ PyTorch Geometric installed successfully ({installed_count}/{len(extensions)} optional extensions)")
    else:
        logger.warning("⚠ PyG extensions not available for this configuration")


def main() -> None:
    """Main installation workflow."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Dynamic PyTorch and PyTorch Geometric installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                 # Install PyTorch and PyG with auto-detected CUDA
  %(prog)s --dry-run       # Preview installation without making changes
  %(prog)s --update        # Uninstall existing versions and install fresh
  %(prog)s --verbose       # Show detailed debug information
        """
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be installed without making changes",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Uninstall existing PyTorch/PyG and install fresh versions",
    )
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(dry_run=args.dry_run or args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("PyTorch and PyTorch Geometric Dynamic Installer")
    if args.dry_run:
        logger.info("[DRY-RUN MODE - No changes will be made]")
    if args.update:
        logger.info("[UPDATE MODE - Existing installations will be replaced]")
    logger.info("=" * 70)
    
    # Detect platform
    os_name = platform.system()
    logger.info(f"\nPlatform: {os_name} ({platform.machine()})")
    
    cuda_version = None
    # macOS doesn't support CUDA
    if os_name == "Darwin":
        logger.info("Note: macOS detected - forcing CPU installation")
    elif os_name not in ["Linux", "Windows"]:
        logger.warning(f"Unsupported platform: {os_name}. Attempting CPU installation...")
    else:
        # Detect CUDA
        cuda_version = detect_cuda_version()
        if cuda_version:
            logger.info(f"CUDA detected: {cuda_version}")
        else:
            logger.info("CUDA not detected - will install CPU version")
    
    try:
        # Check if PyTorch is already installed
        logger.info("\n" + "-" * 70)
        torch_version = get_torch_version()
        
        if torch_version and args.update:
            logger.info(f"PyTorch {torch_version} found - will be updated")
            if args.dry_run:
                logger.info("[DRY-RUN] Would uninstall existing PyTorch")
            else:
                uninstall_torch(dry_run=False)
            torch_version = None  # Force reinstall
        
        if torch_version:
            logger.info(f"PyTorch {torch_version} is already installed")
            
            # Check compatibility with detected CUDA
            if check_torch_cuda_compatibility(torch_version, cuda_version):
                logger.info("✓ Existing PyTorch installation is compatible")
                skip_pytorch_install = True
            else:
                logger.warning("⚠ Existing PyTorch installation may not be fully compatible")
                if args.dry_run:
                    logger.info("[DRY-RUN] Would recommend reinstalling PyTorch")
                    skip_pytorch_install = True
                else:
                    response = input("Reinstall PyTorch with correct CUDA support? [Y/n]: ").strip().lower()
                    if response not in ['n', 'no']:
                        uninstall_torch(dry_run=False)
                        skip_pytorch_install = False
                    else:
                        skip_pytorch_install = True
        else:
            skip_pytorch_install = False
        
        # Install PyTorch if needed
        if not skip_pytorch_install:
            install_pytorch(cuda_version, dry_run=args.dry_run)
            
            # Get installed PyTorch version after installation
            if not args.dry_run:
                torch_version = get_torch_version()
                if torch_version is None:
                    logger.error("Failed to detect PyTorch version after installation")
                    raise RuntimeError("PyTorch installation may have failed")
                logger.info(f"Installed PyTorch version: {torch_version}")
        
        # Handle dry-run mock version
        if args.dry_run and torch_version is None:
            # Use the safe version that would be installed
            torch_version = get_safe_pytorch_version(cuda_version)
            logger.info(f"[DRY-RUN] Simulated PyTorch version: {torch_version}")
        
        if torch_version is None:
            raise RuntimeError("No PyTorch version available")
        
        # Check if PyG is already installed
        logger.info("\n" + "-" * 70)
        pyg_version = get_pyg_version()
        
        if pyg_version and args.update:
            logger.info(f"PyTorch Geometric {pyg_version} found - will be updated")
            if args.dry_run:
                logger.info("[DRY-RUN] Would uninstall existing PyTorch Geometric")
            else:
                uninstall_pyg(dry_run=False)
            pyg_version = None  # Force reinstall
        
        if pyg_version:
            logger.info(f"PyTorch Geometric {pyg_version} is already installed")
            
            # Parse PyG version to check compatibility
            try:
                torch_parts = torch_version.split('.')
                torch_major_minor = (int(torch_parts[0]), int(torch_parts[1]))
                
                if torch_major_minor in PYG_TORCH_SUPPORT:
                    logger.info("✓ PyG version is compatible with installed PyTorch")
                    skip_pyg_install = True
                else:
                    logger.warning(f"⚠ PyG may not be fully compatible with PyTorch {torch_version}")
                    if args.dry_run:
                        logger.info("[DRY-RUN] Would recommend reinstalling PyG")
                        skip_pyg_install = True
                    else:
                        response = input("Reinstall PyTorch Geometric? [Y/n]: ").strip().lower()
                        if response not in ['n', 'no']:
                            uninstall_pyg(dry_run=False)
                            skip_pyg_install = False
                        else:
                            skip_pyg_install = True
            except (ValueError, IndexError):
                logger.warning("Could not parse version numbers for compatibility check")
                skip_pyg_install = True
        else:
            skip_pyg_install = False
        
        # Install PyG if needed
        if not skip_pyg_install:
            install_pyg(cuda_version, torch_version, dry_run=args.dry_run)
        else:
            logger.info("Skipping PyTorch Geometric installation (already installed)")
        
        # Verify installation
        if not args.dry_run:
            logger.info("\n" + "-" * 70)
            logger.info("Verifying installation...")
            try:
                run_command(INSTALLATION_VERIFY_CMD, silent=True)
            except Exception as e:
                logger.warning(f"Verification failed: {e}")
                logger.warning("Installation completed but verification encountered issues")
        
        logger.info("\n" + "=" * 70)
        if args.dry_run:
            logger.info("[DRY-RUN] Preview complete - no changes were made")
        else:
            logger.info("✓ Installation complete!")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.error("\n\nInstallation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n\nInstallation failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


def setup_logging(dry_run: bool = False) -> None:
    """Configure logging with appropriate format and level."""
    log_format = "%(message)s"
    level = logging.DEBUG if dry_run else logging.INFO
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
        
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    main()
