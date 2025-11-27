#!/bin/bash
#
# Initial setup script for seizure-detector VM
# - Install prerequisites (ca-certificates, curl, gnupg)
# - Install Python + venv + pip
# - Configure and install gcsfuse
# - Mount GCS bucket to ~/gcs/inputs
# - Install Emacs
# - Create Python virtual environment (.venv) in current directory
#   and optionally install from requirements.txt
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#

set -e  # Exit immediately on any error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"


echo "=== Step 1: Update package index ==="
sudo apt-get update


echo "=== Step 2: Install base system packages (ca-certificates, curl, gnupg, python3, venv, pip) ==="
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    python3 \
    python3-venv \
    python3-pip


echo "=== Step 3: Configure gcsfuse APT repository (if not already present) ==="
if [ ! -f /etc/apt/sources.list.d/gcsfuse.list ]; then
    echo "deb http://packages.cloud.google.com/apt gcsfuse-$(lsb_release -c -s) main" | \
        sudo tee /etc/apt/sources.list.d/gcsfuse.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
fi


echo "=== Step 4: Update package index again (to include gcsfuse repo) ==="
sudo apt-get update


echo "=== Step 5: Install gcsfuse and Emacs ==="
sudo apt-get install -y gcsfuse emacs


echo "=== Step 6: Create local mount directory for GCS ==="
mkdir -p ~/gcs/inputs


echo "=== Step 7: Mount GCS bucket 'seizure_inputs' to ~/gcs/inputs ==="
gcsfuse --implicit-dirs seizure_inputs ~/gcs/inputs || {
    echo "Warning: gcsfuse mount failed. Please check your GCP permissions or bucket name."
}


echo "=== Step 8: Verify mount contents (if mounted) ==="
ls -al ~/gcs/inputs || echo "Warning: could not list ~/gcs/inputs"


echo "=== Step 9: Create Python virtual environment (.venv) ==="
cd "${PROJECT_DIR}"
if [ -d ".venv" ]; then
    echo ".venv already exists — skipping creation."
else
    python3 -m venv .venv
    echo "Created virtual environment at: ${PROJECT_DIR}/.venv"
fi


echo "=== Step 10: Activate .venv and upgrade pip ==="
source .venv/bin/activate
python -m pip install --upgrade pip


echo "=== Step 11: Install dependencies from requirements.txt (if exists) ==="
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found — skipping pip install."
fi

deactivate


echo "=== Setup complete ==="
echo "Project directory: ${PROJECT_DIR}"
echo "Virtual environment: ${PROJECT_DIR}/.venv"
echo "To start using it:"
echo "  cd ${PROJECT_DIR}"
echo "  source .venv/bin/activate"
