conda env create -f environment.yml
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

export PYTHONPATH="${PYTHONPATH}:./"
