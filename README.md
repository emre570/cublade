# 🗡️ cuBlade — Modular CUDA kernel library

A growing toolbox of GPU blades:
from matmul to softmax, each operator is hand-tuned, lightweight, and production-ready.
Plug in only what you need, or forge your own.

## Installation

uv is recommended for installation, since it's faster than pip.

First, create a virtual enviroment:

`uv venv blade`

Then, activate the enviroment:

`source blade/bin/activate`

Then, you must install PyTorch on your own based on your CUDA version:

`uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`

Then, install dependencies:

`uv sync`

Finally, install the package locally in your enviroment:

`uv pip install -e . --no-build-isolation`

You should be all done!

## Usage

You can find example usage scripts inside examples folder.
