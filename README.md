# Text2CAD

A fork of Text2CAD that adds a small REST server. You send it a text prompt and it returns the generated CAD model as a STEP file.

This is a fork of [SadilKhan/Text2CAD](https://github.com/SadilKhan/Text2CAD), the NeurIPS 2024 model by Mohammad Sadil Khan, Sankalp Sinha and colleagues at DFKI. The model, training code and data preparation are theirs and unchanged. The original README is kept as [UPSTREAM_README.md](UPSTREAM_README.md).

Related: [multimodal-cadgpt](https://github.com/Vedant-29/multimodal-cadgpt) (the app that calls this server)

## What this fork changes

- `simple_text2cad.py`: a Flask server that loads the model once and serves `GET /health` and `POST /generate-cad`.
- `Cad_VLM/config/inference_user_input.yaml`: filled in for the server (checkpoint path, default Hugging Face cache).
- `Cad_VLM/test_user_input.py`: extra logging of the prompts and generated sequences.
- `environment.yml`: adds `flask`.

## Requirements

- Linux
- An NVIDIA GPU. The text encoder is moved to CUDA unconditionally, so CPU only does not work. The weights take about 1.5 GB (BERT-large plus a 92 MB checkpoint), so a card with 8 GB should be enough.
- NVIDIA driver that supports CUDA 12.1
- conda (Miniconda is fine)
- A Hugging Face account, to download the checkpoint

## Setup

```sh
git clone https://github.com/Vedant-29/text2cad.git
cd Text2CAD
conda env create --file environment.yml
conda activate text2cad

# Check that PyTorch sees the GPU
python -c "import torch; print(torch.cuda.is_available())"

# Only if that prints False
pip3 uninstall torch torchvision torchaudio -y
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Weights

1. Open the [SadilKhan/Text2CAD dataset](https://huggingface.co/datasets/SadilKhan/Text2CAD) on Hugging Face and accept its terms. Downloads need a token from an account that has accepted them.
2. Create a read token at https://huggingface.co/settings/tokens.
3. Download the checkpoint (about 92 MB) into the repo:

```sh
export HF_TOKEN=<your token>
wget --header="Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/datasets/SadilKhan/Text2CAD/resolve/main/text2cad_v1.0/Text2CAD_1.0.pth"
```

The `bert-large-uncased` text encoder (about 1.3 GB) downloads automatically from Hugging Face the first time the server starts.

## Usage

Start the server:

```sh
export TEXT2CAD_CHECKPOINT="$PWD/Text2CAD_1.0.pth"
python simple_text2cad.py
```

It listens on port 5000 on all interfaces. Check that the model loaded:

```sh
curl http://localhost:5000/health
```

Generate a STEP file:

```sh
curl -X POST http://localhost:5000/generate-cad \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A rectangular prism with a hole in the middle."}' \
  -o model.step
```

A missing or empty `prompt` returns 400. If the model produces a sequence that cannot be turned into a solid, the server returns 500 with a JSON error.

For the original inference scripts, the Gradio demo, training, evaluation and data preparation, see [UPSTREAM_README.md](UPSTREAM_README.md). Those scripts read the paths in their YAML configs, so edit `Cad_VLM/config/*.yaml` first.

## Environment variables

| Variable | Required | What it is for | Default |
|---|---|---|---|
| `TEXT2CAD_CHECKPOINT` | No | Path to `Text2CAD_1.0.pth` | `test.checkpoint_path` in `Cad_VLM/config/inference_user_input.yaml` (`/workspace/Text2CAD/Text2CAD_1.0.pth`) |
| `PORT` | No | Port the server listens on | `5000` |
| `HF_TOKEN` | For the download only | Hugging Face read token used by the `wget` command above. The server does not read it. | None |

## Notes

- This is research code. The server has been run on a Linux GPU pod on RunPod with CUDA 12.1, called from multimodal-cadgpt. It has not been tested on other setups.
- The server has no authentication and binds to `0.0.0.0`. Do not expose it to the internet without something in front of it.
- Requests are handled one at a time. Output quality is that of the upstream v1.0 checkpoint, and some prompts do not produce a valid solid.
- No weights or data are included in this repo.

## Citation

If you use Text2CAD, cite the original paper:

```bibtex
@inproceedings{text2cad,
	author = {Khan, Mohammad Sadil and Sinha, Sankalp and Sheikh, Talha Uddin and Stricker, Didier and Ali, Sk Aziz and Afzal, Muhammad Zeshan},
	booktitle = {Advances in Neural Information Processing Systems},
	editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
	pages = {7552--7579},
	publisher = {Curran Associates, Inc.},
	title = {Text2CAD: Generating Sequential CAD Designs from Beginner-to-Expert Level Text Prompts},
	url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/0e5b96f97c1813bb75f6c28532c2ecc7-Paper-Conference.pdf},
	volume = {37},
	year = {2024},
	bdsk-url-1 = {https://proceedings.neurips.cc/paper_files/paper/2024/file/0e5b96f97c1813bb75f6c28532c2ecc7-Paper-Conference.pdf}}
```

## License

CC BY-NC-SA 4.0, the same as upstream. See [LICENSE](LICENSE).

Text2CAD was developed by DFKI (Deutsches Forschungszentrum für Künstliche Intelligenz). You may use and adapt it for non-commercial purposes only, you must credit DFKI and the paper authors, and anything you build on it must be shared under the same license. The changes in this fork are released under the same license.
