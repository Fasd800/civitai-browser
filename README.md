# CivitAI Browser

An extension that lets you browse, filter, preview, and download CivitAI models directly from the UI.

## Installation
- Ensure Python and Stable Diffusion WebUI are set up.
- Place this folder in your WebUI extensions directory (e.g., <webui-root>/extensions/civitai-browser).
- Dependencies: `requests` (installed automatically via `install.py` if missing).
- Recommended: get your CivitAI API key from your profile settings on civitai.com and save it immediately in the Settings tab.

## Features
- Search and filter by Type, Sort, Period, Content Levels, and Creator
- Load by direct URL to a model/version
- Gallery thumbnails and details (version, base model, tags, description)
- Trigger words display with click-to-copy
- Download models into the correct folders by type
- LoRA: also saves the first PNG/JPEG preview image alongside the model
- Manage Favorite Creators in Settings

## How to Use
- Load a specific model URL:
  - Paste a CivitAI model or version URL and click “Load from URL”.
  - The details pane shows specifications; choose a version if available and click “Download model”.
- Browse, then refine:
  - Set filters (Type, Sort, Period, Content Levels, Creator) and click “Load models” to fetch results.
  - Browse the gallery; use the Version dropdown for different versions.
  - Enter a keyword and click “Refine results” to filter the loaded results locally.
- Recommendation:
  - Load all available pages first (use “Next” until no more pages are available) before refining with a keyword. This helps surface older models in the results.

## Compatibility
- Tested environment:
  - WebUI version: neo
  - Python: 3.13.12
  - PyTorch: 2.10.0+cu130
  - flash: 2.8.3+cu130torch2.10
  - xformers: 0.0.34
  - gradio: 4.40.0
- Other versions may work, but compatibility is not guaranteed.
