# CivLens

An extension that lets you browse, filter, preview, and download CivitAI models directly from the UI.

## Installation
1. Ensure Python and Stable Diffusion WebUI are set up.
2. Go to the "Extensions" tab in your WebUI interface.
3. Select **Install from URL**.
4. Paste the URL of the repository into the provided field and click "Install".
5. Click on "Apply and restart UI".
6. Recommended: get your CivitAI API key from your profile settings on [civitai.com](https://civitai.com) and save it immediately in the Settings tab.

## Features
- Search and filter by Type, Sort, Period, Base model, Tags, Tag categories, Content rating, and Creator
- Load by direct URL to a model/version
- **Advanced Multi-tab Browsing**: Manage up to 5 concurrent search tabs
- **Send to Tab**: Quickly open a model in a new tab without losing your current search context
- Gallery thumbnails and details (version, base model, tags, description)
- Trigger words display with click-to-copy
- Download models into the correct folders by type (Checkpoints, LoRA, ControlNet, etc.)
- LoRA: also saves the first PNG/JPEG preview image alongside the model
- Hardened networking and downloads (HTTPS-only + CivitAI allowlist, safe filenames)
- Scoped UI/CSS/JS to avoid interfering with other extensions
- Manage Favorite Creators in Settings

## How to Use
- **Load a specific model URL**:
  - Paste a CivitAI model or version URL and click “Load from URL”.
  - The details pane shows specifications; choose a version if available and click “Download model”.
- **Search and Filter**:
  - Set filters (Type, Sort, Period, Base model, Tags, Tag categories, Content rating, Creator).
  - Enter a keyword in the "Search Query" box if desired.
  - Click “Search” (or press Enter) to fetch results.
  - The Smart Search automatically decides whether to fetch new data from CivitAI or filter the existing results locally (e.g., when changing only local filters like Base Model or Tags).
- **Tab Management**:
  - Click the `+` button to open a new tab (max 5).
  - Use the `x` button on any tab handle to close it.
  - Click "Send to Tab" on a model card to open it in a fresh tab for comparison.
- **Recommendation**:
  - This extension is designed primarily for creator-based browsing. For best results (and the intended workflow), set the Creator filter first and click “Search”.
  - If you enter a keyword with a Creator selected, the extension will fetch all models from that creator and then automatically filter them by your keyword.
  - Without a Creator selected, “Search” loads a single page; use “Next” to load more pages.

## Compatibility
- **Tested environment**:
  - WebUI version: neo
  - Python: 3.13.12
  - PyTorch: 2.10.0+cu130
  - flash: 2.8.3+cu130torch2.10
  - xformers: 0.0.34
  - gradio: 4.40.0
- Other versions may work, but compatibility is not guaranteed.
