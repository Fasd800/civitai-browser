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
- Multi-tab browsing + “Send to new tab” for a selected model
- Gallery thumbnails and details (version, base model, tags, description)
- Trigger words display with click-to-copy
- Download models into the correct folders by type
- LoRA: also saves the first PNG/JPEG preview image alongside the model
- Hardened networking and downloads (HTTPS-only + CivitAI allowlist, safe filenames)
- Scoped UI/CSS/JS to avoid interfering with other extensions
- Manage Favorite Creators in Settings

## How to Use
- **Load a specific model URL**:
  - Paste a CivitAI model or version URL and click “Load from URL”.
  - The details pane shows specifications; choose a version if available and click “Download model”.
- **Browse, then refine**:
  - Set filters (Type, Sort, Period, Base model, Tags, Tag categories, Content rating, Creator).
  - Click “Load models” (at the bottom of the Filters section) to fetch results.
  - Browse the gallery; use the Version dropdown for different versions.
  - Enter a keyword and click “Refine results” to filter the loaded results locally.
- **Recommendation**:
  - This extension is designed primarily for creator-based browsing. For best results (and the intended workflow), set the Creator filter first, click “Load models”, then use the keyword filter to refine locally.
  - Without a Creator selected, “Load models” loads a single page; use “Next” to load more pages.

## Compatibility
- **Tested environment**:
  - WebUI version: neo
  - Python: 3.13.12
  - PyTorch: 2.10.0+cu130
  - flash: 2.8.3+cu130torch2.10
  - xformers: 0.0.34
  - gradio: 4.40.0
- Other versions may work, but compatibility is not guaranteed.
