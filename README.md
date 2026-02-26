# CivLens

An extension that lets you browse, filter, preview, and download CivitAI models directly from the UI.

## Installation
1. Ensure Python and Stable Diffusion WebUI are set up.
2. Go to the "Extensions" tab in your WebUI interface.
3. Select **Install from URL**.
4. Paste the URL of the git repository into the provided field and click "Install".
5. Click on "Apply and restart UI".
6. Recommended: get your CivitAI API key from your profile settings on [civitai.com](https://civitai.com/user/account) and save it immediately in the Settings tab.

## Updating
**Important**: Updating the extension will reset your configuration (API key and Favorite Creators).
To avoid re-entering your data:
1. Locate `settings.json` in the extension folder and save a copy *before* updating.
2. Update the extension.
3. Replace the new `settings.json` with your backup copy.
4. Restart the UI to restore your settings.

## Features
- Search and filter by Type, Sort, Period, Base model, Tags, Tag categories, Content rating, and Creator
- Load by direct URL to a model/version
- **Advanced Multi-tab Browsing**: Manage up to 5 concurrent search tabs
- **Send to Tab**: Quickly open a model in a new tab without losing your current search context
- Gallery thumbnails and details (version, base model, tags, description)
- Trigger words display with click-to-copy
- Download models into the correct folders by type (Checkpoints, LoRA, ControlNet, etc.)
- LoRA: also saves the first PNG/JPEG preview image alongside the model
- **Security & Anti-DDoS Protection**:
  - **Smart Rate Limiting**: Global lock across tabs with randomized jitter (0.1-0.6s) to prevent request spikes.
  - **Intelligent Retry Logic**: Automatically handles rate limits (429) and server errors with exponential backoff.
  - **Hardened Networking**: HTTPS-only enforcement, CivitAI domain allowlist (Anti-SSRF), and safe filename generation.
  - **Content Sanitization**: Strips unsafe HTML and scripts from descriptions to prevent XSS.
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

## Limitations
- **Large Creator Catalogs**: When filtering by a creator who has published a large number of models (e.g., thousands), the initial load and search may take longer. This is because the extension fetches the creator's full catalog (up to a maximum of 5,000 models for safety reasons) to perform accurate local filtering and search.
  - **Override at your own risk**: You can increase this limit by editing `scripts/civlens.py`. Search for `if pages >= 50 or len(all_loaded) >= 5000:` and increase the values. Be aware that setting very high limits may cause the WebUI to freeze or crash due to excessive memory usage. To avoid this, I strongly recommend using all available filters — such as Type, Sort by, Period, Base Model, and Tags — to narrow down the results and load the smallest possible number of models.
- **API Rate Limits**: Frequent searches or downloads may trigger CivitAI's API rate limits, causing temporary delays.

## Compatibility
- **Tested environment**:
  - WebUI version: neo
  - Python: 3.13.12
  - PyTorch: 2.10.0+cu130
  - flash: 2.8.3+cu130torch2.10
  - xformers: 0.0.34
  - gradio: 4.40.0
- Other versions may work, but compatibility is not guaranteed.
- If you notice that this extension interferes with other extensions or causes any issues, you can uninstall it by simply deleting the CivLens folder from the Extensions directory and restart the UI.

## Disclaimer
This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

By using this extension, you acknowledge that you are solely responsible for any content you download, view, or distribute. The developer assumes no responsibility for how this tool is used or for any consequences resulting from its use.
