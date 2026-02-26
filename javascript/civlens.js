// CivLens - helper JS
// Handles tab management, input manipulation, and UI interactions for the CivLens extension.
// Adds support for "Send to Tab" functionality and dynamic tab controls.

(function () {
    /**
     * Maximum number of allowed search tabs.
     * @constant {number}
     */
    const MAX_TABS = 5;

    /**
     * The root element ID for the extension UI.
     * @constant {string}
     */
    const ROOT_ID = "civlens-ext";

    /**
     * Retrieves the root DOM element for the extension.
     * Fallback to document if not found.
     * @returns {HTMLElement|Document}
     */
    function getRoot() {
        return document.getElementById(ROOT_ID) || document;
    }

    /**
     * Gets the string value of an input or textarea by its container ID.
     * @param {string} id - The container element ID.
     * @returns {string} The input value.
     */
    function getInputValueById(id) {
        const root = getRoot();
        const el = root.querySelector(`#${id} input, #${id} textarea`);
        return el ? String(el.value || "") : "";
    }

    /**
     * Sets the value of an input or textarea and dispatches change events.
     * @param {string} id - The container element ID.
     * @param {string} value - The value to set.
     * @returns {boolean} True if successful, false otherwise.
     */
    function setInputValueById(id, value) {
        const root = getRoot();
        const el = root.querySelector(`#${id} input, #${id} textarea`);
        if (!el) return false;
        el.value = value;
        el.dispatchEvent(new Event("input", { bubbles: true }));
        el.dispatchEvent(new Event("change", { bubbles: true }));
        return true;
    }

    /**
     * Simulates a click on an element by ID.
     * @param {string} id - The element ID.
     * @returns {boolean} True if clicked, false if not found.
     */
    function clickById(id) {
        const el = document.getElementById(id);
        if (!el) return false;
        el.click();
        return true;
    }

    /**
     * Determines the currently active tab index based on the 'selected' class.
     * @returns {number} The active tab index (0-based).
     */
    function getActiveTabIndex() {
        const buttons = getSearchTabButtons();
        const idx = buttons.findIndex((btn) => btn.classList.contains("selected"));
        return idx >= 0 ? idx : 0;
    }

    /**
     * Retrieves the tab navigation container.
     * @returns {HTMLElement|null}
     */
    function getTabNav() {
        const root = getRoot();
        return root.querySelector("#civlens-search-tabs .tab-nav");
    }

    /**
     * Retrieves all button elements within the tab navigation.
     * @returns {HTMLButtonElement[]}
     */
    function getTabButtons() {
        const nav = getTabNav();
        if (!nav) return [];
        return Array.from(nav.querySelectorAll("button"));
    }

    /**
     * Checks if an element is visible (display != none, visibility != hidden).
     * @param {HTMLElement} el - The element to check.
     * @returns {boolean}
     */
    function isVisible(el) {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        return style && style.display !== "none" && style.visibility !== "hidden";
    }

    /**
     * Retrieves visible search tab buttons (excluding the 'Add' button).
     * @returns {HTMLButtonElement[]}
     */
    function getVisibleSearchButtons() {
        const buttons = getTabButtons();
        if (!buttons.length) return [];
        return buttons.slice(0, -1).filter(isVisible);
    }

    /**
     * Retrieves all search tab buttons (excluding the 'Add' button), visible or not.
     * @returns {HTMLButtonElement[]}
     */
    function getSearchTabButtons() {
        const buttons = getTabButtons();
        if (!buttons.length) return [];
        return buttons.slice(0, -1);
    }

    /**
     * Counts the number of currently visible search tabs.
     * @returns {number}
     */
    function getTabCount() {
        return getVisibleSearchButtons().length || 1;
    }

    /**
     * Clicks a specific search tab button by index.
     * @param {number} index - The tab index to activate.
     * @returns {boolean} True if clicked.
     */
    function clickSearchTab(index) {
        const buttons = getSearchTabButtons();
        const btn = buttons[index];
        if (!btn || !isVisible(btn)) return false;
        btn.click();
        return true;
    }

    /**
     * Clicks the 'Add Tab' button if enabled.
     * @returns {boolean} True if clicked.
     */
    function clickAddTab() {
        const buttons = getTabButtons();
        if (!buttons.length) return false;
        const addBtn = buttons[buttons.length - 1];
        if (!addBtn || addBtn.classList.contains("civlens-tab-add-disabled")) return false;
        addBtn.click();
        return true;
    }

    /**
     * Updates the 'Add Tab' button state (disabled/enabled) based on the tab count.
     */
    function updateAddTabDisabled() {
        const buttons = getTabButtons();
        if (!buttons.length) return;
        const addBtn = buttons[buttons.length - 1];
        const visibleCount = getVisibleSearchButtons().length;
        if (visibleCount >= MAX_TABS) {
            addBtn.classList.add("civlens-tab-add-disabled");
            addBtn.setAttribute("aria-disabled", "true");
        } else {
            addBtn.classList.remove("civlens-tab-add-disabled");
            addBtn.removeAttribute("aria-disabled");
        }
    }

    /**
     * Injects close buttons ('x') into visible tab headers if missing.
     * Wires up the click event to trigger the hidden Gradio close button.
     */
    function attachTabCloseButtons() {
        const buttons = getTabButtons();
        if (!buttons.length) return;
        const searchButtons = buttons.slice(0, -1);
        for (const btn of searchButtons) {
            if (!isVisible(btn)) continue;
            if (btn.querySelector(".civlens-tab-close-btn")) continue;
            const close = document.createElement("span");
            close.className = "civlens-tab-close-btn";
            close.textContent = "×";
            close.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                const text = btn.textContent || "";
                const match = text.match(/Search\s+(\d+)/);
                if (!match) return;
                const idx = parseInt(match[1], 10) - 1;
                clickById(`civlens-close-btn-${idx}`);
            });
            btn.appendChild(close);
        }
    }

    /**
     * Activates a tab and loads a URL into its input field, then triggers loading.
     * @param {number} tabIndex - The target tab index.
     * @param {string} url - The URL to load.
     */
    function openUrlInTab(tabIndex, url) {
        const activeIdx = getActiveTabIndex();
        if (activeIdx !== tabIndex) {
            clickSearchTab(tabIndex);
        }

        setTimeout(() => {
            setInputValueById(`civlens-url-input-${tabIndex}`, url);
            const root = getRoot();
            const btn = root.querySelector(`#civlens-url-btn-${tabIndex} button, #civlens-url-btn-${tabIndex}`);
            if (btn) btn.click();
        }, 250);
    }

    /**
     * Attaches 'Send to Tab' click listeners to buttons.
     * Handles creating a new tab and loading the selected model URL.
     */
    function attachSendToTabButtons() {
        const root = getRoot();
        const nodes = root.querySelectorAll("[id^='civlens-send-tab-']");
        for (const node of nodes) {
            const btn = node.tagName === "BUTTON" ? node : node.querySelector("button");
            if (!btn) continue;
            if (btn.dataset.civitaiSendAttached === "1") continue;
            btn.dataset.civitaiSendAttached = "1";

            btn.addEventListener("click", () => {
                const m = (node.id || "").match(/^civlens-send-tab-(\d+)$/);
                const srcIdx = m ? parseInt(m[1], 10) : 0;
                const url = getInputValueById(`civlens-selected-url-${srcIdx}`);
                if (!url) return;

                const tabCount = getTabCount();
                if (tabCount < MAX_TABS) {
                    if (!clickAddTab()) return;
                    const newIndex = tabCount;
                    let attempts = 0;
                    // Wait for the new tab to appear in the DOM before switching
                    const waitForTab = () => {
                        const btns = getSearchTabButtons();
                        const btn = btns[newIndex];
                        if (btn && isVisible(btn)) {
                            btn.click();
                            setTimeout(() => openUrlInTab(newIndex, url), 200);
                            return;
                        }
                        attempts += 1;
                        if (attempts < 12) {
                            setTimeout(waitForTab, 150);
                        }
                    };
                    waitForTab();
                } else {
                    window.alert(`Maximum tabs reached (${MAX_TABS}). Close a tab to send the model to a new one.`);
                }
            });
        }
    }

    // Initialize observers and event listeners when the DOM is ready
    document.addEventListener("DOMContentLoaded", function () {
        const root = getRoot();
        if (!root) return;
        const observer = new MutationObserver(() => {
            attachSendToTabButtons();
            attachTabCloseButtons();
            updateAddTabDisabled();
        });
        attachSendToTabButtons();
        attachTabCloseButtons();
        updateAddTabDisabled();
        observer.observe(root, { childList: true, subtree: true });
        
        // Periodic check to ensure state consistency (especially during heavy UI loads)
        setInterval(() => {
            updateAddTabDisabled();
        }, 500);
    });

    /**
     * Publicly exposed function to scroll the selected gallery thumbnail into view.
     * This is called from Python/Gradio when a gallery item is selected.
     */
    window.civlens_scroll_gallery = async () => {
        await new Promise(r => setTimeout(r, 200));
        const galleries = document.querySelectorAll('.gradio-gallery');
        galleries.forEach(gallery => {
            const selected = gallery.querySelector('.thumbnails button.selected, .grid-wrap button.selected');
            if (selected) {
                selected.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
            }
        });
    };
})();
