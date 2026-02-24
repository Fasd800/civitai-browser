// Civitai Browser - helper JS
// Adds Enter key support for triggering search
(function () {
    const MAX_TABS = 5;

    function getInputValueById(id) {
        const el = document.querySelector(`#${id} input, #${id} textarea`);
        return el ? String(el.value || "") : "";
    }

    function setInputValueById(id, value) {
        const el = document.querySelector(`#${id} input, #${id} textarea`);
        if (!el) return false;
        el.value = value;
        el.dispatchEvent(new Event("input", { bubbles: true }));
        el.dispatchEvent(new Event("change", { bubbles: true }));
        return true;
    }

    function clickById(id) {
        const el = document.getElementById(id);
        if (!el) return false;
        el.click();
        return true;
    }

    function getActiveTabIndex() {
        const el = document.querySelector(".civitai-tabstrip .civitai-tab.active");
        const raw = el ? el.getAttribute("data-tab-index") : null;
        const idx = raw != null ? parseInt(raw, 10) : 0;
        return Number.isFinite(idx) ? idx : 0;
    }

    function getTabCount() {
        return document.querySelectorAll(".civitai-tabstrip .civitai-tab").length || 1;
    }

    function openUrlInTab(tabIndex, url) {
        const activeIdx = getActiveTabIndex();
        if (activeIdx !== tabIndex) {
            clickById(`civitai-switch-btn-${tabIndex}`);
        }

        setTimeout(() => {
            setInputValueById(`civitai-url-input-${tabIndex}`, url);
            const btn = document.querySelector(`#civitai-url-btn-${tabIndex} button, #civitai-url-btn-${tabIndex}`);
            if (btn) btn.click();
        }, 250);
    }

    function attachSendToTabButtons() {
        const nodes = document.querySelectorAll("[id^='civitai-send-tab-']");
        for (const node of nodes) {
            const btn = node.tagName === "BUTTON" ? node : node.querySelector("button");
            if (!btn) continue;
            if (btn.dataset.civitaiSendAttached === "1") continue;
            btn.dataset.civitaiSendAttached = "1";

            btn.addEventListener("click", () => {
                const m = (node.id || "").match(/^civitai-send-tab-(\d+)$/);
                const srcIdx = m ? parseInt(m[1], 10) : 0;
                const url = getInputValueById(`civitai-selected-url-${srcIdx}`);
                if (!url) return;

                const tabCount = getTabCount();
                if (tabCount < MAX_TABS) {
                    clickById("civitai-add-btn");
                    const newIndex = tabCount;
                    setTimeout(() => openUrlInTab(newIndex, url), 350);
                } else {
                    window.alert(`Maximum tabs reached (${MAX_TABS}). Close a tab to send the model to a new one.`);
                }
            });
        }
    }

    document.addEventListener("DOMContentLoaded", function () {
        const observer = new MutationObserver(() => {
            attachSendToTabButtons();
        });
        attachSendToTabButtons();
        observer.observe(document.body, { childList: true, subtree: true });
    });
})();
