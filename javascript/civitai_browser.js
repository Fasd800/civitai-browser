// Civitai Browser - helper JS
// Adds Enter key support for triggering search
document.addEventListener("DOMContentLoaded", function () {
    const observer = new MutationObserver(() => {
        const queryBox = document.querySelector("#civitai-query input");
        const searchBtn = document.querySelector("#civitai-search-btn");
        if (queryBox && searchBtn && !queryBox._civitaiListenerAdded) {
            queryBox.addEventListener("keydown", function (e) {
                if (e.key === "Enter") {
                    searchBtn.click();
                }
            });
            queryBox._civitaiListenerAdded = true;
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
});
