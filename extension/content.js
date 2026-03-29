const BUTTON_HOST_ID = "ytranslate-pdf-button-host";
const BUTTON_ID = "ytranslate-pdf-button";
const STYLE_ID = "ytranslate-extension-style";
const TOAST_ROOT_ID = "ytranslate-toast-root";
const TOAST_ID = "ytranslate-toast";

let toastQueue = [];
let toastVisible = false;
let hideToastTimer = null;
let processToastTimer = null;
let injectionScheduled = false;
let lastQueuedAt = 0;

function getVideoIdFromUrl(rawUrl) {
  let url;

  try {
    url = new URL(rawUrl);
  } catch (_error) {
    return null;
  }

  if (url.hostname === "youtu.be") {
    return url.pathname.replace(/^\/+/, "").split("/")[0] || null;
  }

  if (url.pathname === "/watch") {
    return url.searchParams.get("v");
  }

  if (url.pathname.startsWith("/shorts/")) {
    return url.pathname.split("/")[2] || null;
  }

  if (url.pathname.startsWith("/embed/")) {
    return url.pathname.split("/")[2] || null;
  }

  if (url.pathname.startsWith("/live/")) {
    return url.pathname.split("/")[2] || null;
  }

  return null;
}

function getCanonicalShareUrl() {
  const videoId = getVideoIdFromUrl(window.location.href);
  if (!videoId) {
    return null;
  }
  return `https://youtu.be/${videoId}`;
}

function isWatchPage() {
  const url = new URL(window.location.href);
  return url.hostname.includes("youtube.com") && url.pathname === "/watch" && !!getVideoIdFromUrl(url.href);
}

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) {
    return;
  }

  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    #${BUTTON_HOST_ID} {
      display: inline-flex;
      margin-right: 8px;
    }

    #${BUTTON_ID} {
      min-width: auto;
    }

    #${BUTTON_ID} .yt-spec-button-shape-next__button-text-content {
      letter-spacing: 0.02em;
    }

    #${TOAST_ROOT_ID} {
      position: fixed;
      left: 24px;
      bottom: 24px;
      z-index: 2147483647;
      pointer-events: none;
    }

    #${TOAST_ID} {
      max-width: min(520px, calc(100vw - 48px));
      padding: 20px 24px;
      border-radius: 16px;
      background: rgba(15, 15, 15, 0.96);
      color: #fff;
      font-size: 18px;
      font-weight: 500;
      line-height: 1.35;
      box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
      opacity: 0;
      transform: translateY(10px);
      transition: opacity 160ms ease, transform 160ms ease;
    }

    #${TOAST_ID}.ytranslate-toast-visible {
      opacity: 1;
      transform: translateY(0);
    }
  `;
  document.documentElement.appendChild(style);
}

function ensureToastRoot() {
  let root = document.getElementById(TOAST_ROOT_ID);
  if (root) {
    return root;
  }

  root = document.createElement("div");
  root.id = TOAST_ROOT_ID;

  const toast = document.createElement("div");
  toast.id = TOAST_ID;
  root.appendChild(toast);

  document.body.appendChild(root);
  return root;
}

function processToastQueue() {
  if (toastVisible || toastQueue.length === 0) {
    return;
  }

  ensureToastRoot();
  const toast = document.getElementById(TOAST_ID);
  if (!toast) {
    return;
  }

  const nextMessage = toastQueue.shift();
  toast.textContent = nextMessage;
  toastVisible = true;
  requestAnimationFrame(() => {
    toast.classList.add("ytranslate-toast-visible");
  });

  if (hideToastTimer) {
    clearTimeout(hideToastTimer);
  }
  hideToastTimer = window.setTimeout(() => {
    toast.classList.remove("ytranslate-toast-visible");
    window.setTimeout(() => {
      toastVisible = false;
      processToastQueue();
    }, 180);
  }, 2800);
}

function showToast(message) {
  toastQueue.push(message);
  if (processToastTimer) {
    clearTimeout(processToastTimer);
  }
  processToastTimer = window.setTimeout(processToastQueue, 0);
}

function getButtonsContainer() {
  return document.querySelector("#top-level-buttons-computed");
}

function buildPdfIconMarkup() {
  return `
    <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 24 24" width="24" focusable="false" aria-hidden="true" style="pointer-events: none; display: inherit; width: 100%; height: 100%;">
      <path d="M6 2h8.5L20 7.5V19a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2Zm7 1.5V8h4.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path>
      <path d="M7.2 16.8v-4.6h1.92c1.19 0 1.9.66 1.9 1.73 0 1.12-.78 1.8-2.02 1.8h-.8v1.05h-1Z" fill="currentColor"></path>
      <path d="M12.1 16.8v-4.6h1.82c1.55 0 2.56.87 2.56 2.28 0 1.42-1.01 2.32-2.56 2.32H12.1Z" fill="currentColor"></path>
      <path d="M17.55 16.8v-4.6h3.05v.85h-2.05v.99h1.86v.83h-1.86v1.93h-1Z" fill="currentColor"></path>
    </svg>
  `;
}

function createPdfButton(container) {
  const shareButton = container.querySelector(
    "yt-button-view-model button, button[aria-label='Share'], button[aria-label^='Share']"
  );

  let button;
  if (shareButton) {
    button = shareButton.cloneNode(true);
  } else {
    button = document.createElement("button");
    button.className = "yt-spec-button-shape-next yt-spec-button-shape-next--tonal yt-spec-button-shape-next--mono yt-spec-button-shape-next--size-m yt-spec-button-shape-next--icon-leading yt-spec-button-shape-next--enable-backdrop-filter-experiment";
    button.innerHTML = `
      <div aria-hidden="true" class="yt-spec-button-shape-next__icon"></div>
      <div class="yt-spec-button-shape-next__button-text-content">PDF</div>
    `;
  }

  button.id = BUTTON_ID;
  button.type = "button";
  button.setAttribute("aria-label", "Generate Russian PDF transcript");
  button.removeAttribute("aria-pressed");

  const icon = button.querySelector(".yt-spec-button-shape-next__icon");
  if (icon) {
    icon.innerHTML = buildPdfIconMarkup();
  }

  const text = button.querySelector(".yt-spec-button-shape-next__button-text-content");
  if (text) {
    text.textContent = "PDF";
  }

  button.addEventListener("click", async (event) => {
    event.preventDefault();
    event.stopPropagation();

    const now = Date.now();
    if (now - lastQueuedAt < 1200) {
      return;
    }
    lastQueuedAt = now;

    const canonicalUrl = getCanonicalShareUrl();
    if (!canonicalUrl) {
      showToast("Could not read this video URL");
      return;
    }

    chrome.runtime.sendMessage(
      {
        type: "queueTranslation",
        url: canonicalUrl
      },
      (response) => {
        if (chrome.runtime.lastError) {
          showToast("Server offline");
          return;
        }

        if (!response || !response.ok) {
          if (response && response.status === 403) {
            showToast("Extension request rejected");
            return;
          }
          if (response && response.status === 400) {
            showToast("Could not queue this video");
            return;
          }
          showToast("Server offline");
          return;
        }

        const payload = response.payload || {};
        if (payload.duplicate) {
          showToast("Already queued");
          return;
        }

        showToast("Sent to ytranslate");
      }
    );
  });

  return button;
}

function injectButton() {
  injectionScheduled = false;

  const existingHost = document.getElementById(BUTTON_HOST_ID);
  if (!isWatchPage()) {
    if (existingHost) {
      existingHost.remove();
    }
    return;
  }

  const container = getButtonsContainer();
  if (!container) {
    return;
  }

  if (existingHost && existingHost.parentElement === container) {
    return;
  }

  if (existingHost) {
    existingHost.remove();
  }

  ensureStyles();
  ensureToastRoot();

  const host = document.createElement("div");
  host.id = BUTTON_HOST_ID;
  host.className = "ytd-menu-renderer";
  host.appendChild(createPdfButton(container));
  container.insertBefore(host, container.firstElementChild);
}

function scheduleInjection() {
  if (injectionScheduled) {
    return;
  }
  injectionScheduled = true;
  window.setTimeout(injectButton, 50);
}

function observePage() {
  const observer = new MutationObserver(() => {
    scheduleInjection();
  });
  observer.observe(document.documentElement, {
    childList: true,
    subtree: true
  });
}

window.addEventListener("yt-navigate-finish", scheduleInjection, true);
window.addEventListener("yt-page-data-updated", scheduleInjection, true);
window.addEventListener("popstate", scheduleInjection, true);

ensureStyles();
ensureToastRoot();
observePage();
scheduleInjection();
