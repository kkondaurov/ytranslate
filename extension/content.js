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
  const cssText = `
    #${BUTTON_HOST_ID} {
      display: inline-flex;
      margin-right: 8px;
    }

    #${BUTTON_ID} {
      min-width: auto;
    }

    #${BUTTON_ID} .ytranslate-button-icon {
      display: inline-flex;
      width: 24px;
      height: 24px;
      align-items: center;
      justify-content: center;
    }

    #${BUTTON_ID} .ytranslate-button-icon svg {
      display: block;
      width: 24px;
      height: 24px;
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

  let style = document.getElementById(STYLE_ID);
  if (!style) {
    style = document.createElement("style");
    style.id = STYLE_ID;
    document.documentElement.appendChild(style);
  }
  style.textContent = cssText;
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
    <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 24 24" width="24" focusable="false" aria-hidden="true" fill="none" style="pointer-events: none; display: inherit; width: 100%; height: 100%;">
      <path d="M7 3.5h7.2L18.5 7.8v10.7A1.5 1.5 0 0 1 17 20H7a1.5 1.5 0 0 1-1.5-1.5V5A1.5 1.5 0 0 1 7 3.5Z" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"></path>
      <path d="M14.2 3.5V7.8h4.3" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"></path>
      <circle cx="10.2" cy="12" r=".85" fill="currentColor"></circle>
      <circle cx="13.8" cy="12" r=".85" fill="currentColor"></circle>
      <path d="M9.7 15c.55.55 1.3.85 2.3.85s1.75-.3 2.3-.85" fill="none" stroke="currentColor" stroke-width="1.55" stroke-linecap="round"></path>
    </svg>
  `;
}

function createPdfButton(container) {
  const button = document.createElement("button");
  button.className =
    "yt-spec-button-shape-next yt-spec-button-shape-next--tonal yt-spec-button-shape-next--mono yt-spec-button-shape-next--size-m yt-spec-button-shape-next--icon-leading yt-spec-button-shape-next--enable-backdrop-filter-experiment";
  button.innerHTML = `
    <div aria-hidden="true" class="yt-spec-button-shape-next__icon ytranslate-button-icon">${buildPdfIconMarkup()}</div>
    <div class="yt-spec-button-shape-next__button-text-content">PDF</div>
    <yt-touch-feedback-shape aria-hidden="true" class="yt-spec-touch-feedback-shape yt-spec-touch-feedback-shape--touch-response">
      <div class="yt-spec-touch-feedback-shape__stroke"></div>
      <div class="yt-spec-touch-feedback-shape__fill"></div>
    </yt-touch-feedback-shape>
  `;

  button.id = BUTTON_ID;
  button.type = "button";
  button.setAttribute("aria-label", "Generate Russian PDF transcript");
  button.removeAttribute("aria-pressed");

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
