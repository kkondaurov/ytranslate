const BUTTON_HOST_ID = "ytranslate-pdf-button-host";
const BUTTON_ID = "ytranslate-pdf-button";
const BUTTON_RENDER_VERSION = "5";
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
      flex: 0 0 auto;
      margin-right: 8px;
    }

    #${BUTTON_ID} {
      min-width: auto;
    }

    #${BUTTON_ID} .ytranslate-button-icon,
    #${BUTTON_ID} .ytSpecButtonShapeNextIcon,
    #${BUTTON_ID} .yt-spec-button-shape-next__icon {
      display: inline-flex;
      width: 24px;
      height: 24px;
      align-items: center;
      justify-content: center;
      flex: 0 0 auto;
    }

    #${BUTTON_ID} .ytranslate-button-icon svg {
      display: block;
      width: 100%;
      height: 100%;
    }

    #${BUTTON_ID} .ytSpecButtonShapeNextButtonTextContent,
    #${BUTTON_ID} .yt-spec-button-shape-next__button-text-content {
      letter-spacing: 0;
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
    <span class="ytIconWrapperHost ytranslate-button-icon" style="width: 24px; height: 24px;">
      <span class="yt-icon-shape ytSpecIconShapeHost">
        <div style="width: 100%; height: 100%; display: block; fill: currentcolor;">
          <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 24 24" width="24" focusable="false" aria-hidden="true" style="pointer-events: none; display: inherit; width: 100%; height: 100%;">
            <path fill-rule="evenodd" clip-rule="evenodd" d="M5 2h9.5L20 7.5V20a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2Zm2 2v16h10a1 1 0 0 0 1-1V9h-5V4H7Zm8 .5V7h2.5L15 4.5Z"></path>
            <path d="M8 12h8v2H8v-2Zm0 4h6v2H8v-2Z"></path>
          </svg>
        </div>
      </span>
    </span>
  `;
}

function findButtonIcon(button) {
  return button.querySelector(".ytSpecButtonShapeNextIcon, .yt-spec-button-shape-next__icon");
}

function findButtonText(button) {
  return button.querySelector(
    ".ytSpecButtonShapeNextButtonTextContent, .yt-spec-button-shape-next__button-text-content"
  );
}

function createFallbackPdfButton() {
  const button = document.createElement("button");
  button.className =
    "ytSpecButtonShapeNextHost ytSpecButtonShapeNextTonal ytSpecButtonShapeNextMono ytSpecButtonShapeNextSizeM ytSpecButtonShapeNextIconLeading ytSpecButtonShapeNextEnableBackdropFilterExperiment";
  button.innerHTML = `
    <div aria-hidden="true" class="ytSpecButtonShapeNextIcon ytranslate-button-icon">${buildPdfIconMarkup()}</div>
    <div class="ytSpecButtonShapeNextButtonTextContent">PDF</div>
    <yt-touch-feedback-shape aria-hidden="true" class="ytSpecTouchFeedbackShapeHost ytSpecTouchFeedbackShapeTouchResponse">
      <div class="ytSpecTouchFeedbackShapeStroke"></div>
      <div class="ytSpecTouchFeedbackShapeFill"></div>
    </yt-touch-feedback-shape>
  `;
  return button;
}

function createPdfButton(container) {
  const templateButton = container.querySelector(
    "yt-button-view-model button[aria-label='Share'], yt-button-view-model button[aria-label^='Share'], button[aria-label='Share'], button[aria-label^='Share']"
  );
  let button = templateButton ? templateButton.cloneNode(true) : null;
  if (!button || !findButtonIcon(button) || !findButtonText(button)) {
    button = createFallbackPdfButton();
  }

  button.id = BUTTON_ID;
  button.type = "button";
  button.dataset.renderVersion = BUTTON_RENDER_VERSION;
  button.title = "";
  button.setAttribute("aria-label", "Generate Russian PDF transcript");
  button.removeAttribute("aria-pressed");

  const icon = findButtonIcon(button);
  if (icon) {
    icon.classList.add("ytranslate-button-icon");
    icon.innerHTML = buildPdfIconMarkup();
  }

  const text = findButtonText(button);
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

  const existingButton = existingHost ? existingHost.querySelector(`#${BUTTON_ID}`) : null;
  if (
    existingHost &&
    existingHost.parentElement === container &&
    existingButton &&
    existingButton.dataset.renderVersion === BUTTON_RENDER_VERSION
  ) {
    ensureStyles();
    ensureToastRoot();
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
