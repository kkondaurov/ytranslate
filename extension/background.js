const SERVER_BASE_URL = "http://127.0.0.1:8765";
const PERSONAL_HQ_SERVER_BASE_URL = "http://127.0.0.1:8766";

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || !message.url || !["queueTranslation", "queuePersonalHQ"].includes(message.type)) {
    return false;
  }

  (async () => {
    try {
      const isPersonalHQ = message.type === "queuePersonalHQ";
      const clientHeader = isPersonalHQ ? "X-Personal-HQ-Client" : "X-YTranslate-Client";
      const clientHeaderValue = isPersonalHQ ? "youtube-extension" : "chrome-extension";
      const response = await fetch(`${isPersonalHQ ? PERSONAL_HQ_SERVER_BASE_URL : SERVER_BASE_URL}${isPersonalHQ ? "/youtube/enqueue" : "/jobs"}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          [clientHeader]: clientHeaderValue
        },
        body: JSON.stringify({ url: message.url })
      });

      let payload = {};
      try {
        payload = await response.json();
      } catch (_error) {
        payload = {};
      }

      sendResponse({
        ok: response.ok,
        status: response.status,
        payload
      });
    } catch (error) {
      sendResponse({
        ok: false,
        networkError: true,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  })();

  return true;
});
