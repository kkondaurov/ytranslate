const SERVER_BASE_URL = "http://127.0.0.1:8765";

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || message.type !== "queueTranslation" || !message.url) {
    return false;
  }

  (async () => {
    try {
      const response = await fetch(`${SERVER_BASE_URL}/jobs`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-YTranslate-Client": "chrome-extension"
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
