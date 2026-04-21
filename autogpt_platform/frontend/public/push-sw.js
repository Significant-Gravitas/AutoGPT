// push-sw.js — Generic Web Push service worker for AutoGPT Platform.
// Handles push events for ALL notification types.
// Plain JS (not TypeScript) because Next.js serves public/ files as-is.

/** @type {Record<string, Record<string, {title: string, body: string, url: string}>>} */
var NOTIFICATION_MAP = {
  copilot_completion: {
    session_completed: {
      title: "AutoPilot is ready",
      body: "A response is waiting for you.",
      url: "/copilot",
    },
    session_failed: {
      title: "AutoPilot session failed",
      body: "Something went wrong with your session.",
      url: "/copilot",
    },
  },
  onboarding: {
    step_completed: {
      title: "Onboarding progress",
      body: "You completed an onboarding step!",
      url: "/",
    },
  },
};

function getNotificationConfig(data) {
  var typeMap = NOTIFICATION_MAP[data.type];
  if (typeMap) {
    var eventConfig = typeMap[data.event];
    if (eventConfig) return eventConfig;
  }
  return {
    title: "AutoGPT Notification",
    body: data.event || "Something happened.",
    url: "/",
  };
}

self.addEventListener("install", function () {
  self.skipWaiting();
});

self.addEventListener("activate", function (event) {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("push", function (event) {
  if (!event.data) return;

  var data;
  try {
    data = event.data.json();
  } catch (_) {
    return;
  }

  var config = getNotificationConfig(data);

  var targetUrl = config.url;
  if (data.type === "copilot_completion" && data.session_id) {
    targetUrl = "/copilot?sessionId=" + data.session_id;
  }

  // Unique tag per push so repeat notifications aren't coalesced by the OS.
  // Falls back to Date.now() if the backend omitted `id`.
  var tag = data.type + ":" + data.event + ":" + (data.id || Date.now());

  var options = {
    body: config.body,
    icon: "/favicon.ico",
    badge: "/favicon.ico",
    tag: tag,
    data: Object.assign({ url: targetUrl }, data),
    renotify: true,
  };

  event.waitUntil(self.registration.showNotification(config.title, options));
});

self.addEventListener("notificationclick", function (event) {
  event.notification.close();
  var url = (event.notification.data && event.notification.data.url) || "/";

  event.waitUntil(
    self.clients
      .matchAll({ type: "window", includeUncontrolled: true })
      .then(function (clientList) {
        for (var i = 0; i < clientList.length; i++) {
          var client = clientList[i];
          if (client.url.includes(url) && "focus" in client) {
            return client.focus();
          }
        }
        for (var j = 0; j < clientList.length; j++) {
          var c = clientList[j];
          if ("focus" in c && "navigate" in c) {
            return c.focus().then(function (focused) {
              return focused.navigate(url);
            });
          }
        }
        if (self.clients.openWindow) {
          return self.clients.openWindow(url);
        }
      }),
  );
});

self.addEventListener("pushsubscriptionchange", function (event) {
  event.waitUntil(
    Promise.resolve(
      event.newSubscription ||
        self.registration.pushManager.subscribe(
          event.oldSubscription
            ? { userVisibleOnly: true, applicationServerKey: event.oldSubscription.options.applicationServerKey }
            : { userVisibleOnly: true },
        ),
    )
      .then(function (newSub) {
        if (!newSub) return;
        return fetch("/api/proxy/api/push/subscribe", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            endpoint: newSub.endpoint,
            keys: {
              p256dh: btoa(String.fromCharCode.apply(null, new Uint8Array(newSub.getKey("p256dh")))),
              auth: btoa(String.fromCharCode.apply(null, new Uint8Array(newSub.getKey("auth")))),
            },
            user_agent: navigator.userAgent || "",
          }),
        });
      })
      .catch(function () {
        // If re-subscription fails, notify open clients so they can retry
        self.clients.matchAll({ type: "window" }).then(function (clientList) {
          for (var i = 0; i < clientList.length; i++) {
            clientList[i].postMessage({ type: "PUSH_SUBSCRIPTION_CHANGED" });
          }
        });
      }),
  );
});
