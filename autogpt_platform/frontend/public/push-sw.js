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

// Chrome's WindowClient.url doesn't update for Next.js client-side navigation
// (history.pushState), so the SW sees stale URLs. Pages postMessage their
// current pathname+search whenever it changes; we cache it per client id and
// use it for suppression decisions instead of trusting client.url.
var _clientUrls = {};

// User-controlled master toggle. Pages postMessage on mount and on change.
// Default true: if the SW restarts and no tab has reported yet, pushes still
// fire — the worst case is one notification before the page mounts and
// re-syncs the flag.
var _notificationsEnabled = true;

self.addEventListener("message", function (event) {
  if (!event.data) return;
  if (event.data.type === "CLIENT_URL") {
    if (!event.source || !event.source.id) return;
    if (typeof event.data.url !== "string") return;
    // Only accept same-origin relative paths. Defends against a hostile
    // (e.g. XSS-injected) page pinning an attacker-controlled URL that
    // would later be passed to clients.openWindow on notification click.
    if (!event.data.url.startsWith("/") || event.data.url.startsWith("//")) {
      return;
    }
    if (event.data.url.length > 2048) return;
    _clientUrls[event.source.id] = event.data.url;
    return;
  }
  if (event.data.type === "NOTIFICATIONS_ENABLED") {
    _notificationsEnabled = event.data.value !== false;
    return;
  }
});

function _effectiveUrl(client) {
  var cached = _clientUrls[client.id];
  if (typeof cached === "string") {
    // Cached value is pathname + search; the URL constructor needs an origin
    // to parse it, so resolve against this SW's origin.
    return new URL(cached, self.location.origin).href;
  }
  return client.url;
}

function isClientViewingTarget(client, targetUrl) {
  if (client.visibilityState !== "visible" || !client.focused) return false;
  try {
    var clientUrl = new URL(client.url);
    var target = new URL(targetUrl, self.location.origin);
    // Pathname-only — any /copilot page covers any session: the sidebar's
    // green-check indicator already surfaces completed sessions, so an OS
    // popup on top of that is noise. The only thing this suppression
    // *prevents* is duplicates while the user is already in the feature.
    return clientUrl.pathname === target.pathname;
  } catch (_) {
    return false;
  }
}

self.addEventListener("push", function (event) {
  if (!event.data) return;
  if (!_notificationsEnabled) return;

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

  event.waitUntil(
    self.clients
      .matchAll({ type: "window", includeUncontrolled: true })
      .then(function (clients) {
        // Drop URL cache entries for clients that no longer exist.
        var liveIds = {};
        clients.forEach(function (c) {
          liveIds[c.id] = true;
        });
        Object.keys(_clientUrls).forEach(function (id) {
          if (!liveIds[id]) delete _clientUrls[id];
        });
        var userIsAlreadyLooking = clients.some(function (c) {
          return isClientViewingTarget(
            {
              visibilityState: c.visibilityState,
              focused: c.focused,
              url: _effectiveUrl(c),
            },
            targetUrl,
          );
        });
        if (userIsAlreadyLooking) return;

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

        return self.registration.showNotification(config.title, options);
      }),
  );
});

self.addEventListener("notificationclick", function (event) {
  event.notification.close();
  var url = (event.notification.data && event.notification.data.url) || "/";

  function openFresh() {
    return self.clients.openWindow ? self.clients.openWindow(url) : undefined;
  }

  event.waitUntil(
    self.clients
      .matchAll({ type: "window", includeUncontrolled: true })
      .then(function (clientList) {
        // Use _effectiveUrl(c), not c.url — c.url is stale for SPA navigation.
        // Only focus a tab whose pathname already matches the target; never
        // navigate an unrelated tab away from what the user was doing.
        // Substring matching (.includes) is wrong here: a fallback target of
        // "/" would match every URL, and "/copilot" would match
        // "/copilot-archive".
        var targetPath;
        try {
          targetPath = new URL(url, self.location.origin).pathname;
        } catch (_) {
          return openFresh();
        }
        for (var i = 0; i < clientList.length; i++) {
          var client = clientList[i];
          if (!("focus" in client)) continue;
          try {
            if (new URL(_effectiveUrl(client)).pathname === targetPath) {
              return client.focus().catch(openFresh);
            }
          } catch (_) {
            // skip clients with unparseable URLs
          }
        }
        return openFresh();
      }),
  );
});

// Decodes a base64url-encoded string (no padding) into a Uint8Array.
// Needed because PushManager.subscribe wants applicationServerKey as bytes
// but the backend serves the VAPID key as a base64url string.
function _base64UrlToUint8(value) {
  var padded = value + "===".slice((value.length + 3) % 4);
  var b64 = padded.replace(/-/g, "+").replace(/_/g, "/");
  var raw = atob(b64);
  var out = new Uint8Array(raw.length);
  for (var i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
  return out;
}

function _resubscribeOptions(oldSubscription) {
  if (oldSubscription && oldSubscription.options.applicationServerKey) {
    return Promise.resolve({
      userVisibleOnly: true,
      applicationServerKey: oldSubscription.options.applicationServerKey,
    });
  }
  // Firefox occasionally fires pushsubscriptionchange with no oldSubscription;
  // without applicationServerKey the new sub can't be VAPID-authenticated and
  // every push from the backend will fail. Pull the public key from the API.
  return fetch("/api/proxy/api/push/vapid-key", { credentials: "include" })
    .then(function (r) {
      return r.ok ? r.json() : null;
    })
    .then(function (data) {
      if (!data || !data.public_key) return { userVisibleOnly: true };
      return {
        userVisibleOnly: true,
        applicationServerKey: _base64UrlToUint8(data.public_key),
      };
    })
    .catch(function () {
      return { userVisibleOnly: true };
    });
}

self.addEventListener("pushsubscriptionchange", function (event) {
  event.waitUntil(
    (event.newSubscription
      ? Promise.resolve(event.newSubscription)
      : _resubscribeOptions(event.oldSubscription).then(function (opts) {
          return self.registration.pushManager.subscribe(opts);
        }))
      .then(function (newSub) {
        if (!newSub) return;
        // toJSON() returns base64url-encoded keys per the Web Push spec, which
        // is what pywebpush on the backend expects. Building the keys via
        // btoa(...) would produce standard base64 (with +/=) and break decoding.
        var json = newSub.toJSON();
        return fetch("/api/proxy/api/push/subscribe", {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            endpoint: json.endpoint,
            keys: {
              p256dh: (json.keys && json.keys.p256dh) || "",
              auth: (json.keys && json.keys.auth) || "",
            },
          }),
        });
      })
      .catch(function () {
        // If re-subscription fails, notify open clients so they can retry.
        // Return the promise so event.waitUntil keeps the SW alive until
        // every postMessage has been delivered.
        return self.clients
          .matchAll({ type: "window" })
          .then(function (clientList) {
            for (var i = 0; i < clientList.length; i++) {
              clientList[i].postMessage({ type: "PUSH_SUBSCRIPTION_CHANGED" });
            }
          });
      }),
  );
});
