import { describe, expect, it } from "vitest";
import { REDACTED, redactURL, redactURLsInText } from "../sentry-url-redaction";

describe("redactURL", () => {
  // backend/util/url_redaction_test.py's cases, which both sides must agree on.
  it.each([
    ["/ws?token=a.b.c", `/ws?token=${REDACTED}`],
    ["/x?token=a&page=2&&flag", `/x?token=${REDACTED}&page=${REDACTED}&&flag`],
    ["/x?token=a=b;c=d", `/x?token=${REDACTED}`],
    ["/x?a=1?token=b", `/x?a=${REDACTED}`],
    ["/x?=bare", `/x?=${REDACTED}`],
    ["/ws?a=1#&token=b", `/ws?a=${REDACTED}&token=${REDACTED}`],
    [
      "http://evil?/api/public/shared/chats/s-1/messages",
      `http://evil?/api/public/shared/chats/${REDACTED}/messages`,
    ],
    [
      "http://x/api/invitations/api/public/shared/chats/s-1/messages",
      `http://x/api/invitations/${REDACTED}/public/shared/chats/${REDACTED}/messages`,
    ],
    [
      "/ws?/api/integrations/a=x/y&/webhooks/t",
      `/ws?/api/integrations/a=${REDACTED}&/webhooks/${REDACTED}`,
    ],
    ["/ws?q=it's&token=a", `/ws?q=${REDACTED}&token=${REDACTED}`],
    ['/ws?q="&token=a', `/ws?q=${REDACTED}&token=${REDACTED}`],
    ["/ws?q=a b&token=c", `/ws?q=${REDACTED}&token=${REDACTED}`],
    [
      "https://h:1/api/email/unsubscribe?token=dXNlcjpzaWc%3D",
      `https://h:1/api/email/unsubscribe?token=${REDACTED}`,
    ],
    [
      "/api/integrations/github/webhooks/w-1/ingress",
      `/api/integrations/github/webhooks/${REDACTED}/ingress`,
    ],
    [
      "/api/integrations/webhooks/w-1/ping",
      `/api/integrations/webhooks/${REDACTED}/ping`,
    ],
    [
      "/api/platform-linking/tokens/t-1/confirm",
      `/api/platform-linking/tokens/${REDACTED}/confirm`,
    ],
    [
      "/api/platform-linking/user-tokens/t-1/confirm",
      `/api/platform-linking/user-tokens/${REDACTED}/confirm`,
    ],
    ["/api/public/shared/s-1", `/api/public/shared/${REDACTED}`],
    [
      "/api/public/shared/s-1/files/f-1/download",
      `/api/public/shared/${REDACTED}/files/f-1/download`,
    ],
    [
      "/api/public/shared/chats/s-1/messages?limit=5",
      `/api/public/shared/chats/${REDACTED}/messages?limit=${REDACTED}`,
    ],
    ["/api/invitations/i-1/accept", `/api/invitations/${REDACTED}/accept`],
    ["/api/invitations/i-1", `/api/invitations/${REDACTED}`],
    ["/api/invitations/pending-1/x", `/api/invitations/${REDACTED}/x`],
    [
      "/backend/api/invitations/i-1/accept",
      `/backend/api/invitations/${REDACTED}/accept`,
    ],
    [`/ws?token=${REDACTED}`, `/ws?token=${REDACTED}`],
  ])("masks %s as the backend does", (url, expected) => {
    expect(redactURL(url)).toBe(expected);
  });

  it.each([
    [
      "https://platform.agpt.co/reset-password?token=r-1",
      `https://platform.agpt.co/reset-password?token=${REDACTED}`,
    ],
    [
      "https://platform.agpt.co/link/t-1?platform=discord",
      `https://platform.agpt.co/link/${REDACTED}?platform=${REDACTED}`,
    ],
    // Next fetches a page's server components from its own address.
    ["/link/t-1?_rsc=1ab2c", `/link/${REDACTED}?_rsc=${REDACTED}`],
    ["/share/s-1", `/share/${REDACTED}`],
    ["/share/chat/s-1", `/share/chat/${REDACTED}`],
    [
      "/api/proxy/api/platform-linking/tokens/t-1/info",
      `/api/proxy/api/platform-linking/tokens/${REDACTED}/info`,
    ],
    [
      "/auth/callback#access_token=a&type=recovery",
      `/auth/callback#access_token=${REDACTED}&type=${REDACTED}`,
    ],
  ])("masks the browser's %s", (url, expected) => {
    expect(redactURL(url)).toBe(expected);
  });

  it.each([
    "https://platform.agpt.co/library/agents/a-1",
    "/marketplace#featured",
    "/api/orgs/o-1/invitations/i-1",
    "/api/invitations/pending",
    "/api/integrations/github/webhooks",
    "/api/library/agents/a-1",
    "/api/chat/sessions/s-1/share/state",
    "/link/slack",
    "/link/slack/installed",
    // Route names, which is how Next names a page's transactions.
    "/link/[token]",
    "/share/[token]",
    "/share/chat/[token]",
  ])("keeps %s", (url) => {
    expect(redactURL(url)).toBe(url);
  });
});

describe("redactURLsInText", () => {
  it.each([
    [
      '1.2.3.4:5 - "GET /api/invitations/i-1/decline?x=1 HTTP/1.1" 401',
      `1.2.3.4:5 - "GET /api/invitations/${REDACTED}/decline?x=${REDACTED} HTTP/1.1" 401`,
    ],
    [
      '"GET /ws?q=it\'s&token=a HTTP/1.1"',
      `"GET /ws?q=${REDACTED}&token=${REDACTED} HTTP/1.1"`,
    ],
    [
      "acting as user u for requesting GET https://h/api/public/shared/s-1?x=1",
      `acting as user u for requesting GET https://h/api/public/shared/${REDACTED}?x=${REDACTED}`,
    ],
    [
      '"GET /ws?a=1#&token=t HTTP/1.1"',
      `"GET /ws?a=${REDACTED}&token=${REDACTED} HTTP/1.1"`,
    ],
    ["GET /link/t-1", `GET /link/${REDACTED}`],
    [
      "GET https://platform.agpt.co/share/chat/s-1",
      `GET https://platform.agpt.co/share/chat/${REDACTED}`,
    ],
  ])("masks %s", (text, expected) => {
    expect(redactURLsInText(text)).toBe(expected);
  });

  it.each(["Did it work? Yes.", "GET /library", "GET /link/[token]"])(
    "keeps %s",
    (text) => {
      expect(redactURLsInText(text)).toBe(text);
    },
  );
});
