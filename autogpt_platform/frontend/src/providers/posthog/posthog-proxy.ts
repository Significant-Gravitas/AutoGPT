// Browser PostHog traffic goes through our own domain so blanket ad-block
// lists for *.posthog.com don't drop consented visitors' events. The path is
// deliberately neutral: names like /posthog, /analytics or /tracking are
// blocklist targets themselves. Consent is unaffected: posthog-js sends
// nothing until analytics is accepted, wherever it sends to.
export const POSTHOG_PROXY_PATH = "/relay";

const POSTHOG_CLOUD_HOST = /^https:\/\/(us|eu)(?:\.i)?\.posthog\.com\/?$/i;

interface PostHogConfig {
  key?: string;
  host?: string;
}

// Only PostHog Cloud is proxied, in whichever region NEXT_PUBLIC_POSTHOG_HOST
// names. A self-hosted PostHog, or none at all, keeps talking to its own host.
export function getPostHogProxyTarget({ key, host }: PostHogConfig) {
  if (!key) return null;
  const region = host?.trim().match(POSTHOG_CLOUD_HOST)?.[1]?.toLowerCase();
  if (!region) return null;
  return {
    apiHost: `https://${region}.i.posthog.com`,
    assetsHost: `https://${region}-assets.i.posthog.com`,
    uiHost: `https://${region}.posthog.com`,
  };
}

export function getPostHogClientHosts({ key, host }: PostHogConfig) {
  const target = getPostHogProxyTarget({ key, host });
  if (!target) return { api_host: host };
  return { api_host: POSTHOG_PROXY_PATH, ui_host: target.uiHost };
}
