export const MOBILE_AUTH_CALLBACK_PATH = "/auth/mobile/callback";

const IOS_BUNDLE_ID = "co.agpt.autogpt";
const ANDROID_PACKAGE = "co.agpt.autogpt";
const ANDROID_DEBUG_PACKAGE = "co.agpt.autogpt.dev";
const APPLE_TEAM_ID = /^[A-Z0-9]{10}$/;
const SHA256_FINGERPRINT = /^[0-9A-F]{2}(?::[0-9A-F]{2}){31}$/;

type Env = Record<string, string | undefined>;

// These files let only our signed apps receive the sign-in callback. Each is a
// 404 until configured, so mobile sign-in fails closed.
export function appleAppSiteAssociationResponse(env: Env = process.env) {
  const teamID = readValue(env, "MOBILE_APPLE_TEAM_ID", APPLE_TEAM_ID);
  if (!teamID) return notFound();
  const appID = `${teamID}.${IOS_BUNDLE_ID}`;
  return Response.json(
    {
      applinks: {
        details: [
          { appIDs: [appID], components: [{ "/": MOBILE_AUTH_CALLBACK_PATH }] },
        ],
      },
      webcredentials: { apps: [appID] },
    },
    { headers: { "Cache-Control": "public, max-age=3600" } },
  );
}

export function androidAssetLinksResponse(env: Env = process.env) {
  const statements = [
    assetLink(ANDROID_PACKAGE, env, "MOBILE_ANDROID_CERT_SHA256"),
    assetLink(ANDROID_DEBUG_PACKAGE, env, "MOBILE_ANDROID_DEBUG_CERT_SHA256"),
  ].filter((statement) => statement !== null);
  if (statements.length === 0) return notFound();
  return Response.json(statements, {
    headers: { "Cache-Control": "public, max-age=3600" },
  });
}

function assetLink(packageName: string, env: Env, name: string) {
  const list = env[name]?.trim();
  if (!list) return null;
  const fingerprints = list
    .split(",")
    .map((value) => value.trim().toUpperCase());
  if (!fingerprints.every((value) => SHA256_FINGERPRINT.test(value))) {
    console.error(`${name} must list SHA-256 certificate fingerprints`);
    return null;
  }
  return {
    relation: ["delegate_permission/common.handle_all_urls"],
    target: {
      namespace: "android_app",
      package_name: packageName,
      sha256_cert_fingerprints: fingerprints,
    },
  };
}

function readValue(env: Env, name: string, pattern: RegExp) {
  const value = env[name]?.trim();
  if (!value) return null;
  if (pattern.test(value)) return value;
  console.error(`${name} has an invalid value`);
  return null;
}

function notFound() {
  return new Response(null, { status: 404 });
}
