type PublicGooglePickerConfig = {
  clientId: string | null;
  developerKey: string | null;
  appId: string | null;
};

export const dynamic = "force-dynamic";

const GOOGLE_OAUTH_CLIENT_ID =
  /^[0-9A-Za-z._-]+\.apps\.googleusercontent\.com$/;
const GOOGLE_BROWSER_API_KEY = /^AIza[0-9A-Za-z_-]{35}$/;
const GOOGLE_CLOUD_PROJECT_NUMBER = /^\d+$/;

function readBrowserConfig(
  names: string[],
  isValid: (value: string) => boolean,
) {
  for (const name of names) {
    const value = process.env[name]?.trim();
    if (value && isValid(value)) return value;
  }
  return null;
}

export function GET() {
  const config: PublicGooglePickerConfig = {
    clientId: readBrowserConfig(
      ["GOOGLE_PICKER_CLIENT_ID", "NEXT_PUBLIC_GOOGLE_CLIENT_ID"],
      (value) => GOOGLE_OAUTH_CLIENT_ID.test(value),
    ),
    // This route is public by design. Never fall back to GOOGLE_API_KEY: that
    // commonly names a server credential with no browser referrer restriction.
    developerKey: readBrowserConfig(
      ["GOOGLE_PICKER_API_KEY", "NEXT_PUBLIC_GOOGLE_API_KEY"],
      (value) => GOOGLE_BROWSER_API_KEY.test(value),
    ),
    appId: readBrowserConfig(
      ["GOOGLE_PICKER_APP_ID", "NEXT_PUBLIC_GOOGLE_APP_ID"],
      (value) => GOOGLE_CLOUD_PROJECT_NUMBER.test(value),
    ),
  };

  return Response.json(config, {
    headers: { "Cache-Control": "private, no-store" },
  });
}
