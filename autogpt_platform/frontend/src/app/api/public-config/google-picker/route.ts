type PublicGooglePickerConfig = {
  clientId: string | null;
  developerKey: string | null;
  appId: string | null;
};

export const dynamic = "force-dynamic";

export function GET() {
  const config: PublicGooglePickerConfig = {
    clientId:
      process.env.GOOGLE_CLIENT_ID ||
      process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID ||
      null,
    developerKey:
      process.env.GOOGLE_API_KEY ||
      process.env.NEXT_PUBLIC_GOOGLE_API_KEY ||
      null,
    appId:
      process.env.GOOGLE_APP_ID ||
      process.env.NEXT_PUBLIC_GOOGLE_APP_ID ||
      null,
  };

  return Response.json(config, {
    headers: { "Cache-Control": "private, no-store" },
  });
}
