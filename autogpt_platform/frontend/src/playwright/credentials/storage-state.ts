export function buildCookieConsentStorageState(
  origin: string = "http://localhost:3000",
) {
  return {
    cookies: [],
    origins: [
      {
        origin,
        localStorage: [
          {
            name: "autogpt_cookie_consent",
            value: JSON.stringify({
              hasConsented: true,
              timestamp: Date.now(),
              analytics: true,
              monitoring: true,
            }),
          },
        ],
      },
    ],
  };
}
