export type EnvironmentDrivenGoogleConfig = {
  clientId?: string | undefined;
  developerKey?: string | undefined;
  appId?: string | undefined; // Cloud project number
};

export function readEnvGoogleConfig(): EnvironmentDrivenGoogleConfig {
  return {
    clientId: process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID,
    developerKey: process.env.NEXT_PUBLIC_GOOGLE_API_KEY,
    appId: process.env.NEXT_PUBLIC_GOOGLE_APP_ID,
  };
}
