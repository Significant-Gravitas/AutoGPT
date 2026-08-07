const HOSTED_PLATFORM_ORIGIN = "https://platform.agpt.co";
const HOSTED_GA_MEASUREMENT_ID = "G-FH2XK2W4GN";

export type RuntimeControlEnvironment = {
  AUTOGPT_TELEMETRY_ENABLED?: string;
  AUTOGPT_FEEDBACK_ENABLED?: string;
  AUTOGPT_DEVELOPER_UI_ENABLED?: string;
  AUTOGPT_GA_MEASUREMENT_ID?: string;
  NEXT_PUBLIC_GA_MEASUREMENT_ID?: string;
};

function isEnabled(value: string | undefined) {
  return ["1", "true", "yes", "on"].includes(value?.trim().toLowerCase() ?? "");
}

function firstValue(...values: Array<string | undefined>) {
  return values.map((value) => value?.trim()).find(Boolean);
}

export function isHostedPlatformOrigin(origin: string | undefined) {
  if (!origin) return false;
  try {
    const url = new URL(origin);
    return (
      url.origin === HOSTED_PLATFORM_ORIGIN &&
      !url.username &&
      !url.password &&
      url.pathname === "/" &&
      !url.search &&
      !url.hash
    );
  } catch {
    return false;
  }
}

export function resolveRuntimeControls({
  publicOrigin,
  isDev,
  env,
}: {
  publicOrigin: string | undefined;
  isDev: boolean;
  env: RuntimeControlEnvironment;
}) {
  const isHosted = isHostedPlatformOrigin(publicOrigin);
  const telemetryEnabled = isHosted || isEnabled(env.AUTOGPT_TELEMETRY_ENABLED);
  const developerUiEnabled =
    isDev || isEnabled(env.AUTOGPT_DEVELOPER_UI_ENABLED);

  return {
    hostedPlatform: isHosted,
    telemetryEnabled,
    feedbackEnabled: isHosted || isEnabled(env.AUTOGPT_FEEDBACK_ENABLED),
    developerUiEnabled,
    gaMeasurementId: telemetryEnabled
      ? firstValue(
          env.AUTOGPT_GA_MEASUREMENT_ID,
          isHosted ? env.NEXT_PUBLIC_GA_MEASUREMENT_ID : undefined,
        ) || HOSTED_GA_MEASUREMENT_ID
      : undefined,
  };
}
