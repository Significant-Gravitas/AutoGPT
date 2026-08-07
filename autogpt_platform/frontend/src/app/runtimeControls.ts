const HOSTED_PLATFORM_HOSTNAME = "platform.agpt.co";
const HOSTED_GA_MEASUREMENT_ID = "G-FH2XK2W4GN";

export type RuntimeControlEnvironment = {
  AUTOGPT_TELEMETRY_ENABLED?: string;
  AUTOGPT_FEEDBACK_ENABLED?: string;
  AUTOGPT_DEVELOPER_UI_ENABLED?: string;
  AUTOGPT_GA_MEASUREMENT_ID?: string;
  NEXT_PUBLIC_GA_MEASUREMENT_ID?: string;
  NEXT_PUBLIC_REACT_QUERY_DEVTOOL?: string;
};

function isEnabled(value: string | undefined) {
  return ["1", "true", "yes", "on"].includes(value?.trim().toLowerCase() ?? "");
}

function firstValue(...values: Array<string | undefined>) {
  return values.map((value) => value?.trim()).find(Boolean);
}

export function isHostedPlatformHost(host: string) {
  const firstHost = host.split(",", 1)[0]?.trim().toLowerCase() ?? "";
  const hostname = firstHost.startsWith("[")
    ? firstHost.slice(1, firstHost.indexOf("]"))
    : firstHost.replace(/:\d+$/, "");
  return hostname === HOSTED_PLATFORM_HOSTNAME;
}

export function resolveRuntimeControls({
  host,
  isDev,
  env,
}: {
  host: string;
  isDev: boolean;
  env: RuntimeControlEnvironment;
}) {
  const isHosted = isHostedPlatformHost(host);
  const developerUiEnabled =
    isDev || isEnabled(env.AUTOGPT_DEVELOPER_UI_ENABLED);

  return {
    telemetryEnabled: isHosted || isEnabled(env.AUTOGPT_TELEMETRY_ENABLED),
    feedbackEnabled: isHosted || isEnabled(env.AUTOGPT_FEEDBACK_ENABLED),
    developerUiEnabled,
    reactQueryDevtoolsEnabled:
      developerUiEnabled && isEnabled(env.NEXT_PUBLIC_REACT_QUERY_DEVTOOL),
    gaMeasurementId:
      firstValue(
        env.AUTOGPT_GA_MEASUREMENT_ID,
        env.NEXT_PUBLIC_GA_MEASUREMENT_ID,
      ) || (isHosted ? HOSTED_GA_MEASUREMENT_ID : undefined),
  };
}
