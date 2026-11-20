export function describeNativeAuthError(
  error: unknown,
  environment: Readonly<Record<string, string | undefined>> = process.env,
): string {
  let diagnostic =
    error instanceof Error ? error.stack || error.message : String(error);
  for (const [name, value] of Object.entries(environment)) {
    if (
      value &&
      value.length >= 4 &&
      /SECRET|PASSWORD|TOKEN|KEY|DATABASE_URL|DIRECT_URL/i.test(name)
    ) {
      diagnostic = diagnostic.replaceAll(value, "[redacted]");
    }
  }
  return diagnostic.replace(
    /\b(?:https?|postgres(?:ql)?):\/\/[^\s"'<>]+/gi,
    "[redacted URL]",
  );
}
