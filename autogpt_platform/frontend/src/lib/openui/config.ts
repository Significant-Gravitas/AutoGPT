export function isOpenUIEnabled() {
  return process.env.NEXT_PUBLIC_OPENUI_EXPERIMENT === "true";
}

export function getOpenUIConfig() {
  const apiKey = process.env.OPENUI_API_KEY;
  const baseURL = process.env.OPENUI_BASE_URL;
  const model = process.env.OPENUI_MODEL;
  return apiKey && baseURL && model ? { apiKey, baseURL, model } : null;
}
