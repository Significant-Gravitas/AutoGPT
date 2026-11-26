/** A ChatGPT sign-in: the code to enter, and the page to enter it on. */
export interface DeviceLogin {
  loginID: string;
  userCode: string;
  verificationURL: string;
}

const LOGIN_PATH = "/device-login/";

function decode(value: string) {
  try {
    return decodeURIComponent(value.replace(/\+/g, " "));
  } catch {
    return null;
  }
}

function readFragment(fragment: string) {
  const values: Record<string, string> = {};
  for (const pair of fragment.split("&")) {
    const [key, ...rest] = pair.split("=");
    const value = decode(rest.join("="));
    if (key && value !== null) values[key] = value;
  }
  return values;
}

/**
 * Reads the backend's `build_device_login_url`:
 * `…/codex/device-login/{id}#state=…&verification_url=…&user_code=…`.
 * Null unless it has a login id, a code and an https page to enter it on.
 */
export function parseDeviceLogin(loginURL: string): DeviceLogin | null {
  const hashAt = loginURL.indexOf("#");
  if (hashAt < 0) return null;
  const path = loginURL.slice(0, hashAt).split("?")[0];
  const pathAt = path.indexOf(LOGIN_PATH);
  if (pathAt < 0) return null;
  const rawID = path.slice(pathAt + LOGIN_PATH.length);
  if (!rawID || rawID.includes("/")) return null;
  const loginID = decode(rawID);
  const fragment = readFragment(loginURL.slice(hashAt + 1));
  const userCode = fragment.user_code?.trim();
  const verificationURL = fragment.verification_url?.trim();
  if (!loginID || !userCode || !verificationURL) return null;
  if (!/^https:\/\/[^\s/?#]+/i.test(verificationURL)) return null;
  return { loginID, userCode, verificationURL };
}
