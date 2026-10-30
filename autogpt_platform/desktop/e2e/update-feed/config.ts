const VERSION = /^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$/;
const VARIANT = /^[a-z0-9](?:[a-z0-9-]{0,22}[a-z0-9])?$/;

export function readProofConfig(env: NodeJS.ProcessEnv) {
  function required(name: string) {
    const value = env[name];
    if (!value) throw new Error(`${name} is required for the real-feed proof`);
    return value;
  }
  function pinned(name: string, pattern: RegExp) {
    const value = required(name);
    if (!pattern.test(value)) throw new Error(`${name} has an invalid value`);
    return value;
  }
  if (required("AUTOGPT_E2E_REAL_FEED") !== "1" || required("AUTOGPT_DESKTOP_UPDATES") !== "on") {
    throw new Error("Real-feed proof requires explicit opt-in and updates=on");
  }
  const variant = env.AUTOGPT_DESKTOP_VARIANT || "";
  if (variant && (!VARIANT.test(variant) || /(?:^|-)updater$/.test(variant))) {
    throw new Error("Invalid dedicated feed-test variant");
  }
  if (!variant && env.AUTOGPT_E2E_FEED_DISPOSABLE_NORMAL !== "1") {
    throw new Error("The normal latest feed requires AUTOGPT_E2E_FEED_DISPOSABLE_NORMAL=1 on a disposable machine");
  }
  for (const key of ["AUTOGPT_E2E_DATA_DIR", "AUTOGPT_DESKTOP_DATA_DIR", "AUTOGPT_E2E_INSTALL_DIR", "AUTOGPT_DESKTOP_RUNTIME"]) {
    if (env[key]) throw new Error(`${key} is not allowed: automatic updater relaunch must use the install's real default paths`);
  }
  const kind = required("AUTOGPT_E2E_KIND");
  if (kind === "deb") {
    throw new Error("The deb is notify-only; use AppImage to prove Linux self-update");
  }
  if (!["nsis", "dmg", "appimage"].includes(kind)) throw new Error("Unsupported self-update installer kind");
  const fromVersion = pinned("AUTOGPT_E2E_FEED_FROM_VERSION", VERSION);
  const toVersion = pinned("AUTOGPT_E2E_FEED_TO_VERSION", VERSION);
  const [a, b] = [fromVersion, toVersion].map((version) => version.split(".").map(Number));
  const index = a.findIndex((part, position) => part !== b[position]);
  if (index === -1 || a[index] >= b[index]) throw new Error("B must be a strictly newer stable version than A");
  const tagPrefix = variant ? `desktop-${variant}-v` : "desktop-v";
  return {
    variant, fromVersion, toVersion, tagPrefix,
    installer: required("AUTOGPT_E2E_INSTALLER"),
    fromSource: pinned("AUTOGPT_E2E_FEED_FROM_SOURCE", /^[a-f0-9]{40}$/),
    toSource: pinned("AUTOGPT_E2E_FEED_TO_SOURCE", /^[a-f0-9]{40}$/),
    fromSha256: pinned("AUTOGPT_E2E_FEED_FROM_SHA256", /^[a-f0-9]{64}$/),
    toSha256: pinned("AUTOGPT_E2E_FEED_TO_SHA256", /^[a-f0-9]{64}$/),
    toArtifact: pinned("AUTOGPT_E2E_FEED_TO_ARTIFACT", /^[A-Za-z0-9_.-]+$/),
  };
}

export type ProofConfig = ReturnType<typeof readProofConfig>;
