import { afterEach, describe, expect, test, vi } from "vitest";
import { AppEnv, environment } from "@/services/environment";
import { isTestDataSurfaceEnabled } from "./helpers";

function stub(isLocal: boolean, appEnv: AppEnv) {
  vi.spyOn(environment, "isLocal").mockReturnValue(isLocal);
  vi.spyOn(environment, "getAppEnv").mockReturnValue(appEnv);
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("isTestDataSurfaceEnabled", () => {
  test("enabled only when both behave-as and app-env are local", () => {
    stub(true, AppEnv.LOCAL);
    expect(isTestDataSurfaceEnabled()).toBe(true);
  });

  test("disabled when app-env is not local even if behave-as is", () => {
    // The backend only mounts the router on app_env LOCAL, so this combination
    // would render a page whose endpoint 404s.
    stub(true, AppEnv.DEV);
    expect(isTestDataSurfaceEnabled()).toBe(false);
  });

  test("disabled when behave-as is cloud even if app-env is local", () => {
    stub(false, AppEnv.LOCAL);
    expect(isTestDataSurfaceEnabled()).toBe(false);
  });
});
