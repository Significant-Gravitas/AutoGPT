import { AppEnv, environment } from "@/services/environment";

// Mirror the backend exactly: the router is only mounted when app_env is
// LOCAL (rest_api.py), and _guard_local_only additionally requires
// behave_as LOCAL. Checking only one of the two would show a page whose
// endpoint 404s or 403s.
export function isTestDataSurfaceEnabled() {
  return environment.isLocal() && environment.getAppEnv() === AppEnv.LOCAL;
}
