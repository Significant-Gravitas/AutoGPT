import { describe, expect, it } from "vitest";

import { ProviderMetadataSupportedAuthTypesItem as AuthType } from "@/app/api/__generated__/models/providerMetadataSupportedAuthTypesItem";
import { METHOD_ORDER } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/ConnectMethodView/ConnectMethodView";

describe("METHOD_ORDER", () => {
  // The dialog renders `METHOD_ORDER.filter(...)`, so a method missing here is
  // not merely mis-ordered — it disappears. `device_code` was added to
  // METHOD_COPY but not to this list, so a device-auth provider rendered an
  // empty method list with no way to connect.
  it("covers every auth type a provider can advertise", () => {
    const missing = Object.values(AuthType).filter(
      (type) => !METHOD_ORDER.includes(type),
    );

    expect(missing).toEqual([]);
  });

  it("offers OAuth and device auth ahead of the manual methods", () => {
    expect(METHOD_ORDER.indexOf(AuthType.device_code)).toBeLessThan(
      METHOD_ORDER.indexOf(AuthType.api_key),
    );
  });
});
