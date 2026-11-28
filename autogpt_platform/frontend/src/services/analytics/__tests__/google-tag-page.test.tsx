import { configureCookiebot } from "@/tests/integrations/cookiebot";
import { render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DATA_LAYER_NAME } from "../gtag";
import { SetupAnalytics } from "../index";

// Renders the Google tag as the root layout does on the production host and
// reads the commands its init script queues for gtag.js.

const ORIGIN = "https://platform.agpt.co";
const LINK_VALUE = "one-time-link-value";
const INIT_SCRIPT_ID = "_custom-ga-init";

// next/script runs an inline script once per id per process; this keeps the
// content each render hands it.
const scripts = vi.hoisted(() => new Map<string, string>());
vi.mock("next/script", () => ({
  default: (props: {
    id?: string;
    dangerouslySetInnerHTML?: { __html: string };
  }) => {
    if (props.id && props.dangerouslySetInnerHTML) {
      scripts.set(props.id, props.dangerouslySetInnerHTML.__html);
    }
    return null;
  },
}));

beforeEach(() => {
  configureCookiebot();
  scripts.clear();
});

afterEach(() => {
  delete window.gtag;
  Reflect.deleteProperty(window, DATA_LAYER_NAME);
  Reflect.deleteProperty(document, "referrer");
  setPageURL(`${ORIGIN}/`);
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
});

describe("the Google tag", () => {
  it("gets the stripped page and referrer ahead of its config call", () => {
    setPageURL(`${ORIGIN}/reset-password?token=${LINK_VALUE}&next=%2Fhome`);
    Object.defineProperty(document, "referrer", {
      configurable: true,
      value: `${ORIGIN}/link/${LINK_VALUE}`,
    });

    const commands = loadGoogleTag();

    expect(commands.map(([command]) => command)).toEqual([
      "js",
      "set",
      "config",
    ]);
    expect(commands[1][1]).toEqual({
      page_location: `${ORIGIN}/reset-password?token=[REDACTED]&next=%2Fhome`,
      page_referrer: `${ORIGIN}/link/[REDACTED]`,
    });
  });

  it("is left to read a page without a credential itself", () => {
    setPageURL(`${ORIGIN}/marketplace?utm_source=mail`);

    const commands = loadGoogleTag();

    expect(commands.map(([command]) => command)).toEqual(["js", "config"]);
  });
});

function loadGoogleTag() {
  render(<SetupAnalytics ga={{ gaId: "G-TEST" }} host="platform.agpt.co" />);
  const dataLayer: IArguments[] = [];
  Reflect.set(window, DATA_LAYER_NAME, dataLayer);
  // The script defines the gtag shim and queues its commands, as in a browser.
  new Function(scripts.get(INIT_SCRIPT_ID) ?? "")();
  return dataLayer.map((entry) => Array.from(entry));
}

function setPageURL(url: string) {
  const { happyDOM } = window as unknown as {
    happyDOM?: { setURL: (url: string) => void };
  };
  happyDOM?.setURL(url);
}
