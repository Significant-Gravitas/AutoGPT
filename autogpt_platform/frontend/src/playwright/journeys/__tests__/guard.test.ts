import type { BrowserContext, Page, Response } from "@playwright/test";
import { describe, expect, it } from "vitest";
import {
  watchForUnexpectedErrors,
  watchPageForUnexpectedErrors,
} from "../guard";

const activationURL =
  "http://localhost:3000/api/proxy/api/credits/pro-activation/current";

async function collect(
  status: number,
  url: string,
  method: string,
  body: unknown,
) {
  let listener: ((response: Response) => void) | undefined;
  const context = {
    on(event: string, handler: (response: Response) => void) {
      if (event === "response") listener = handler;
    },
  } as unknown as BrowserContext;
  const guard = watchForUnexpectedErrors(context, []);
  listener?.({
    url: () => url,
    status: () => status,
    request: () => ({ method: () => method }),
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as unknown as Response);
  return guard.collect();
}

describe("journey activation lookup contract", () => {
  it("accepts the exact absence response from the optional current lookup", async () => {
    expect(
      await collect(404, activationURL, "GET", {
        detail: "No activation is available",
      }),
    ).toEqual([]);
  });

  it.each([
    [500, activationURL, "GET", { detail: "No activation is available" }],
    [404, activationURL, "POST", { detail: "No activation is available" }],
    [
      404,
      `${activationURL}/other`,
      "GET",
      { detail: "No activation is available" },
    ],
    [404, activationURL, "GET", { detail: "User not found" }],
    [
      404,
      activationURL,
      "GET",
      { detail: "No activation is available", error: "unexpected" },
    ],
    [404, activationURL, "GET", "invalid body"],
  ] as const)(
    "keeps unexpected response %s %s %s visible",
    async (status, url, method, body) => {
      const errors = await collect(status, url, method, body);
      expect(errors).toHaveLength(1);
      expect(errors[0]).toContain(`${status} ${method} ${url}`);
    },
  );
});

describe("a response the journey fakes itself", () => {
  const schedulesURL = "http://localhost:3000/api/proxy/api/schedules";
  const injected = { name: "x-journey-injected", value: "scheduler-outage" };

  async function collectFaked(headers: Record<string, string>) {
    let listener: ((response: Response) => void) | undefined;
    const context = {
      on(event: string, handler: (response: Response) => void) {
        if (event === "response") listener = handler;
      },
    } as unknown as BrowserContext;
    const guard = watchForUnexpectedErrors(context, [
      { status: 503, method: "GET", url: "/api/schedules", header: injected },
    ]);
    listener?.({
      url: () => schedulesURL,
      status: () => 503,
      headers: () => headers,
      request: () => ({ method: () => "GET" }),
      json: async () => ({}),
      text: async () => "{}",
    } as unknown as Response);
    return guard.collect();
  }

  it("is exempt when it carries the journey's header", async () => {
    expect(
      await collectFaked({
        [injected.name]: injected.value,
        "retry-after": "15",
      }),
    ).toEqual([]);
  });

  it("keeps a real failure on the same URL visible", async () => {
    const errors = await collectFaked({ "retry-after": "15" });
    expect(errors).toHaveLength(1);
    expect(errors[0]).toContain(`503 GET ${schedulesURL}`);
  });
});

describe("the guard on one page", () => {
  type Handler = (payload: never) => void;

  function fakePage() {
    const handlers = new Map<string, Handler>();
    const page = {
      on(event: string, handler: Handler) {
        handlers.set(event, handler);
      },
      url: () => "http://localhost:3000/library",
    } as unknown as Page;
    function emit(event: string, payload: unknown) {
      handlers.get(event)?.(payload as never);
    }
    return { page, emit };
  }

  it("collects the page's failed API call, console.error and page error", async () => {
    const { page, emit } = fakePage();
    const guard = watchPageForUnexpectedErrors(page, []);
    emit("response", {
      url: () => "http://localhost:3000/api/proxy/api/library/agents",
      status: () => 500,
      headers: () => ({}),
      request: () => ({ method: () => "GET" }),
      text: async () => "boom",
    });
    emit("console", {
      type: () => "error",
      text: () => "Something broke",
      location: () => ({ url: "app.js", lineNumber: 3 }),
    });
    emit("pageerror", new Error("Minified React error #418"));

    const problems = await guard.collect();

    expect(problems).toHaveLength(3);
    expect(problems[0]).toContain(
      "500 GET http://localhost:3000/api/proxy/api/library/agents",
    );
    expect(problems[1]).toContain("console.error at app.js:3: Something broke");
    expect(problems[2]).toContain(
      "pageerror on http://localhost:3000/library: Minified React error #418",
    );
  });

  it("collects an error that arrives while it is collecting an earlier one", async () => {
    const { page, emit } = fakePage();
    const guard = watchPageForUnexpectedErrors(page, []);
    emit("pageerror", new Error("Minified React error #418"));

    // collect() is waiting on the first error when the second comes in.
    const collecting = guard.collect(10_000);
    emit("pageerror", new Error("Minified React error #423"));

    expect(await collecting).toEqual([
      expect.stringContaining(
        "pageerror on http://localhost:3000/library: Minified React error #418",
      ),
      expect.stringContaining(
        "pageerror on http://localhost:3000/library: Minified React error #423",
      ),
    ]);
  });

  it("stops collecting at its deadline, and says what it left", async () => {
    const { page, emit } = fakePage();
    const guard = watchPageForUnexpectedErrors(page, []);
    emit("pageerror", new Error("Minified React error #418"));

    expect(await guard.collect(0)).toEqual([
      "1 more error(s) arrived after the guard stopped collecting",
    ]);
  });

  it("leaves out what it would leave out for a whole context", async () => {
    const { page, emit } = fakePage();
    const guard = watchPageForUnexpectedErrors(page, [
      { status: 404, url: "/api/platform-linking/tokens/" },
    ]);
    emit("response", {
      url: () =>
        "http://localhost:3000/api/proxy/api/platform-linking/tokens/x/info",
      status: () => 404,
      headers: () => ({}),
      request: () => ({ method: () => "GET" }),
      text: async () => "{}",
    });
    emit("console", {
      type: () => "error",
      text: () => "Failed to load resource: the server responded with 404",
      location: () => ({ url: "", lineNumber: 0 }),
    });

    expect(await guard.collect()).toEqual([]);
  });
});
