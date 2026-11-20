import type { BrowserContext, Response } from "@playwright/test";
import { describe, expect, it } from "vitest";
import { watchForUnexpectedErrors } from "../guard";

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
