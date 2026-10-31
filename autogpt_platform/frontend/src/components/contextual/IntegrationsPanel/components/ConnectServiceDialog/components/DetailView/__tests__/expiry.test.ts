import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook } from "@testing-library/react";
import { createElement, type ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { postV1CreateCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";

import {
  apiKeyConnectSchema,
  dateInputToEndOfLocalDayUnix,
  expiryInputToUnix,
  toDateTimeLocalValue,
} from "../schema";
import { useApiKeyConnectForm } from "../useApiKeyConnectForm";

vi.mock("@/app/api/__generated__/endpoints/integrations/integrations", () => ({
  postV1CreateCredentials: vi.fn(async () => ({
    status: 201,
    data: { id: "cred-1" },
  })),
}));

const originalTZ = process.env.TZ;

function inZone(tz: string) {
  process.env.TZ = tz;
}

function expiryError(expiresAt: string) {
  const result = apiKeyConnectSchema.safeParse({
    title: "K",
    apiKey: "sk",
    expiresAt,
  });
  return result.success ? undefined : result.error.issues[0]?.message;
}

afterEach(() => {
  process.env.TZ = originalTZ;
  vi.useRealTimers();
});

describe("dateInputToEndOfLocalDayUnix", () => {
  it("is 23:59:59 on the picked day in Chicago", () => {
    inZone("America/Chicago");
    // 2026-12-31T23:59:59-06:00
    expect(dateInputToEndOfLocalDayUnix("2026-12-31")).toBe(1798783199);
  });

  it("is 23:59:59 on the picked day in Tokyo", () => {
    inZone("Asia/Tokyo");
    // 2026-12-31T23:59:59+09:00
    expect(dateInputToEndOfLocalDayUnix("2026-12-31")).toBe(1798729199);
  });

  it.each(["", "2026-02-31", "2026-13-01", "not a date"])(
    "returns undefined for %j",
    (value) => {
      expect(dateInputToEndOfLocalDayUnix(value)).toBeUndefined();
    },
  );
});

describe("expiryInputToUnix", () => {
  it("keeps the picked local time of a date-time", () => {
    inZone("America/Chicago");
    // 2026-10-26T20:15:00-05:00
    expect(expiryInputToUnix("2026-10-26T20:15")).toBe(1793063700);
    inZone("Asia/Tokyo");
    // 2026-10-26T20:15:00+09:00
    expect(expiryInputToUnix("2026-10-26T20:15")).toBe(1793013300);
  });

  it("uses the end of the local day for a date", () => {
    inZone("America/Chicago");
    expect(expiryInputToUnix("2026-12-31")).toBe(1798783199);
  });

  it.each(["", "2026-02-31T10:00", "2026-10-26T24:00", "2026-10-26 10:00"])(
    "returns undefined for %j",
    (value) => {
      expect(expiryInputToUnix(value)).toBeUndefined();
    },
  );
});

describe("toDateTimeLocalValue", () => {
  it("formats local wall-clock time", () => {
    inZone("America/Chicago");
    expect(toDateTimeLocalValue(new Date("2026-01-05T15:07:00Z"))).toBe(
      "2026-01-05T09:07",
    );
  });
});

describe("apiKeyConnectSchema expiry", () => {
  it("accepts today and tomorrow and rejects yesterday in a Chicago evening", () => {
    inZone("America/Chicago");
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 9, 26, 20, 30));

    expect(expiryError("2026-10-27")).toBeUndefined();
    expect(expiryError("2026-10-26")).toBeUndefined();
    expect(expiryError("2026-10-25")).toBe("Expiry must be in the future");
  });

  it("accepts a blank expiry", () => {
    expect(expiryError("")).toBeUndefined();
  });
});

describe("useApiKeyConnectForm", () => {
  it("submits the end of the picked local day", async () => {
    inZone("America/Chicago");
    const queryClient = new QueryClient();
    const wrapper = ({ children }: { children: ReactNode }) =>
      createElement(QueryClientProvider, { client: queryClient }, children);
    const { result } = renderHook(
      () => useApiKeyConnectForm({ provider: "openai", onSuccess: vi.fn() }),
      { wrapper },
    );

    await act(() =>
      result.current.handleSubmit({
        title: "K",
        apiKey: "sk",
        expiresAt: "2026-12-31",
      }),
    );

    expect(postV1CreateCredentials).toHaveBeenCalledWith(
      "openai",
      expect.objectContaining({ expires_at: 1798783199 }),
    );
  });
});
