import { focusManager, onlineManager } from "@tanstack/react-query";
import { act, cleanup, render, screen } from "@testing-library/react";
import { useState } from "react";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ActivationResponse } from "@/app/api/__generated__/models/activationResponse";
import { server } from "@/mocks/mock-server";
import { useActivationRecovery } from "../useActivationRecovery";
import { quote } from "./fixtures";

const requests = vi.fn();
let response: ActivationResponse;

function Subject() {
  const [attempt, setAttempt] = useState<ActivationResponse>({
    ...quote,
    status: "processing",
    retry_after_seconds: 3,
  });
  useActivationRecovery({
    userID: "user-1",
    attempt,
    accept: setAttempt,
    onError: vi.fn(),
  });
  return <span>{attempt.status}</span>;
}

async function advance(milliseconds: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(milliseconds);
  });
}

async function expectReady() {
  await vi.waitFor(async () => {
    await advance(1);
    expect(screen.getByText("ready")).toBeDefined();
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  focusManager.setFocused(true);
  onlineManager.setOnline(true);
  requests.mockReset();
  response = { ...quote, status: "processing", retry_after_seconds: 3 };
  server.use(
    http.get(
      "*/api/proxy/api/credits/pro-activation/current",
      () => new HttpResponse(null, { status: 404 }),
    ),
    http.get("*/api/proxy/api/credits/pro-activation/attempt-1", () => {
      requests();
      return HttpResponse.json(response);
    }),
  );
});

afterEach(() => {
  cleanup();
  focusManager.setFocused(undefined);
  onlineManager.setOnline(true);
  vi.useRealTimers();
});

describe("pending activation synchronization", () => {
  it("pauses while hidden or offline, checks immediately on return, and stops at ready", async () => {
    render(<Subject />);
    act(() => focusManager.setFocused(false));
    await advance(30_000);
    expect(requests).not.toHaveBeenCalled();
    act(() => onlineManager.setOnline(false));
    act(() => focusManager.setFocused(true));
    await advance(30_000);
    expect(requests).not.toHaveBeenCalled();
    response = { ...quote, status: "ready" };
    act(() => onlineManager.setOnline(true));
    await expectReady();
    expect(requests).toHaveBeenCalledTimes(1);
    await advance(30_000);
    act(() => focusManager.setFocused(false));
    act(() => focusManager.setFocused(true));
    await advance(3_000);
    expect(requests).toHaveBeenCalledTimes(1);
  });

  it("keeps one in-flight status check when visibility changes during the request", async () => {
    let release!: () => void;
    const pending = new Promise<void>((resolve) => {
      release = resolve;
    });
    server.use(
      http.get("*/api/proxy/api/credits/pro-activation/attempt-1", async () => {
        requests();
        await pending;
        return HttpResponse.json({ ...quote, status: "ready" });
      }),
    );
    render(<Subject />);
    await advance(3_000);
    await vi.waitFor(() => expect(requests).toHaveBeenCalledTimes(1));
    act(() => focusManager.setFocused(false));
    act(() => focusManager.setFocused(true));
    await advance(9_000);
    expect(requests).toHaveBeenCalledTimes(1);
    release();
    await expectReady();
    await advance(30_000);
    expect(requests).toHaveBeenCalledTimes(1);
  });
});
