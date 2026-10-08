import { describe, expect, it } from "vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse, delay } from "msw";
import { server } from "@/mocks/mock-server";
import { render } from "@/tests/integrations/test-utils";
import { OpenUILab } from "../components/OpenUILab/OpenUILab";

const source =
  'root = Workspace("A live workspace", "A response from the configured model", [note])\nnote = Insight("Start here", "An example generated response", "neutral")';

function beginLiveRequest() {
  fireEvent.click(screen.getByRole("button", { name: "Live AI" }));
  fireEvent.change(screen.getByLabelText("Message"), {
    target: { value: "Make a new workspace" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Send message" }));
}

describe("live workspace generation", () => {
  it("renders a streamed response and labels its provenance accurately", async () => {
    server.use(
      http.post(
        "*/api/openui",
        () =>
          new HttpResponse(
            JSON.stringify({ type: "delta", text: source }) +
              '\n{"type":"done"}\n',
          ),
      ),
    );
    render(<OpenUILab liveAvailable />);
    fireEvent.click(screen.getByRole("button", { name: "Live AI" }));
    expect(
      screen.getByText("Sample data · no connected accounts"),
    ).toBeDefined();
    beginLiveRequest();
    expect(await screen.findByText("A live workspace")).toBeDefined();
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Stop generation" }),
      ).toBeNull(),
    );
    expect(screen.getByText("AI-generated · review before use")).toBeDefined();
  });

  it.each([
    [
      "provider error",
      '{"type":"error","message":"Provider unavailable. Please try again."}\n',
    ],
    [
      "truncated stream",
      JSON.stringify({ type: "delta", text: source }) + "\n",
    ],
    [
      "invalid UI",
      '{"type":"delta","text":"root = UnknownWidget()"}\n{"type":"done"}\n',
    ],
  ])("keeps the completed workspace after a %s", async (_name, body) => {
    server.use(http.post("*/api/openui", () => new HttpResponse(body)));
    render(<OpenUILab liveAvailable />);
    beginLiveRequest();
    expect(await screen.findByRole("alert")).toBeDefined();
    expect(screen.getByText("Your agents, at a glance")).toBeDefined();
    expect(
      screen.getByText("Sample data · no connected accounts"),
    ).toBeDefined();
  });

  it("ignores a late response after cancellation", async () => {
    server.use(
      http.post("*/api/openui", async () => {
        await delay(150);
        return new HttpResponse(
          JSON.stringify({ type: "delta", text: source }) +
            '\n{"type":"done"}\n',
        );
      }),
    );
    render(<OpenUILab liveAvailable />);
    beginLiveRequest();
    fireEvent.click(
      await screen.findByRole("button", { name: "Stop generation" }),
    );
    await delay(200);
    expect(screen.getByText("Your agents, at a glance")).toBeDefined();
    expect(screen.queryByText("A live workspace")).toBeNull();
  });
});
