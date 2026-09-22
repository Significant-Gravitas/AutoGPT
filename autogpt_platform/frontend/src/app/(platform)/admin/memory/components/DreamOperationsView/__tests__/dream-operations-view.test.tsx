import { describe, expect, test } from "vitest";

import { render, screen } from "@/tests/integrations/test-utils";
import type { DreamOperationsSnapshot } from "@/app/api/__generated__/models/dreamOperationsSnapshot";

import { DreamOperationsView } from "../DreamOperationsView";

function sampleSnapshot(): DreamOperationsSnapshot {
  return {
    writes: [
      {
        edge_uuid: "edge-write-aaaaaaaaaaaaaaaaaa",
        content: "User prefers terse code reviews.",
        scope: "preferences",
        confidence: 0.92,
        status: "active",
        source_episode_uuids: ["epi-1", "epi-2"],
      },
    ],
    proposals: [
      {
        edge_uuid: null,
        content: "User is exploring memory consolidation work.",
        scope: "interests",
        confidence: 0.6,
        status: "tentative",
        source_episode_uuids: ["epi-3"],
      },
      {
        edge_uuid: "edge-prop-bbbbbbbbbbbbbbbbbb",
        content: "Possibly working at AutoGPT.",
        confidence: null,
        status: "tentative",
      },
    ],
    demotions: [
      {
        edge_uuid: "edge-demo-ccccccccccccccccc",
        reason: "Contradicted by newer episode",
        new_status: "superseded",
        applied: true,
      },
    ],
    entity_invalidations: [
      {
        entity_uuid: "ent-ddddddddddddddddddd",
        reason: "Merged with another entity",
        edges_touched: [
          "edge-x-111",
          "edge-x-222",
          "edge-x-333",
          "edge-x-444",
          "edge-x-555",
          "edge-x-666",
          "edge-x-777",
        ],
      },
    ],
  };
}

describe("DreamOperationsView", () => {
  test("renders all four sections with their item counts", async () => {
    render(<DreamOperationsView operations={sampleSnapshot()} />);

    expect(await screen.findByText(/writes/i)).toBeDefined();
    expect(await screen.findByText(/proposals/i)).toBeDefined();
    expect(await screen.findByText(/demotions/i)).toBeDefined();
    expect(await screen.findByText(/entity invalidations/i)).toBeDefined();

    // Each section's count badge is shown next to the label.
    // writes: 1, proposals: 2, demotions: 1, entities: 1.
    expect(screen.getAllByText("1").length).toBeGreaterThanOrEqual(3);
    expect(screen.getByText("2")).toBeDefined();
  });

  test("renders sample row content + shortened edge UUIDs", async () => {
    render(<DreamOperationsView operations={sampleSnapshot()} />);

    expect(
      await screen.findByText("User prefers terse code reviews."),
    ).toBeDefined();
    expect(
      await screen.findByText("Contradicted by newer episode"),
    ).toBeDefined();
    expect(await screen.findByText("Merged with another entity")).toBeDefined();

    // Edge UUID is shown shortened with the `edge:` prefix.
    expect(screen.getByText((c) => c.includes("edge-write-"))).toBeDefined();

    // Confidence rendered as percentage.
    expect(screen.getByText(/confidence: 92%/i)).toBeDefined();
  });

  test("shows skeleton/empty state when operations is null", async () => {
    render(<DreamOperationsView operations={null} />);
    expect(
      await screen.findByText(/no per-edge operations were returned/i),
    ).toBeDefined();
  });

  test("shows per-section empty rows when an array is empty", async () => {
    render(
      <DreamOperationsView
        operations={{
          writes: [],
          proposals: [],
          demotions: [],
          entity_invalidations: [],
        }}
      />,
    );
    expect(await screen.findByText(/no writes recorded/i)).toBeDefined();
    expect(screen.getByText(/no proposals recorded/i)).toBeDefined();
    expect(screen.getByText(/no demotions recorded/i)).toBeDefined();
    expect(screen.getByText(/no entity invalidations recorded/i)).toBeDefined();
  });
});
