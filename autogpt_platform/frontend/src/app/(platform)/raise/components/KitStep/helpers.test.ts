import { describe, expect, test } from "vitest";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import {
  combineSearchHits,
  failedAttachmentMessage,
  limitSearchHits,
  parseCredits,
  scoreSearchHit,
  toRaiseAttachments,
  type SearchHit,
} from "./helpers";

const storeAgent = {
  slug: "seo-writer",
  agent_name: "SEO Blog Writer",
  agent_image: "",
  creator: "Acme",
  creator_avatar: "",
  sub_heading: "Writes posts",
  description: "",
  runs: 1,
  rating: 5,
  agent_graph_id: "graph-1",
} as StoreAgent;

describe("kit helpers", () => {
  test("parses credit amounts and rejects junk", () => {
    expect(parseCredits("500")).toBe(500);
    expect(parseCredits(" 0 ")).toBe(0);
    expect(parseCredits("")).toBeNull();
    expect(parseCredits("12.5")).toBeNull();
    expect(parseCredits("-1")).toBeNull();
  });

  test("combines marketplace and library workflows in marketplace scope", () => {
    const hits = combineSearchHits({
      query: "seo",
      storeAgents: [storeAgent],
      libraryAgents: [{ id: "lib-1", name: "Local SEO" } as LibraryAgent],
      skills: [{ name: "seo-audit", description: "Audit pages" }],
      scope: "marketplace",
    });

    expect(hits.map((hit) => hit.subtitle)).toEqual([
      "Marketplace workflow",
      "Library workflow",
    ]);
  });

  test("combines marketplace-as-skill and library skills", () => {
    const hits = combineSearchHits({
      query: "seo",
      storeAgents: [storeAgent],
      libraryAgents: [{ id: "lib-1", name: "Local SEO" } as LibraryAgent],
      skills: [
        { name: "seo-audit", description: "Audit pages" },
        { name: "unrelated", description: "Something else" },
      ],
      scope: "skills",
    });

    expect(hits.map((hit) => hit.subtitle)).toEqual([
      "Marketplace skill",
      "Library skill",
    ]);
    expect(
      hits.find((hit) => hit.kind === "skill" && hit.source === "library")?.id,
    ).toBe("seo-audit");
  });

  test("limits default marketplace results to three items", () => {
    const hits = combineSearchHits({
      query: "",
      storeAgents: [
        { ...storeAgent, agent_name: "Workflow 1", slug: "wf-1" },
        { ...storeAgent, agent_name: "Workflow 2", slug: "wf-2" },
        { ...storeAgent, agent_name: "Workflow 3", slug: "wf-3" },
        { ...storeAgent, agent_name: "Workflow 4", slug: "wf-4" },
      ],
      libraryAgents: [
        { id: "lib-1", name: "Library 1" } as LibraryAgent,
        { id: "lib-2", name: "Library 2" } as LibraryAgent,
      ],
      skills: [],
      scope: "marketplace",
    });

    expect(hits).toHaveLength(3);
    expect(hits.map((hit) => hit.name)).toEqual([
      "Workflow 1",
      "Workflow 2",
      "Workflow 3",
    ]);
  });

  test("limits default skills results to three library skills", () => {
    const hits = combineSearchHits({
      query: "",
      storeAgents: [],
      libraryAgents: [],
      skills: [
        { name: "skill-a", description: "A" },
        { name: "skill-b", description: "B" },
        { name: "skill-c", description: "C" },
        { name: "skill-d", description: "D" },
      ],
      scope: "skills",
    });

    expect(hits).toHaveLength(3);
    expect(hits.map((hit) => hit.id)).toEqual([
      "skill-a",
      "skill-b",
      "skill-c",
    ]);
  });

  test("ranks search hits by relevance and keeps the top three", () => {
    const hits: SearchHit[] = [
      {
        key: "1",
        name: "Content helper",
        subtitle: "Library skill",
        kind: "skill",
        source: "library",
        id: "content-helper",
        description: "General helper",
      },
      {
        key: "2",
        name: "seo-audit",
        subtitle: "Library skill",
        kind: "skill",
        source: "library",
        id: "seo-audit",
        description: "Audit pages",
      },
      {
        key: "3",
        name: "Writer",
        subtitle: "Library skill",
        kind: "skill",
        source: "library",
        id: "writer",
        description: "seo templates",
      },
      {
        key: "4",
        name: "Unrelated",
        subtitle: "Library skill",
        kind: "skill",
        source: "library",
        id: "unrelated",
        description: "Nothing here",
      },
    ];

    expect(scoreSearchHit(hits[1], "seo")).toBeGreaterThan(
      scoreSearchHit(hits[0], "seo"),
    );
    expect(limitSearchHits(hits, "seo").map((hit) => hit.id)).toEqual([
      "seo-audit",
      "writer",
      "content-helper",
    ]);
  });

  test("maps drafts to the raise payload shape", () => {
    expect(
      toRaiseAttachments([
        {
          kind: "workflow",
          source: "marketplace",
          id: "listing-1",
          name: "SEO Blog Writer",
          marketplaceKey: "acme/seo-writer",
        },
      ]),
    ).toEqual([{ kind: "workflow", source: "marketplace", id: "listing-1" }]);
  });

  test("explains failed attachments by name", () => {
    expect(
      failedAttachmentMessage(
        [
          {
            kind: "workflow",
            source: "marketplace",
            id: "listing-1",
            reason: "unavailable",
          },
        ],
        [
          {
            kind: "workflow",
            source: "marketplace",
            id: "listing-1",
            name: "SEO Blog Writer",
          },
        ],
      ),
    ).toBe("SEO Blog Writer is no longer available");
  });

  test("separates multiple failed attachments into sentences", () => {
    expect(
      failedAttachmentMessage(
        [
          {
            kind: "workflow",
            source: "marketplace",
            id: "listing-1",
            reason: "unavailable",
          },
          {
            kind: "skill",
            source: "library",
            id: "seo-audit",
            reason: "installation_failed",
          },
        ],
        [
          {
            kind: "workflow",
            source: "marketplace",
            id: "listing-1",
            name: "SEO Blog Writer",
          },
          {
            kind: "skill",
            source: "library",
            id: "seo-audit",
            name: "SEO audit",
          },
        ],
      ),
    ).toBe(
      "SEO Blog Writer is no longer available. SEO audit couldn't be installed",
    );
  });
});
