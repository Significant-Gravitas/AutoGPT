import { PLAN_REQUIRED_MESSAGE } from "../../chat/chatErrors";
import { managedAvatars } from "../catalog";
import {
  canCreate,
  colourFamilies,
  createErrorMessage,
  createExpertBody,
  defaultAvatarPath,
} from "../helpers";

interface Catalog {
  identities: { visual_category: string; url: string }[];
}

// The web's own copy, read by relative path (the app has no Node types for `fs`).
const catalog = jest.requireActual<Catalog>(
  "../../../../../frontend/src/components/molecules/ExpertAvatar/catalog.json",
);

describe("managedAvatars", () => {
  it("matches the web catalog, Otto left out", () => {
    expect(managedAvatars).toEqual(
      catalog.identities
        .filter((identity) => identity.visual_category !== "otto")
        .map((identity) => ({ family: identity.visual_category, url: identity.url })),
    );
  });
});

describe("colourFamilies", () => {
  it("offers every family with an avatar of its own, never Otto's or content", () => {
    expect(colourFamilies.map((option) => option.family)).toEqual([
      "marketing",
      "sales",
      "finance",
      "support",
      "operations",
      "research",
      "development",
      "general",
    ]);
  });
});

describe("defaultAvatarPath", () => {
  it("hashes the trimmed, lower-cased name into the family, as the web does", () => {
    // "juno" hashes to 3273804, and 3273804 % 8 = 4: the fifth marketing avatar, Zara.
    expect(defaultAvatarPath("marketing", "  Juno ")).toBe(
      "/autogpt-characters/v2.1/expert-zara/neutral/128.webp",
    );
  });

  it("gives the general family its one avatar", () => {
    expect(defaultAvatarPath("general", "")).toBe(
      "/autogpt-characters/v2.1/expert-general-01/neutral/128.webp",
    );
  });
});

describe("canCreate", () => {
  it("needs a name with text", () => {
    expect(canCreate("   ")).toBe(false);
    expect(canCreate(" Juno")).toBe(true);
  });
});

describe("createExpertBody", () => {
  it("sends the family's colour token and role with its avatar", () => {
    expect(
      createExpertBody({ name: " Juno ", jobTitle: " Inbox triage ", family: "marketing" }),
    ).toEqual({
      name: "Juno",
      role: "Marketer",
      job_title: "Inbox triage",
      avatar_url: "/autogpt-characters/v2.1/expert-zara/neutral/128.webp",
      color: "orange-300",
    });
  });

  it("sends no job title when it is blank", () => {
    expect(createExpertBody({ name: "Ada", jobTitle: "  ", family: "general" })).toMatchObject({
      job_title: null,
      role: "Assistant",
      color: "amber-300",
    });
  });
});

describe("createErrorMessage", () => {
  it("asks for a plan on a 402", () => {
    expect(createErrorMessage({ status: 402 })).toBe(PLAN_REQUIRED_MESSAGE);
  });

  it("names the team size on a full team", () => {
    expect(createErrorMessage({ status: 409, code: "active_expert_limit", limit: 5 })).toBe(
      "Your team is full (5 experts). Remove one on the web to add another.",
    );
  });

  it("explains the lifetime limit", () => {
    expect(
      createErrorMessage({ status: 409, code: "raised_expert_lifetime_limit", limit: 20 }),
    ).toBe("You've created the most experts your plan allows.");
  });

  it("falls back for anything else", () => {
    expect(createErrorMessage(new Error("offline"))).toBe(
      "Couldn't create that expert. Try again.",
    );
    expect(createErrorMessage({ status: 500 })).toBe("Couldn't create that expert. Try again.");
  });
});
