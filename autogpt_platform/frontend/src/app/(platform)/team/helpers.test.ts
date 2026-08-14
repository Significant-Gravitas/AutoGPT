import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertPod } from "@/app/api/__generated__/models/expertPod";
import { describe, expect, test } from "vitest";
import { groupExpertsByPods } from "./helpers";

function makeExpert(id: string, podId: string | null = null): Expert {
  return {
    id,
    name: id,
    avatar_url: null,
    role: "Role",
    tagline: null,
    bio: null,
    skills: [],
    identity: "identity",
    voice_preferences: "",
    boundaries: "",
    protected_soul_rules: [],
    is_template: false,
    source_template_id: null,
    is_archived: false,
    workflows: [],
    pod_id: podId,
  };
}

function makePod(id: string, name: string): ExpertPod {
  return { id, name, created_at: new Date("2026-08-14T00:00:00Z") };
}

describe("groupExpertsByPods", () => {
  test("places experts under their pod and the rest ungrouped", () => {
    const growth = makePod("pod-growth", "Growth");
    const support = makePod("pod-support", "Support");
    const { groups, ungrouped } = groupExpertsByPods(
      [
        makeExpert("maria", "pod-growth"),
        makeExpert("sam", "pod-support"),
        makeExpert("lee"),
      ],
      [growth, support],
    );

    expect(groups.map((g) => g.pod.name)).toEqual(["Growth", "Support"]);
    expect(groups[0].experts.map((e) => e.id)).toEqual(["maria"]);
    expect(groups[1].experts.map((e) => e.id)).toEqual(["sam"]);
    expect(ungrouped.map((e) => e.id)).toEqual(["lee"]);
  });

  test("keeps an empty pod as a group", () => {
    const { groups } = groupExpertsByPods(
      [makeExpert("lee")],
      [makePod("pod-empty", "Empty")],
    );
    expect(groups).toHaveLength(1);
    expect(groups[0].experts).toEqual([]);
  });

  test("treats a dangling pod_id as ungrouped", () => {
    const { groups, ungrouped } = groupExpertsByPods(
      [makeExpert("maria", "pod-deleted")],
      [makePod("pod-growth", "Growth")],
    );
    expect(groups[0].experts).toEqual([]);
    expect(ungrouped.map((e) => e.id)).toEqual(["maria"]);
  });

  test("returns everything ungrouped when there are no pods", () => {
    const { groups, ungrouped } = groupExpertsByPods(
      [makeExpert("maria"), makeExpert("sam")],
      [],
    );
    expect(groups).toEqual([]);
    expect(ungrouped).toHaveLength(2);
  });
});
