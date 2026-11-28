import { noticeLinkTarget, parseNotice } from "../notice";

const base = { kind: "scheduled_chat_notice", headline: "Skipped", body: "No budget left." };

test("only a scheduled-chat notice with a headline and a body is a notice", () => {
  expect(parseNotice({ kind: "other", headline: "x", body: "y" })).toBeNull();
  expect(parseNotice({ ...base, body: "" })).toBeNull();
  expect(parseNotice(base)).toEqual({ reason: "usage", headline: "Skipped", body: "No budget left.", scheduleKey: null, actions: [] });
});

test.each(["https://evil.example", "//evil.example", "/\\evil.example", "settings/billing"])("drops a link to %s", (href) => {
  expect(parseNotice({ ...base, actions: [{ kind: "link", label: "Open", href }] })?.actions).toEqual([]);
});

test("resume and run less often need to know which chat", () => {
  const actions = [{ kind: "resume", label: "Resume" }, { kind: "run_less_often", label: "Run less often" }, { kind: "reconnect", label: "Reconnect", provider: "github" }];
  expect(parseNotice({ ...base, actions })?.actions.map((a) => a.kind)).toEqual(["reconnect"]);
  expect(parseNotice({ ...base, schedule_key: "s1", actions })?.actions.map((a) => a.kind)).toEqual(["resume", "run_less_often", "reconnect"]);
});

test("billing opens the plan screen; anything else opens the web", () => {
  expect(noticeLinkTarget("/settings/billing", "https://app.example")).toEqual({ kind: "plan" });
  expect(noticeLinkTarget("/settings/integrations", "https://app.example")).toEqual({ kind: "web", url: "https://app.example/settings/integrations" });
});
