import { eraseWarning, factText, understandingRows, withFactBack, withoutFact } from "../helpers";

test("lists only what Otto has kept", () => {
  expect(understandingRows({ updated_at: null })).toEqual([]);
  expect(understandingRows({ updated_at: "2026-11-01T00:00:00Z", business_name: "Acme", industry: "", current_software: ["Xero", "Slack"], pain_points: [] })).toEqual([
    { label: "Business", value: "Acme" },
    { label: "Tools you use", value: "Xero, Slack" },
  ]);
});
test("a memory reads as its fact, else its name, else its two ends", () => {
  expect(factText({ fact: "Invoices go out on Fridays", name: "n", source: "a", target: "b" })).toBe("Invoices go out on Fridays");
  expect(factText({ fact: null, name: "Billing day", source: "a", target: "b" })).toBe("Billing day");
  expect(factText({ fact: null, name: null, source: "Acme", target: "Xero" })).toBe("Acme → Xero");
});
test("the erase warning counts what goes", () => {
  expect(eraseWarning({ facts: 1 })).toBe("This erases 1 memory and what Otto knows about your work. It can't be undone.");
  expect(eraseWarning({ facts: 12 })).toBe("This erases 12 memories and what Otto knows about your work. It can't be undone.");
});
test("a failed forget puts back only its own row", () => {
  const row = (uuid: string) => ({ uuid, fact: uuid, name: null, source: "", target: "", created_at: null });
  const before = { items: [row("a"), row("b"), row("c")] };
  const now = withoutFact(withoutFact(before, "a"), "c");
  expect(now.items.map((item) => item.uuid)).toEqual(["b"]);
  expect(withFactBack(before, now, "a").items.map((item) => item.uuid)).toEqual(["a", "b"]);
});
