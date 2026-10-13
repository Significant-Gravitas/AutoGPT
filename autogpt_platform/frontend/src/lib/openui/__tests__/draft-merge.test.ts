import { describe, expect, it } from "vitest";
import { mergeDraft } from "../draft-merge";
import { draftFields, fieldValue, packFields } from "../draft-fields";
import { dataDashboard, formView } from "./capability-fixtures";

describe("reconciling drafts with model revisions", () => {
  it("retains a user's cell edit while accepting a new source row and removing a deleted one", () => {
    const before = dataDashboard();
    const field = draftFields(before).find(
      (field) => field.type === "DataGrid",
    )!;
    const state = packFields([
      {
        field,
        value: {
          rows: [
            { id: "jan", cells: { month: "January", revenue: 150, cost: 40 } },
            {
              id: "feb",
              cells: { month: "February", revenue: 200, cost: null },
            },
          ],
          selected: ["feb"],
        },
      },
    ]);
    const after = dataDashboard([
      { id: "jan", cells: { month: "January", revenue: 100, cost: 40 } },
      { id: "mar", cells: { month: "March", revenue: 300, cost: 40 } },
    ]);
    const result = mergeDraft(before, after, state);
    expect(
      fieldValue(
        result.state,
        draftFields(after).find((field) => field.type === "DataGrid")!,
      ),
    ).toEqual({
      rows: [
        { id: "jan", cells: { month: "January", revenue: 150, cost: 40 } },
        { id: "mar", cells: { month: "March", revenue: 300, cost: 40 } },
      ],
      selected: [],
    });
  });
  it("uses an explicit revised value and reports the change instead of silently restoring stale input", () => {
    const before = formView('NumberField("count", "Guests", 2, 1, 10, 1)');
    const after = before.replace('"Guests", 2,', '"Guests", 4,');
    const state = packFields([{ field: draftFields(before)[0], value: 3 }]);
    expect(mergeDraft(before, after, state).revised).toBe(1);
    expect(
      fieldValue(mergeDraft(before, after, state).state, draftFields(after)[0]),
    ).toBe(4);
  });
  it("does not transfer edits between different control types or option identities", () => {
    const before = formView(
      'SelectField("room", "Room", "a", [{label:"A",value:"a"},{label:"B",value:"b"}])',
    );
    const after = before.replace(
      '{label:"B",value:"b"}',
      '{label:"C",value:"c"}',
    );
    const state = packFields([{ field: draftFields(before)[0], value: "b" }]);
    expect(
      fieldValue(mergeDraft(before, after, state).state, draftFields(after)[0]),
    ).toBe("a");
  });
});

describe("checklist identity across revisions", () => {
  it("keeps completion attached to the item after reorder and additions", () => {
    const before =
      'root = Workspace("Trip", "", [list], "trip-view")\nlist = Checklist("Before leaving", [{title:"Pack",detail:"Bag",done:false},{title:"Book",detail:"Room",done:false}])';
    const after =
      'root = Workspace("Trip renamed", "", [list], "trip-view")\nlist = Checklist("Before leaving", [{title:"Book",detail:"Room",done:false},{title:"Pack",detail:"Bag",done:false},{title:"Keys",detail:"Door",done:false}])';
    const field = draftFields(before)[0];
    const state = packFields([{ field, value: [0] }]);
    expect(
      fieldValue(mergeDraft(before, after, state).state, draftFields(after)[0]),
    ).toEqual([1]);
  });
});
