import {
  draftFields,
  fieldValue,
  packFields,
  record,
  type DraftField,
} from "./draft-fields";

function equal(a: unknown, b: unknown) {
  return JSON.stringify(a) === JSON.stringify(b);
}
function mergeValue(before: unknown, after: unknown, local: unknown): unknown {
  if (local === undefined || equal(local, before)) return after;
  if (equal(before, after)) return local;
  const old = record(before),
    next = record(after),
    edited = record(local);
  if (old && next && edited)
    return Object.fromEntries(
      Object.entries(next).map(([key, value]) => [
        key,
        mergeValue(old[key], value, edited[key]),
      ]),
    );
  if (
    Array.isArray(before) &&
    Array.isArray(after) &&
    Array.isArray(local) &&
    [...before, ...after, ...local].every(
      (item) => typeof record(item)?.id === "string",
    )
  ) {
    const oldByID = new Map(before.map((item) => [item.id, item]));
    const localByID = new Map(local.map((item) => [item.id, item]));
    const nextByID = new Map(after.map((item) => [item.id, item]));
    const sourceOrderUnchanged = equal(
      before.filter((item) => nextByID.has(item.id)).map((item) => item.id),
      after.filter((item) => oldByID.has(item.id)).map((item) => item.id),
    );
    const ordered = sourceOrderUnchanged
      ? [
          ...local.filter((item) => nextByID.has(item.id)),
          ...after.filter((item) => !localByID.has(item.id)),
        ]
      : after;
    const values = ordered.flatMap((item) =>
      oldByID.has(item.id) &&
      !localByID.has(item.id) &&
      equal(oldByID.get(item.id), item)
        ? []
        : [
            mergeValue(
              oldByID.get(item.id),
              nextByID.get(item.id),
              localByID.get(item.id),
            ),
          ],
    );
    return [
      ...values,
      ...local.filter(
        (item) => !oldByID.has(item.id) && !nextByID.has(item.id),
      ),
    ];
  }
  return after;
}

function validSelections(field: DraftField, value: unknown) {
  const choices = field.props.options as
    | { id?: string; value?: string }[]
    | undefined;
  if (
    ["ResultList", "SearchSelect", "MultiSelectField"].includes(field.type) &&
    Array.isArray(value)
  )
    return value.filter((id) =>
      choices?.some((choice) => (choice.id ?? choice.value) === id),
    );
  if (
    ["Comparison", "SelectField"].includes(field.type) &&
    choices &&
    !choices.some((choice) => (choice.id ?? choice.value) === value)
  )
    return field.value;
  if (
    field.type === "DataGrid" &&
    record(value) &&
    Array.isArray(record(value)!.rows) &&
    Array.isArray(record(value)!.selected)
  ) {
    const rows = record(value)!.rows as { id: string }[];
    return {
      ...record(value),
      selected: (record(value)!.selected as unknown[]).filter((id) =>
        rows.some((row) => row.id === id),
      ),
    };
  }
  if (field.type === "RankedList" && Array.isArray(value)) {
    const ids = (field.props.items as { id: string }[]).map((item) => item.id);
    return [
      ...value.filter((id) => ids.includes(String(id))),
      ...ids.filter((id) => !value.includes(id)),
    ];
  }
  return value;
}

function checklistProgress(old: DraftField, field: DraftField, value: unknown) {
  const before = old.props.items as { title: string; done?: boolean }[];
  const after = field.props.items as { title: string; done?: boolean }[];
  if (
    !Array.isArray(value) ||
    new Set(before.map((item) => item.title)).size !== before.length
  )
    return field.value;
  return after.flatMap((item, index) => {
    const previous = before.findIndex(
      (previous) => previous.title === item.title,
    );
    const completed =
      previous < 0 || !!before[previous].done !== !!item.done
        ? !!item.done
        : value.includes(previous);
    return completed ? [index] : [];
  });
}

export function mergeDraft(
  previousSource: string,
  source: string,
  state: Record<string, unknown>,
) {
  const previous = draftFields(previousSource);
  let revised = 0;
  const fields = draftFields(source).map((field) => {
    const old = previous.find(
      (item) => item.form === field.form && item.key === field.key,
    );
    if (!old || !equal(old.contract, field.contract)) {
      if (old) revised++;
      return { field, value: field.value };
    }
    const local = fieldValue(state, old);
    const value = validSelections(
      field,
      field.type === "Checklist"
        ? checklistProgress(old, field, local)
        : mergeValue(old.value, field.value, local),
    );
    if (
      local !== undefined &&
      !equal(value, local) &&
      !equal(old.value, field.value)
    )
      revised++;
    return { field, value };
  });
  return { state: packFields(fields), revised };
}
