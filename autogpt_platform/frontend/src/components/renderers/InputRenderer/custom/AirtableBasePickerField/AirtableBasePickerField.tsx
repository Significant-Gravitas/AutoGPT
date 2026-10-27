"use client";

import { Text } from "@/components/atoms/Text/Text";
import { FieldProps } from "@rjsf/utils";
import { BaseDropdown } from "./components/BaseDropdown";
import {
  BasesError,
  LoadingBases,
  NewBaseActions,
  NoBasesYet,
} from "./components/PickerStates";
import {
  allows,
  noUsableBaseText,
  pickerAccess,
  pickerCredentialsID,
  pickerHint,
} from "./helpers";
import { useAirtableBasePickerField } from "./useAirtableBasePickerField";

export function AirtableBasePickerField(props: FieldProps) {
  const credentialsID = pickerCredentialsID(props.schema);
  if (!credentialsID) return null;
  return <BasePicker {...props} credentialsID={credentialsID} />;
}

function BasePicker({
  schema,
  formData,
  onChange,
  fieldPathId,
  credentialsID,
}: FieldProps & { credentialsID: string }) {
  const picker = useAirtableBasePickerField(credentialsID);
  const value = typeof formData === "string" ? formData : "";
  const labelID = `${fieldPathId.$id}-label`;
  const needed = pickerAccess(schema);
  const ready = !picker.isLoading && !picker.error && picker.bases.length > 0;
  const noneUsable =
    ready && picker.bases.every((base) => !allows(base.access, needed));

  function body() {
    if (picker.isLoading) return <LoadingBases />;
    if (picker.error)
      return (
        <BasesError
          message={picker.error}
          onRetry={picker.refresh}
          retrying={picker.isRefreshing}
        />
      );
    if (picker.bases.length === 0)
      return (
        <NoBasesYet
          onRefresh={picker.refresh}
          refreshing={picker.isRefreshing}
        />
      );
    return (
      <BaseDropdown
        id={fieldPathId.$id}
        bases={picker.bases}
        shown={picker.shown}
        value={value}
        labelID={labelID}
        needed={needed}
        open={picker.open}
        onOpenChange={picker.setOpen}
        search={picker.search}
        onSearchChange={picker.setSearch}
        onChange={(id) => onChange(id, fieldPathId.path)}
      />
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1">
        <Text variant="body-medium" id={labelID} className="text-zinc-900">
          Choose an Airtable base
        </Text>
        <Text variant="small" id={`${labelID}-hint`} className="text-zinc-600">
          {pickerHint(needed)}
        </Text>
      </div>
      {body()}
      {noneUsable && (
        <Text variant="small" className="text-zinc-600">
          {noUsableBaseText(needed)}
        </Text>
      )}
      {ready && picker.truncated && (
        <Text variant="small" className="text-zinc-500">
          Showing the first {picker.bases.length} of your bases.
        </Text>
      )}
      {ready && (
        <div className="flex flex-wrap items-center gap-x-2">
          <Text variant="small" className="text-zinc-500">
            Want a new, empty one?
          </Text>
          <NewBaseActions
            onRefresh={picker.refresh}
            refreshing={picker.isRefreshing}
          />
        </div>
      )}
    </div>
  );
}
