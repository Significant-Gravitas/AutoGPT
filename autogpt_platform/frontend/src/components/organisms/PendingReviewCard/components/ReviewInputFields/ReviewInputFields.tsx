import { RunAgentInputs } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/modals/RunAgentInputs/RunAgentInputs";
import { Text } from "@/components/atoms/Text/Text";
import type { ReviewField } from "./helpers";

interface Props {
  fields: ReviewField[];
  values: Record<string, unknown>;
  onChange: (values: Record<string, unknown>) => void;
  readOnly?: boolean;
}

export function ReviewInputFields({
  fields,
  values,
  onChange,
  readOnly = false,
}: Props) {
  return (
    <div className="flex flex-col gap-4">
      {fields.map((field) => (
        <ReviewInputField
          key={field.key}
          field={field}
          value={values[field.key]}
          readOnly={readOnly}
          onChange={(value) => onChange({ ...values, [field.key]: value })}
        />
      ))}
    </div>
  );
}

function ReviewInputField({
  field,
  value,
  readOnly,
  onChange,
}: {
  field: ReviewField;
  value: unknown;
  readOnly: boolean;
  onChange: (value: unknown) => void;
}) {
  switch (field.kind) {
    case "input":
      // RunAgentInputs' readOnly only blocks the pointer; a keyboard could
      // still edit a value the approval would never send.
      if (readOnly) {
        return <StaticField label={field.label} text={displayValue(value)} />;
      }
      return (
        <RunAgentInputs
          schema={field.schema}
          value={value}
          onChange={onChange}
        />
      );
    case "group":
      return (
        <fieldset className="flex flex-col gap-3 rounded-lg border border-zinc-200 p-3">
          <legend className="px-1 text-sm font-medium">{field.label}</legend>
          <ReviewInputFields
            fields={field.fields}
            values={(value ?? {}) as Record<string, unknown>}
            onChange={onChange}
            readOnly={readOnly}
          />
        </fieldset>
      );
    case "credentials":
      return <StaticField label={field.label} text={field.text} />;
    case "secret":
      return <StaticField label={field.label} text="••••••••" />;
    case "raw":
      return (
        <StaticField
          label={field.label}
          text={JSON.stringify(field.value, null, 2)}
          mono
        />
      );
  }
}

function StaticField({
  label,
  text,
  mono = false,
}: {
  label: string;
  text: string;
  mono?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1">
      <Text variant="large-medium">{label}</Text>
      <Text
        variant="body"
        className={mono ? "whitespace-pre-wrap font-mono text-sm" : undefined}
      >
        {text}
      </Text>
    </div>
  );
}

function displayValue(value: unknown) {
  if (value === undefined || value === null || value === "") return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return typeof value === "string" ? value : JSON.stringify(value);
}
