import { useState } from "react";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";

interface Props {
  serverURL: string;
  onChange: (url: string) => void;
  readOnly: boolean;
  disabled: boolean;
  options?: { label: string; url: string }[];
}

export function MCPServerURLField({
  serverURL,
  onChange,
  readOnly,
  disabled,
  options = [],
}: Props) {
  const [customSelected, setCustomSelected] = useState(false);
  const matched = options.find((option) => option.url === serverURL);
  const selection = customSelected ? "custom" : (matched?.url ?? "");

  function selectURL(value: string) {
    setCustomSelected(value === "custom");
    onChange(value === "custom" ? "" : value);
  }

  function changeURL(value: string) {
    if (options.length) setCustomSelected(true);
    onChange(value);
  }

  return (
    <>
      {!readOnly && options.length > 0 && (
        <Select
          id="mcp-server-region"
          label="Server region"
          placeholder="Choose your account's region"
          value={selection}
          onValueChange={selectURL}
          disabled={disabled}
          options={[
            ...options.map((option) => ({
              label: option.label,
              value: option.url,
            })),
            { label: "Custom URL", value: "custom" },
          ]}
        />
      )}
      <Input
        id="mcp-server-url"
        label="Server URL"
        type="url"
        placeholder="https://mcp.example.com"
        value={serverURL}
        onChange={(event) => changeURL(event.target.value)}
        disabled={disabled}
        readOnly={readOnly || Boolean(matched && !customSelected)}
        autoFocus={!readOnly && options.length === 0}
      />
    </>
  );
}
