import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { render } from "@/tests/integrations/test-utils";
import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import type { FieldProps } from "@rjsf/utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { CredentialsField } from "../CredentialField";

const ollamaSchema = {
  credentials_provider: ["ollama"],
  credentials_types: ["api_key"],
} as BlockIOCredentialsSubSchema;

function makeFieldProps(overrides: Partial<FieldProps> = {}): FieldProps {
  return {
    formData: undefined,
    onChange: vi.fn(),
    schema: ollamaSchema,
    registry: {
      formContext: { nodeId: "node-1" },
    },
    fieldPathId: { path: ["credentials"] },
    required: true,
    ...overrides,
  } as FieldProps;
}

describe("CredentialsField", () => {
  beforeEach(() => {
    useNodeStore.setState({
      nodes: [
        {
          id: "node-1",
          data: { hardcodedValues: {}, metadata: {} },
        },
      ],
      setCredentialsOptional: vi.fn(),
    } as Partial<ReturnType<typeof useNodeStore.getState>>);
  });

  it("renders nothing when provider is ollama", () => {
    const { container } = render(<CredentialsField {...makeFieldProps()} />);

    expect(container.firstChild).toBeNull();
  });
});
