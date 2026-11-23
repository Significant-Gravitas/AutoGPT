import RJSFForm from "@rjsf/core";
import type { RJSFSchema } from "@rjsf/utils";
import { act } from "@testing-library/react";
import { createRef, type Ref } from "react";
import { describe, expect, it } from "vitest";
import { render } from "@/tests/integrations/test-utils";
import Form from "../registry";
import { customValidator } from "../utils/custom-validator";

const schema: RJSFSchema = {
  type: "object",
  properties: { name: { type: "string", title: "Name" } },
};

function themedForm(ref: Ref<RJSFForm>, name: string) {
  return (
    <Form
      ref={ref}
      schema={schema}
      formData={{ name }}
      validator={customValidator}
      formContext={{ nodeId: "themed-form", showHandles: false }}
    />
  );
}

describe("the themed form", () => {
  it("passes a ref through to the RJSF form, so a caller can validate it", () => {
    const ref = createRef<RJSFForm>();
    render(themedForm(ref, "first"));
    expect(ref.current).toBeInstanceOf(RJSFForm);
    let valid: boolean | undefined;
    act(() => {
      valid = ref.current?.validateForm();
    });
    expect(valid).toBe(true);
  });

  it("keeps its RJSF form across renders", () => {
    const ref = createRef<RJSFForm>();
    const { rerender } = render(themedForm(ref, "first"));
    const rendered = ref.current;
    rerender(themedForm(ref, "second"));
    expect(rendered).toBeInstanceOf(RJSFForm);
    expect(ref.current).toBe(rendered);
  });
});
