import { ResponseType } from "@/app/api/__generated__/models/responseType";
import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import { BlockDetailsCard } from "./BlockDetailsCard";

describe("BlockDetailsCard field labels", () => {
  // The backend drops a title its property name reproduces, so the card must
  // derive that label itself and still prefer one a person wrote.
  it("labels an untitled field from its name and keeps an authored title", () => {
    render(
      <BlockDetailsCard
        output={{
          type: ResponseType.block_details,
          message: "Block details",
          user_authenticated: true,
          block: {
            id: "block-id",
            name: "Example Block",
            description: "",
            inputs: {
              properties: {
                repo_url: { type: "string" },
                sys_prompt: { type: "string", title: "System Prompt" },
              },
            },
            outputs: {},
            credentials: [],
          },
        }}
      />,
    );
    expect(screen.getByText("Repo Url")).not.toBeNull();
    expect(screen.getByText("System Prompt")).not.toBeNull();
    expect(screen.queryByText("repo_url")).toBeNull();
    expect(screen.queryByText("Sys Prompt")).toBeNull();
  });
});
