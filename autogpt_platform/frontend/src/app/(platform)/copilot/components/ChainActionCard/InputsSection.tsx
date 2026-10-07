"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { FormRenderer } from "@/components/renderers/InputRenderer/FormRenderer";
import { PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { formatInputsTitle, type InputsRequest } from "./helpers";

interface Props {
  requests: InputsRequest[];
}

/** One card per block/agent asking for inputs, titled by its name so two
 *  blocks' forms never blend into one table. */
export function InputsSection({ requests }: Props) {
  const visible = requests.filter(
    (request) => request.schema !== null || request.hasAdvanced,
  );
  if (visible.length === 0) return null;

  return (
    <>
      {visible.map((request) => (
        <div
          key={request.id}
          className="overflow-hidden rounded-3xl border border-zinc-100 bg-white"
        >
          <div className="flex items-center gap-2.5 border-b border-zinc-100 px-4 py-3">
            <Icon icon={PencilEdit02Icon} size={18} className="text-zinc-400" />
            <span className="text-sm font-medium text-zinc-900">
              {request.title
                ? formatInputsTitle(request.title)
                : "Fill in the details"}
            </span>
            {request.title && (
              <span className="ml-auto text-xs text-zinc-400">
                Fill in the details
              </span>
            )}
          </div>

          {/* Each RJSF field carries an inline 16px bottom margin — the
              negative margin swallows the last field's so the card edge
              sits ~8px below the form. */}
          <div className="px-4 pt-2 pb-0">
            {request.schema && (
              <FormRenderer
                jsonSchema={request.schema}
                className="mt-0 -mb-2"
                handleChange={(v) => request.onChange(v.formData ?? {})}
                uiSchema={{
                  "ui:submitButtonOptions": { norender: true },
                }}
                initialValues={request.values}
                formContext={{
                  showHandles: false,
                  size: "small",
                }}
              />
            )}
            {request.hasAdvanced && (
              <Button
                type="button"
                variant="link"
                className="mt-1 mb-2 h-auto min-w-0 p-0 text-xs font-normal text-muted-foreground"
                onClick={request.onToggleAdvanced}
              >
                {request.showAdvanced
                  ? "Hide advanced fields"
                  : "Show advanced fields"}
              </Button>
            )}
          </div>
        </div>
      ))}
    </>
  );
}
