import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import {
  ArrayFieldItemTemplateProps,
  getTemplate,
  getUiOptions,
} from "@rjsf/utils";

export default function ArrayFieldItemTemplate(
  props: ArrayFieldItemTemplateProps,
) {
  const {
    children,
    buttonsProps,
    displayLabel,
    hasToolbar,
    uiSchema,
    registry,
  } = props;
  const uiOptions = getUiOptions(uiSchema);
  const ArrayFieldItemButtonsTemplate = getTemplate(
    "ArrayFieldItemButtonsTemplate",
    registry,
    uiOptions,
  );

  return (
    <div>
      <div className="mb-2 flex flex-row flex-wrap items-center">
        <div className="shrink grow">
          <div className="shrink grow">{children}</div>
        </div>
        <div className="flex items-end justify-end p-0.5">
          {hasToolbar && (
            <div
              className="flex gap-2"
              style={{
                marginLeft: "5px",
                marginTop: displayLabel ? `-6px` : undefined,
              }}
            >
              <ArrayFieldItemButtonsTemplate {...buttonsProps} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
