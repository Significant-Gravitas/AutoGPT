import {
  ArrayFieldItemTemplateProps,
  FormContextType,
  getTemplate,
  getUiOptions,
  RJSFSchema,
  StrictRJSFSchema,
} from "@rjsf/utils";
import { ArrayItemProvider } from "./context/array-item-context";

export default function ArrayFieldItemTemplate<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(props: ArrayFieldItemTemplateProps<T, S, F>) {
  const {
    children,
    buttonsProps,
    displayLabel,
    hasDescription,
    hasToolbar,
    uiSchema,
    registry,
  } = props;
  const uiOptions = getUiOptions<T, S, F>(uiSchema);
  const ArrayFieldItemButtonsTemplate = getTemplate<
    "ArrayFieldItemButtonsTemplate",
    T,
    S,
    F
  >("ArrayFieldItemButtonsTemplate", registry, uiOptions);

  return (
    <div>
      <div className="mb-2 flex flex-row flex-wrap items-center">
        <div className="shrink grow">
          <ArrayItemProvider>
            <div className="shrink grow">{children}</div>
          </ArrayItemProvider>
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
