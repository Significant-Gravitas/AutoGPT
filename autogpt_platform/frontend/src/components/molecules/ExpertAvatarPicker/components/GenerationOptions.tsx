import { ExpertAvatarRequestBase } from "@/app/api/__generated__/models/expertAvatarRequestBase";
import { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";
import { ExpertAvatarRequestExpression } from "@/app/api/__generated__/models/expertAvatarRequestExpression";
import { ExpertAvatarRequestInlay } from "@/app/api/__generated__/models/expertAvatarRequestInlay";
import { ExpertAvatarRequestShape } from "@/app/api/__generated__/models/expertAvatarRequestShape";
import { ExpertAvatarRequestTilt } from "@/app/api/__generated__/models/expertAvatarRequestTilt";
import { Select } from "@/components/atoms/Select/Select";
import { getCategoryHex } from "../../ExpertAvatar/colors";

interface Props {
  category: ExpertAvatarRequestCategory;
  setCategory: (value: ExpertAvatarRequestCategory) => void;
  shape: ExpertAvatarRequestShape;
  setShape: (value: ExpertAvatarRequestShape) => void;
  base: ExpertAvatarRequestBase;
  setBase: (value: ExpertAvatarRequestBase) => void;
  tilt: ExpertAvatarRequestTilt;
  setTilt: (value: ExpertAvatarRequestTilt) => void;
  inlay: ExpertAvatarRequestInlay;
  setInlay: (value: ExpertAvatarRequestInlay) => void;
  expression: ExpertAvatarRequestExpression;
  setExpression: (value: ExpertAvatarRequestExpression) => void;
  isBusy: boolean;
}

/** What a generated candidate may vary. The category fixes the one material
 *  color; the cream section always sits on the lower form; everything else
 *  about the material and face is locked by the design system. */
export function GenerationOptions(props: Props) {
  return (
    <div className="grid w-full grid-cols-2 gap-3">
      <Choice
        label="Category"
        value={props.category}
        values={Object.values(ExpertAvatarRequestCategory)}
        onChange={props.setCategory}
        disabled={props.isBusy}
      />
      <Choice
        label="Head"
        value={props.shape}
        values={Object.values(ExpertAvatarRequestShape)}
        onChange={props.setShape}
        disabled={props.isBusy}
      />
      <Choice
        label="Base"
        value={props.base}
        values={Object.values(ExpertAvatarRequestBase)}
        onChange={props.setBase}
        disabled={props.isBusy}
      />
      <Choice
        label="Tilt"
        value={props.tilt}
        values={Object.values(ExpertAvatarRequestTilt)}
        onChange={props.setTilt}
        disabled={props.isBusy}
      />
      <Choice
        label="Cream section"
        value={props.inlay}
        values={Object.values(ExpertAvatarRequestInlay)}
        onChange={props.setInlay}
        disabled={props.isBusy}
      />
      <Choice
        label="Expression"
        value={props.expression}
        values={Object.values(ExpertAvatarRequestExpression)}
        onChange={props.setExpression}
        disabled={props.isBusy}
      />
    </div>
  );
}

function Choice<T extends string>({
  label,
  value,
  values,
  onChange,
  disabled,
}: {
  label: string;
  value: T;
  values: readonly T[];
  onChange: (value: T) => void;
  disabled: boolean;
}) {
  return (
    <Select
      id={`avatar-${label.toLowerCase().replaceAll(" ", "-")}`}
      label={label}
      value={value}
      disabled={disabled}
      options={values.map((value) => ({
        value,
        icon:
          label === "Category" ? (
            <span
              aria-hidden
              className="size-3 shrink-0 rounded-full border border-border"
              style={{ backgroundColor: getCategoryHex(value) }}
            />
          ) : undefined,
        label: value[0].toUpperCase() + value.slice(1),
      }))}
      onValueChange={(value) => {
        const choice = values.find((choice) => choice === value);
        if (choice) onChange(choice);
      }}
    />
  );
}
