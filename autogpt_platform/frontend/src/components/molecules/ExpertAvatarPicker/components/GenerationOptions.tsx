import { ExpertAvatarRequestAccentPlacement } from "@/app/api/__generated__/models/expertAvatarRequestAccentPlacement";
import { ExpertAvatarRequestAccentCount } from "@/app/api/__generated__/models/expertAvatarRequestAccentCount";
import { Select } from "@/components/atoms/Select/Select";
import { ExpertAvatarRequestShape } from "@/app/api/__generated__/models/expertAvatarRequestShape";
import { ExpertAvatarRequestExpression } from "@/app/api/__generated__/models/expertAvatarRequestExpression";
import type { ExpertAvatarRequestShade } from "@/app/api/__generated__/models/expertAvatarRequestShade";
import { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";
import { ExpertAvatarRequestBase } from "@/app/api/__generated__/models/expertAvatarRequestBase";
import { ExpertAvatarRequestTilt } from "@/app/api/__generated__/models/expertAvatarRequestTilt";
import { ExpertAvatarRequestInlay } from "@/app/api/__generated__/models/expertAvatarRequestInlay";
import { EXPERT_AVATARS } from "../../ExpertAvatar/helpers";

interface Props {
  shape: ExpertAvatarRequestShape;
  expression: ExpertAvatarRequestExpression;
  category: ExpertAvatarRequestCategory;
  setCategory: (value: ExpertAvatarRequestCategory) => void;
  shade: NonNullable<ExpertAvatarRequestShade>;
  base: ExpertAvatarRequestBase;
  tilt: ExpertAvatarRequestTilt;
  inlay: ExpertAvatarRequestInlay;
  accentPlacement: ExpertAvatarRequestAccentPlacement;
  accentCount: ExpertAvatarRequestAccentCount;
  setAccentPlacement: (value: ExpertAvatarRequestAccentPlacement) => void;
  setAccentCount: (value: ExpertAvatarRequestAccentCount) => void;
  isBusy: boolean;
  setShape: (value: ExpertAvatarRequestShape) => void;
  setExpression: (value: ExpertAvatarRequestExpression) => void;
  setShade: (value: NonNullable<ExpertAvatarRequestShade>) => void;
  setBase: (value: ExpertAvatarRequestBase) => void;
  setTilt: (value: ExpertAvatarRequestTilt) => void;
  setInlay: (value: ExpertAvatarRequestInlay) => void;
}

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
        label="Shade"
        value={props.shade}
        values={["standard", "light", "dark"]}
        onChange={props.setShade}
        disabled={props.isBusy}
      />
      <Choice
        label="Shape"
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
        label="Accent shape"
        value={props.inlay}
        values={Object.values(ExpertAvatarRequestInlay)}
        onChange={props.setInlay}
        disabled={props.isBusy}
      />
      <Choice
        label="Accent placement"
        value={props.accentPlacement}
        values={Object.values(ExpertAvatarRequestAccentPlacement)}
        onChange={props.setAccentPlacement}
        disabled={props.isBusy}
      />
      <Choice
        label="Accents per part"
        value={props.accentCount}
        values={Object.values(ExpertAvatarRequestAccentCount)}
        onChange={props.setAccentCount}
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
              style={{
                backgroundColor: EXPERT_AVATARS.find(
                  (color) => color.id === value,
                )?.hex,
              }}
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
