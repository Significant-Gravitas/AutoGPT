import { Select } from "@/components/atoms/Select/Select";
import { ExpertAvatarRequestShape } from "@/app/api/__generated__/models/expertAvatarRequestShape";
import { ExpertAvatarRequestExpression } from "@/app/api/__generated__/models/expertAvatarRequestExpression";
import type { ExpertAvatarRequestColor } from "@/app/api/__generated__/models/expertAvatarRequestColor";
import { ExpertAvatarRequestBase } from "@/app/api/__generated__/models/expertAvatarRequestBase";
import { ExpertAvatarRequestTilt } from "@/app/api/__generated__/models/expertAvatarRequestTilt";
import { ExpertAvatarRequestInlay } from "@/app/api/__generated__/models/expertAvatarRequestInlay";
import { EXPERT_AVATAR_COLORS } from "../../ExpertAvatar/helpers";

interface Props {
  shape: ExpertAvatarRequestShape;
  expression: ExpertAvatarRequestExpression;
  mineralColor: NonNullable<ExpertAvatarRequestColor>;
  base: ExpertAvatarRequestBase;
  tilt: ExpertAvatarRequestTilt;
  inlay: ExpertAvatarRequestInlay;
  isBusy: boolean;
  setShape: (value: ExpertAvatarRequestShape) => void;
  setExpression: (value: ExpertAvatarRequestExpression) => void;
  setMineralColor: (value: NonNullable<ExpertAvatarRequestColor>) => void;
  setBase: (value: ExpertAvatarRequestBase) => void;
  setTilt: (value: ExpertAvatarRequestTilt) => void;
  setInlay: (value: ExpertAvatarRequestInlay) => void;
}

export function GenerationOptions(props: Props) {
  return (
    <div className="grid w-full grid-cols-2 gap-3">
      <Choice
        label="Color"
        value={props.mineralColor}
        values={EXPERT_AVATAR_COLORS.map(
          (color) => color.id as NonNullable<ExpertAvatarRequestColor>,
        )}
        onChange={props.setMineralColor}
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
        label="Cream inlay"
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
          label === "Color" ? (
            <span
              aria-hidden
              className="size-3 shrink-0 rounded-full border border-border"
              style={{
                backgroundColor: EXPERT_AVATAR_COLORS.find(
                  (color) => color.id === value,
                )?.hex,
              }}
            />
          ) : undefined,
        label:
          label === "Color"
            ? (EXPERT_AVATAR_COLORS.find((color) => color.id === value)
                ?.label ?? value)
            : value[0].toUpperCase() + value.slice(1),
      }))}
      onValueChange={(value) => {
        const choice = values.find((choice) => choice === value);
        if (choice) onChange(choice);
      }}
    />
  );
}
