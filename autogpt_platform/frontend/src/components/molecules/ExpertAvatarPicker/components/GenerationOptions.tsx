import { Select } from "@/components/atoms/Select/Select";
import { ExpertAvatarRequestShape } from "@/app/api/__generated__/models/expertAvatarRequestShape";
import { ExpertAvatarRequestExpression } from "@/app/api/__generated__/models/expertAvatarRequestExpression";

interface Props {
  shape: ExpertAvatarRequestShape;
  expression: ExpertAvatarRequestExpression;
  isBusy: boolean;
  setShape: (shape: ExpertAvatarRequestShape) => void;
  setExpression: (expression: ExpertAvatarRequestExpression) => void;
}

export function GenerationOptions(options: Props) {
  return (
    <div className="grid w-full grid-cols-2 gap-3">
      <Select
        id="avatar-shape"
        label="Shape"
        value={options.shape}
        disabled={options.isBusy}
        options={Object.values(ExpertAvatarRequestShape).map((value) => ({
          value,
          label: value[0].toUpperCase() + value.slice(1),
        }))}
        onValueChange={(value) => {
          const shape = Object.values(ExpertAvatarRequestShape).find(
            (shape) => shape === value,
          );
          if (shape) options.setShape(shape);
        }}
      />
      <Select
        id="avatar-expression"
        label="Expression"
        value={options.expression}
        disabled={options.isBusy}
        options={Object.values(ExpertAvatarRequestExpression).map((value) => ({
          value,
          label: value[0].toUpperCase() + value.slice(1),
        }))}
        onValueChange={(value) => {
          const expression = Object.values(ExpertAvatarRequestExpression).find(
            (expression) => expression === value,
          );
          if (expression) options.setExpression(expression);
        }}
      />
    </div>
  );
}
