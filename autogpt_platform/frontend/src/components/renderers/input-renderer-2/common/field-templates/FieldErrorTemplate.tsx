import { FieldErrorProps, errorId } from "@rjsf/utils";

export default function FieldErrorTemplate(props: FieldErrorProps) {
  const { errors = [], fieldPathId } = props;
  if (errors.length === 0) {
    return null;
  }
  const id = errorId(fieldPathId);

  return (
    <div className="flex flex-col gap-1" id={id}>
      {errors.map((error, i: number) => {
        return (
          <span className={"mb-1 text-xs font-medium text-destructive"} key={i}>
            {error}
          </span>
        );
      })}
    </div>
  );
}
