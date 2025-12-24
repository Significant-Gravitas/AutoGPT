import { GridTemplateProps } from "@rjsf/utils";
import { cn } from "../lib/utils";

/** Renders a `GridTemplate` for mui, which is expecting the column sizing information coming in via the
 * extra props provided by the caller, which are spread directly on the `Grid2`.
 *
 * @param props - The GridTemplateProps, including the extra props containing the mui grid positioning details
 */
export default function GridTemplate(props: GridTemplateProps) {
  const { children, column, className, ...rest } = props;
  return (
    <div className={cn("grid gap-4", className)} {...rest}>
      {children}
    </div>
  );
}
