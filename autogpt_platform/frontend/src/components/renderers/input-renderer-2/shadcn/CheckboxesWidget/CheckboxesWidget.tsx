import {
  ariaDescribedByIds,
  enumOptionsDeselectValue,
  enumOptionsIsSelected,
  enumOptionsSelectValue,
  enumOptionsValueForIndex,
  FormContextType,
  optionId,
  RJSFSchema,
  StrictRJSFSchema,
  WidgetProps,
} from '@rjsf/utils';
import { FocusEvent } from 'react';

import { Checkbox } from '../components/ui/checkbox';
import { Label } from '../components/ui/label';
import { cn } from '../lib/utils';

/** The `CheckboxesWidget` is a widget for rendering checkbox groups.
 *  It is typically used to represent an array of enums.
 *
 * @param props - The `WidgetProps` for this component
 */
export default function CheckboxesWidget<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>({
  id,
  htmlName,
  disabled,
  options,
  value,
  autofocus,
  readonly,
  required,
  onChange,
  onBlur,
  onFocus,
  className,
}: WidgetProps<T, S, F>) {
  const { enumOptions, enumDisabled, inline, emptyValue } = options;
  const checkboxesValues = Array.isArray(value) ? value : [value];

  const _onBlur = ({ target }: FocusEvent<HTMLButtonElement>) =>
    onBlur(id, enumOptionsValueForIndex<S>(target && (target as any).value, enumOptions, emptyValue));
  const _onFocus = ({ target }: FocusEvent<HTMLButtonElement>) =>
    onFocus(id, enumOptionsValueForIndex<S>(target && (target as any).value, enumOptions, emptyValue));

  return (
    <div className={cn({ 'flex flex-col gap-2': !inline, 'flex flex-row gap-4 flex-wrap': inline })}>
      {Array.isArray(enumOptions) &&
        enumOptions.map((option, index: number) => {
          const checked = enumOptionsIsSelected<S>(option.value, checkboxesValues);
          const itemDisabled = Array.isArray(enumDisabled) && enumDisabled.indexOf(option.value) !== -1;
          const indexOptionId = optionId(id, index);

          return (
            <div className='flex items-center gap-2' key={indexOptionId}>
              <Checkbox
                id={indexOptionId}
                name={htmlName || id}
                required={required}
                disabled={disabled || itemDisabled || readonly}
                onCheckedChange={(state) => {
                  if (state) {
                    onChange(enumOptionsSelectValue<S>(index, checkboxesValues, enumOptions));
                  } else {
                    onChange(enumOptionsDeselectValue<S>(index, checkboxesValues, enumOptions));
                  }
                }}
                className={className}
                checked={checked}
                autoFocus={autofocus && index === 0}
                onBlur={_onBlur}
                onFocus={_onFocus}
                aria-describedby={ariaDescribedByIds(id)}
              />
              <Label className='leading-tight' htmlFor={optionId(id, index)}>
                {option.label}
              </Label>
            </div>
          );
        })}
    </div>
  );
}
