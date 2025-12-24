import {
  ariaDescribedByIds,
  descriptionId,
  FormContextType,
  getTemplate,
  labelValue,
  RJSFSchema,
  schemaRequiresTrueValue,
  StrictRJSFSchema,
  WidgetProps,
} from '@rjsf/utils';
import { Checkbox } from '../components/ui/checkbox';
import { Label } from '../components/ui/label';

/** The `CheckBoxWidget` is a widget for rendering boolean properties.
 *  It is typically used to represent a boolean.
 *
 * @param props - The `WidgetProps` for this component
 */
export default function CheckboxWidget<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(props: WidgetProps<T, S, F>) {
  const {
    id,
    htmlName,
    value,
    disabled,
    readonly,
    label,
    hideLabel,
    schema,
    autofocus,
    options,
    onChange,
    onBlur,
    onFocus,
    registry,
    uiSchema,
    className,
  } = props;
  // Because an unchecked checkbox will cause html5 validation to fail, only add
  // the "required" attribute if the field value must be "true", due to the
  // "const" or "enum" keywords
  const required = schemaRequiresTrueValue<S>(schema);
  const DescriptionFieldTemplate = getTemplate<'DescriptionFieldTemplate', T, S, F>(
    'DescriptionFieldTemplate',
    registry,
    options,
  );

  const _onChange = (checked: boolean) => onChange(checked);
  const _onBlur = () => onBlur(id, value);
  const _onFocus = () => onFocus(id, value);

  const description = options.description || schema.description;
  return (
    <div
      className={`relative ${disabled || readonly ? 'cursor-not-allowed opacity-50' : ''}`}
      aria-describedby={ariaDescribedByIds(id)}
    >
      {!hideLabel && description && (
        <DescriptionFieldTemplate
          id={descriptionId(id)}
          description={description}
          schema={schema}
          uiSchema={uiSchema}
          registry={registry}
        />
      )}
      <div className='flex items-center gap-2 my-2'>
        <Checkbox
          id={id}
          name={htmlName || id}
          checked={typeof value === 'undefined' ? false : Boolean(value)}
          required={required}
          disabled={disabled || readonly}
          autoFocus={autofocus}
          onCheckedChange={_onChange}
          onBlur={_onBlur}
          onFocus={_onFocus}
          className={className}
        />
        <Label className='leading-tight' htmlFor={id}>
          {labelValue(label, hideLabel || !label)}
        </Label>
      </div>
    </div>
  );
}
