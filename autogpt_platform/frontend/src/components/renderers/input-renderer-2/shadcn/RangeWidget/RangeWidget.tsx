import { ariaDescribedByIds, FormContextType, rangeSpec, RJSFSchema, StrictRJSFSchema, WidgetProps } from '@rjsf/utils';
import _pick from 'lodash/pick';

import { Slider } from '../components/ui/slider';

const allowedProps = [
  'name',
  'min',
  'max',
  'step',
  'orientation',
  'disabled',
  'defaultValue',
  'value',
  'onValueChange',
  'className',
  'dir',
  'inverted',
  'minStepsBetweenThumbs',
];

/**
 * A range widget component that renders a slider for number input
 * @param {object} props - The widget properties
 * @param {number} props.value - The current value of the range
 * @param {boolean} props.readonly - Whether the widget is read-only
 * @param {boolean} props.disabled - Whether the widget is disabled
 * @param {object} props.options - Additional options for the widget
 * @param props.schema - The JSON schema for this field
 * @param {(value: any) => void} props.onChange - Callback for when the value changes
 * @param {string} props.label - The label for the range input
 * @param {string} props.id - The unique identifier for the widget
 * @returns {JSX.Element} The rendered range widget
 */
export default function RangeWidget<T = any, S extends StrictRJSFSchema = RJSFSchema, F extends FormContextType = any>({
  value,
  readonly,
  disabled,
  options,
  schema,
  onChange,
  label,
  id,
}: WidgetProps<T, S, F>): JSX.Element {
  const _onChange = (value: number[]) => onChange(value[0]);

  const sliderProps = { value, label, id, ...rangeSpec<S>(schema) };
  const uiProps = { id, ..._pick((options.props as object) || {}, allowedProps) };
  return (
    <>
      <Slider
        disabled={disabled || readonly}
        min={sliderProps.min}
        max={sliderProps.max}
        step={sliderProps.step}
        value={[value as number]}
        onValueChange={_onChange}
        {...uiProps}
        aria-describedby={ariaDescribedByIds(id)}
      />
      {value}
    </>
  );
}
