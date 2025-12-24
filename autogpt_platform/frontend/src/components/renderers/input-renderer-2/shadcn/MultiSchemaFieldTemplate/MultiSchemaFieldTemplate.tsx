import { FormContextType, MultiSchemaFieldTemplateProps, RJSFSchema, StrictRJSFSchema } from '@rjsf/utils';
import { cn } from '../lib/utils';

export default function MultiSchemaFieldTemplate<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>({ selector, optionSchemaField }: MultiSchemaFieldTemplateProps<T, S, F>) {
  return (
    <div className={cn('p-4 border rounded-md bg-background shadow-sm')}>
      <div className={cn('mb-4')}>{selector}</div>
      {optionSchemaField}
    </div>
  );
}
