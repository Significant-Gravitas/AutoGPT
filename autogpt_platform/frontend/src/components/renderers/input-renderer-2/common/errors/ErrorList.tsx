import { ErrorListProps, FormContextType, RJSFSchema, StrictRJSFSchema, TranslatableString } from '@rjsf/utils';
import { AlertCircle } from 'lucide-react';

import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';

/** The `ErrorList` component is the template that renders the all the errors associated with the fields in the `Form`
 *
 * @param props - The `ErrorListProps` for this component
 */
export default function ErrorList<T = any, S extends StrictRJSFSchema = RJSFSchema, F extends FormContextType = any>({
  errors,
  registry,
}: ErrorListProps<T, S, F>) {
  const { translateString } = registry;
  return (
    <Alert variant='destructive' className='mb-2'>
      <AlertCircle className='h-4 w-4' />
      <AlertTitle>{translateString(TranslatableString.ErrorsLabel)}</AlertTitle>
      <AlertDescription className='flex flex-col gap-1'>
        {errors.map((error, i: number) => {
          return <span key={i}>&#x2022; {error.stack}</span>;
        })}
      </AlertDescription>
    </Alert>
  );
}
