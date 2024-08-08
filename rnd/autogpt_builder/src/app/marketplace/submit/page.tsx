import React, { useState } from 'react';
import { useRouter } from 'next/router';
import { useForm, Controller } from 'react-hook-form';
import MarketplaceAPI from '@/lib/marketplace-api';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';

type FormData = {
  name: string;
  description: string;
  author: string;
  keywords: string[];
  categories: string[];
  graphFile: File | null;
};

const SubmitPage: React.FC = () => {
  const router = useRouter();
  const { control, handleSubmit, formState: { errors } } = useForm<FormData>();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const onSubmit = async (data: FormData) => {
    setIsSubmitting(true);
    setSubmitError(null);

    try {
      if (!data.graphFile) {
        throw new Error('Graph file is required');
      }

      const fileContent = await readFileAsJson(data.graphFile);

      const submission = {
        graph: {
          ...fileContent,
          name: data.name,
          description: data.description,
        },
        author: data.author,
        keywords: data.keywords,
        categories: data.categories,
      };

      const api = new MarketplaceAPI();
      await api.submitAgent(submission);

      router.push('/marketplace?submission=success');
    } catch (error) {
      console.error('Submission error:', error);
      setSubmitError(error instanceof Error ? error.message : 'An unknown error occurred');
    } finally {
      setIsSubmitting(false);
    }
  };

  const readFileAsJson = (file: File): Promise<any> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const json = JSON.parse(event.target?.result as string);
          resolve(json);
        } catch (error) {
          reject(new Error('Invalid JSON file'));
        }
      };
      reader.onerror = (error) => reject(error);
      reader.readAsText(file);
    });
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">Submit Your Agent</h1>
      <Card className="p-6">
        <form onSubmit={handleSubmit(onSubmit)}>
          <div className="space-y-4">
            <Controller
              name="name"
              control={control}
              rules={{ required: 'Name is required' }}
              render={({ field }) => (
                <Input
                  label="Agent Name"
                  placeholder="Enter your agent's name"
                  error={errors.name?.message}
                  {...field}
                />
              )}
            />

            <Controller
              name="description"
              control={control}
              rules={{ required: 'Description is required' }}
              render={({ field }) => (
                <TextArea
                  label="Description"
                  placeholder="Describe your agent"
                  error={errors.description?.message}
                  {...field}
                />
              )}
            />

            <Controller
              name="author"
              control={control}
              rules={{ required: 'Author is required' }}
              render={({ field }) => (
                <Input
                  label="Author"
                  placeholder="Your name or username"
                  error={errors.author?.message}
                  {...field}
                />
              )}
            />

            <Controller
              name="keywords"
              control={control}
              rules={{ required: 'At least one keyword is required' }}
              render={({ field }) => (
                <MultiSelect
                  label="Keywords"
                  placeholder="Add keywords"
                  error={errors.keywords?.message}
                  {...field}
                />
              )}
            />

            <Controller
              name="categories"
              control={control}
              rules={{ required: 'At least one category is required' }}
              render={({ field }) => (
                <MultiSelect
                  label="Categories"
                  placeholder="Select categories"
                  options={['Productivity', 'Entertainment', 'Education', 'Business', 'Other']}
                  error={errors.categories?.message}
                  {...field}
                />
              )}
            />

            <Controller
              name="graphFile"
              control={control}
              rules={{ required: 'Graph file is required' }}
              render={({ field: { onChange, value, ...field } }) => (
                <Input
                  type="file"
                  label="Graph File (JSON)"
                  accept=".json"
                  error={errors.graphFile?.message}
                  onChange={(e) => onChange(e.target.files?.[0] || null)}
                  {...field}
                />
              )}
            />

            {submitError && (
              <Alert variant="destructive">
                <AlertTitle>Submission Failed</AlertTitle>
                <AlertDescription>{submitError}</AlertDescription>
              </Alert>
            )}

            <Button type="submit" className="w-full" disabled={isSubmitting}>
              {isSubmitting ? 'Submitting...' : 'Submit Agent'}
            </Button>
          </div>
        </form>
      </Card>
    </div>
  );
};

export default SubmitPage;