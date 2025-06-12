# Loading States Implementation Guide

This document outlines the loading state patterns used throughout the AutoGPT Platform frontend and provides guidance on implementing loading states for async operations.

## Overview

Since making functions like `getServerSupabase()` async, we need proper loading states throughout the application to provide user feedback during async operations. The project uses several patterns for loading states:

## Available Loading Components

### 1. Core Loading Components
- **`LoadingBox`** - Full-page loading with centered spinner
- **`LoadingSpinner`** - Reusable spinning icon component
- **`LoadingButton`** - Button with built-in loading state (NEW)

```tsx
// LoadingBox usage
<LoadingBox className="h-[80vh]" />

// LoadingSpinner usage
<LoadingSpinner className="h-4 w-4" />

// LoadingButton usage
<LoadingButton loading={isLoading} loadingText="Saving...">
  Save Changes
</LoadingButton>
```

### 2. Page-Level Loading
Pages have dedicated `loading.tsx` files for route-level loading:
- `app/(platform)/profile/(user)/settings/loading.tsx`
- `app/(platform)/monitoring/loading.tsx`
- `app/(platform)/library/agents/[id]/loading.tsx`

### 3. Skeleton Loading
Used for content placeholders while data loads:
```tsx
import { Skeleton } from "@/components/ui/skeleton";
<Skeleton className="h-4 w-32" />
```

## Implementation Patterns

### 1. Form Loading States

#### React Hook Form Pattern
```tsx
const form = useForm();

// Built-in loading state
<Button 
  disabled={form.formState.isSubmitting}
  type="submit"
>
  {form.formState.isSubmitting ? "Saving..." : "Save"}
</Button>
```

#### Custom Loading State Pattern
```tsx
const [isLoading, setIsLoading] = useState(false);

const handleSubmit = async (data) => {
  setIsLoading(true);
  try {
    await asyncOperation(data);
  } finally {
    setIsLoading(false);
  }
};

<LoadingButton loading={isLoading} loadingText="Processing...">
  Submit
</LoadingButton>
```

### 2. Authentication Loading

#### AuthButton Component
```tsx
<AuthButton
  onClick={handleLogin}
  isLoading={isLoading}
  disabled={disabled}
>
  Log In
</AuthButton>
```

#### Page-Level Auth Loading
```tsx
if (isUserLoading || user) {
  return <LoadingBox className="h-[80vh]" />;
}
```

### 3. Server Component Loading

For server components using async functions like `getServerUser()`:
- Create `loading.tsx` files in the same directory
- Use Suspense boundaries where appropriate
- Server components will wait for async operations to complete

### 4. Client Component Async Operations

For client components making API calls:
```tsx
const [isLoading, setIsLoading] = useState(false);

const handleAction = async () => {
  setIsLoading(true);
  try {
    await api.someOperation();
  } catch (error) {
    // Handle error
  } finally {
    setIsLoading(false);
  }
};
```

## Recently Updated Components

### 1. Admin Marketplace Components
- **Approve/Reject Buttons** - Added loading states for approval/rejection operations
- **Add Money Button** - Added loading state for adding credits

### 2. Agent Import Form
- Added loading state for the import and creation process
- Disabled form fields during import

### 3. Profile Settings
- Already has proper loading states using `form.formState.isSubmitting`

## Best Practices

### 1. User Feedback
- Always provide visual feedback for operations > 200ms
- Use descriptive loading text ("Saving...", "Importing...", etc.)
- Disable interactive elements during loading

### 2. Error Handling
```tsx
const [isLoading, setIsLoading] = useState(false);
const [error, setError] = useState(null);

const handleAction = async () => {
  setIsLoading(true);
  setError(null);
  try {
    await asyncOperation();
  } catch (err) {
    setError(err.message);
  } finally {
    setIsLoading(false);
  }
};
```

### 3. Form Validation
```tsx
// Prevent submission while loading
<Button 
  disabled={isLoading || !form.formState.isValid}
  type="submit"
>
  {isLoading ? "Processing..." : "Submit"}
</Button>
```

### 4. Multiple States
For complex operations, consider multiple loading states:
```tsx
const [isApproving, setIsApproving] = useState(false);
const [isRejecting, setIsRejecting] = useState(false);

// Disable both buttons when either operation is running
disabled={isApproving || isRejecting}
```

## Where Loading States Are Needed

### ✅ Already Implemented
- Login/Signup forms
- Settings forms
- Library upload dialog
- Wallet refill components
- Admin approval/rejection buttons
- Agent import form

### 🔄 Server Components (Handled by loading.tsx)
- Settings page (`loading.tsx` exists)
- Monitoring page (`loading.tsx` exists)
- Profile pages
- Marketplace pages

### ✅ Authentication Flow
- Login page (has loading states)
- Signup page (has loading states)
- Reset password page (has loading states)

## Component Examples

### Simple Loading Button
```tsx
import { LoadingButton } from "@/components/ui/loading-button";

<LoadingButton 
  loading={isLoading} 
  loadingText="Saving..."
  onClick={handleSave}
>
  Save Changes
</LoadingButton>
```

### Form with Loading State
```tsx
const [isSubmitting, setIsSubmitting] = useState(false);

<form onSubmit={async (e) => {
  e.preventDefault();
  setIsSubmitting(true);
  try {
    await submitForm();
  } finally {
    setIsSubmitting(false);
  }
}}>
  <input disabled={isSubmitting} />
  <LoadingButton loading={isSubmitting} type="submit">
    Submit
  </LoadingButton>
</form>
```

### Dialog with Loading State
```tsx
const [isOpen, setIsOpen] = useState(false);
const [isLoading, setIsLoading] = useState(false);

<Dialog open={isOpen} onOpenChange={setIsOpen}>
  <DialogContent>
    <form action={async (formData) => {
      setIsLoading(true);
      try {
        await handleSubmit(formData);
        setIsOpen(false);
      } finally {
        setIsLoading(false);
      }
    }}>
      <input disabled={isLoading} />
      <DialogFooter>
        <Button variant="outline" disabled={isLoading}>
          Cancel
        </Button>
        <LoadingButton loading={isLoading} type="submit">
          Confirm
        </LoadingButton>
      </DialogFooter>
    </form>
  </DialogContent>
</Dialog>
```

## Migration Checklist

When adding loading states to existing components:

1. **Identify async operations** - Look for API calls, form submissions, redirects
2. **Add loading state** - Use `useState` for loading boolean
3. **Update UI** - Show loading indicators, disable inputs
4. **Handle errors** - Add error states where appropriate
5. **Test edge cases** - Rapid clicks, network failures, etc.

## Summary

The project now has comprehensive loading state patterns that provide good user feedback for async operations. The key areas that needed loading states after making authentication functions async have been addressed:

- ✅ Form submissions
- ✅ Admin operations  
- ✅ Authentication flows
- ✅ File uploads/imports
- ✅ Server-side rendering (via loading.tsx files)

All critical user interactions now have appropriate loading feedback, making the application feel more responsive and professional. 