'use server'
import { revalidatePath } from 'next/cache'
import { redirect } from 'next/navigation'
import { createServerClient } from '@/lib/supabase/server'
import { z } from 'zod'

const loginFormSchema = z.object({
  email: z.string().email().min(2).max(64),
  password: z.string().min(6).max(64),
})

export async function login(values: z.infer<typeof loginFormSchema>) {
  const supabase = createServerClient()

  if (!supabase) {
    redirect('/error')
  }

  // We are sure that the values are of the correct type because zod validates the form
  const { error } = await supabase.auth.signInWithPassword(values)


  if (error) {
    console.log('error', error)
    redirect('/error')
  }

  revalidatePath('/', 'layout')
  redirect('/')
}

export async function signup(values: z.infer<typeof loginFormSchema>) {
  const supabase = createServerClient()

  if (!supabase) {
    redirect('/error')
  }

  // We are sure that the values are of the correct type because zod validates the form
  const { error } = await supabase.auth.signUp(values)

  if (error) {
    console.log('error', error)
    redirect('/error')
  }

  revalidatePath('/', 'layout')
  redirect('/')
}
