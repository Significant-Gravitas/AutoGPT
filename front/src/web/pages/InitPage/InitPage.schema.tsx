import * as yup from "yup"

export const schema = yup.object().shape({
  ai_name: yup.string().required(),
  ai_role: yup.string().required(),
  ai_goals: yup.array().of(yup.string()).min(1).max(5),
  continuous: yup.boolean().default(false),
})
