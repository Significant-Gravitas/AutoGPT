import type Stripe from "stripe";

export const getCustomerId = (
  customer: string | Stripe.Customer | Stripe.DeletedCustomer | null
) => {
  if (!customer) throw new Error("No customer found");

  switch (typeof customer) {
    case "string":
      return customer;
    case "object":
      return customer.id;
    default:
      throw new Error("Unexpected customer type");
  }
};

export const getCustomerEmail = async (
  stripe: Stripe,
  customer: string | Stripe.Customer | Stripe.DeletedCustomer | null
) => {
  if (!customer) throw new Error("No customer found");

  let c = customer;
  if (typeof customer === "string") {
    c = await stripe.customers.retrieve(customer);
  }

  return (c as Stripe.Customer).email ?? "";
};
