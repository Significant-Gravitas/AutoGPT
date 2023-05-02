import { buffer } from "micro";
import Cors from "micro-cors";
import type { NextApiRequest, NextApiResponse } from "next";

import Stripe from "stripe";
import { env } from "../../../env/server.mjs";
import { prisma } from "../../../server/db";
import { getCustomerEmail } from "../../../utils/stripe-utils";

const stripe = new Stripe(env.STRIPE_SECRET_KEY ?? "", {
  apiVersion: "2022-11-15",
});

const webhookSecret = env.STRIPE_WEBHOOK_SECRET ?? "";

// Stripe requires the raw body to construct the event.
export const config = {
  api: {
    bodyParser: false,
  },
};

const cors = Cors({
  allowMethods: ["POST", "HEAD"],
});

function success(res: NextApiResponse) {
  res.status(200).json({ received: true });
}

const webhookHandler = async (req: NextApiRequest, res: NextApiResponse) => {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    res.status(405).end("Method Not Allowed");
    return;
  }

  const buf = await buffer(req);
  const sig = req.headers["stripe-signature"]!;

  let event: Stripe.Event;

  try {
    event = stripe.webhooks.constructEvent(buf.toString(), sig, webhookSecret);
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : "Unknown error";
    if (err instanceof Error) console.log(err);
    console.log(`❌ Error message: ${errorMessage}`);
    res.status(400).send(`Webhook Error: ${errorMessage}`);
    return;
  }

  if (!event.type.startsWith("customer.subscription")) {
    success(res);
    return;
  }

  const subscription = event.data.object as Stripe.Subscription;
  const email = await getCustomerEmail(stripe, subscription.customer);
  const user = await prisma.user.findUniqueOrThrow({
    where: {
      email: email,
    },
  });

  // Handle the event
  switch (event.type) {
    case "customer.subscription.deleted":
    case "customer.subscription.paused":
    case "customer.subscription.updated":
    case "customer.subscription.resumed":
      await updateUserSubscription(user.id, subscription);
      break;
    default:
      console.log(`Unhandled event type ${event.type}.`);
  }

  success(res);
};

const updateUserSubscription = async (
  userId: string,
  subscription: Stripe.Subscription
) => {
  const subscriptionId =
    subscription.status === "active" || subscription.status === "trialing"
      ? subscription.id
      : undefined;

  await prisma.user.update({
    where: {
      id: userId,
    },
    data: {
      subscriptionId,
    },
  });
};

// eslint-disable-next-line @typescript-eslint/no-unsafe-call
export default cors(webhookHandler as any);
