require("dotenv").config();
require("@shopify/shopify-api/adapters/node");
const { shopifyApi, LATEST_API_VERSION } = require("@shopify/shopify-api");
const express = require("express");
const cors = require("cors");
const { json } = require("body-parser");

const SHOPIFY_API_KEY =
  process.env.SHOPIFY_API_KEY || "4dc8f496291746f5f3b4b822b06ee5dd";
const SHOPIFY_SECRET_KEY =
  process.env.SHOPIFY_SECRET_KEY || "749be039f3afec53d0bbbe020f2934dc";
const scopes = [
  "read_products",
  "write_products",
  "read_locations",
  "write_locations",
  "read_orders",
  "write_orders",
];
const PORT = process.env.PORT || 8021;
const hostName =
  process.env.SHOPIFY_INTEGRATION_HOSTNAME || `localhost:${PORT}`;
const callbackPath =
  process.env.SHOPIFY_INTEGRATION_CALLBACK_APTH || "/oauth/callback";

const shopify = shopifyApi({
  apiKey: SHOPIFY_API_KEY,
  apiSecretKey: SHOPIFY_SECRET_KEY,
  scopes: scopes,
  hostName,
  apiVersion: LATEST_API_VERSION,
  isEmbeddedApp: false,
  hostScheme: hostName.includes("localhost") ? "http" : "https",
});

const app = express();

app.use(cors());
app.use(json());

app.get("/", (req, res) => {
  res.json({ ts: new Date().toISOString() });
});

app.get("/oauth", async (req, res) => {
  const shopName = req.query.shop;
  if (!shopName) {
    res.status(400).send("shop name is require");
    return;
  }
  const shop = shopify.utils.sanitizeShop(`${shopName}.myshopify.com`, true);
  if (!shop) {
    res.status(400).send("invalid shop");
    return;
  }

  await shopify.auth.begin({
    shop,
    callbackPath,
    isOnline: false,
    rawRequest: req,
    rawResponse: res,
  });
});

app.get("/oauth/callback", async (req, res) => {
  // The library will automatically set the appropriate HTTP headers
  const callback = await shopify.auth.callback({
    rawRequest: req,
    rawResponse: res,
  });

  res.json(callback.session.toObject());
});

app.listen(PORT, () => {
  console.log(`:${PORT}`);
});
