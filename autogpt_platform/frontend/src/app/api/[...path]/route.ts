import { NextRequest, NextResponse } from "next/server";

// TODO : Need to use proxyApiRequest or proxyFileUpload from PR https://github.com/Significant-Gravitas/AutoGPT/pull/10201
// If content type == "application/json" then use proxyApiRequest otherwise use proxyFileUpload

export async function GET(req: NextRequest) {
  console.log("GET Request : ", req);
  return NextResponse.json({ message: "GET request received successfully." });
}

export async function POST(req: NextRequest) {
  console.log("POST Request : ", req);
  return NextResponse.json({ message: "POST request received successfully." });
}

export async function PUT(req: NextRequest) {
  console.log("PUT Request : ", req);
  return NextResponse.json({ message: "PUT request received successfully." });
}

export async function DELETE(req: NextRequest) {
  console.log("DELETE Request : ", req);
  return NextResponse.json({
    message: "DELETE request received successfully.",
  });
}

export async function PATCH(req: NextRequest) {
  console.log("PATCH Request : ", req);
  return NextResponse.json({ message: "PATCH request received successfully." });
}
