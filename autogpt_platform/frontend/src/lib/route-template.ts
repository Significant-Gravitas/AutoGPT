const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const HEX_ID = /^[0-9a-f]{8,}$/i;

// An opaque ID (UUID, number, hash, nanoid) becomes {id} so every agent, run
// and session shares one route. A word or a short slug stays as it is.
function isOpaqueID(segment: string) {
  if (UUID.test(segment) || /^\d+$/.test(segment) || HEX_ID.test(segment)) {
    return true;
  }
  if (segment.includes("@") || segment.includes("%")) return true;
  return segment.length >= 12 && /\d/.test(segment);
}

export function routeTemplate(url: string) {
  const path = url.split(/[?#]/)[0] || "/";
  return path
    .split("/")
    .map((segment) => (segment && isOpaqueID(segment) ? "{id}" : segment))
    .join("/");
}
