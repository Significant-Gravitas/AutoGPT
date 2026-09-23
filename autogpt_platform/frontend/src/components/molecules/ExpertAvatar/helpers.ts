import manifest from "../../../../public/autogpt-characters/manifest.json";

export function getManagedAvatar(avatarUrl: string | null, size: number) {
  for (const [assetID, identity] of Object.entries(manifest.identities)) {
    if (
      !Object.values(identity.files).some(
        (file) => `/${file.path.replace(/^public\//, "")}` === avatarUrl,
      )
    )
      continue;
    const pixels = manifest.logicalSizes.find((value) => value >= size) ?? 512;
    return {
      assetID,
      base: `${manifest.baseUrl}/${assetID}/neutral`,
      pixels,
    };
  }
  return null;
}
