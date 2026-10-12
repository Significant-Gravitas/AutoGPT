"""The choices a brand-constrained avatar candidate may vary, in the words the
design system uses (expert-design-system: 01 Construction, generation-constraints).

Everything else is locked: two touching primary masses, one category color over
both, cream only on the lower form, the shared gentle face, low-sheen clay and
the warm studio tile.
"""

from typing import Literal

AvatarShape = Literal[
    "pebble",
    "slab",
    "wedge",
    "dome",
    "capsule",
    "bean",
    "saddle",
    "kite",
    "fan",
    "arch",
    "shield",
    "crescent",
]

# Each head is a rounded sculptural volume with depth, never a flat cut-out.
SHAPES: dict[AvatarShape, str] = {
    "pebble": "broad weathered river pebble, uneven gentle slopes, deep rounded sides",
    "slab": "wide low rounded slab, nearly twice as wide as tall, thick soft edges",
    "wedge": "tall soft asymmetric pyramid with a blunt rounded apex and receding sides",
    "dome": "low asymmetric dome, arched crown, rounded flat underside",
    "capsule": "upright rounded capsule, softly flattened ends, generous depth",
    "bean": "upright kidney bean with one smooth shallow inward curve",
    "saddle": "wide cushion with a shallow rounded dip in its top edge; blunt shoulders, no ears",
    "kite": "rounded rhombohedral stone, four very blunt corners, wide middle, visible depth",
    "fan": "soft fan flaring to a convex top and narrowing to a blunt bottom",
    "arch": "upright arch with one high domed shoulder and one lower shoulder",
    "shield": "rounded shield, broad soft shoulders tapering to a blunt bottom",
    "crescent": "wide scooped stone, shallow rounded concavity on top, blunt rounded ends; no horns",
}
BASES = {
    "compact": "compact rounded base, a little narrower than the head, stable flat footprint",
    "wide": "wide low rounded base spreading past the head, stable flat footprint",
    "tall": "taller rounded tapered base; the head is about 55 percent of the figure's height",
}
TILTS = {
    "level": "head upright and nearly level, with natural asymmetry",
    "left": "head tilted gently left by about ten degrees, still touching the base",
    "right": "head tilted gently right by about ten degrees, still touching the base",
}
# The one cream section, always on the lower form (generation-constraints.md).
AvatarInlay = Literal["sweep", "field", "cloud", "bank", "pool", "inlet", "wrap"]
INLAYS: dict[AvatarInlay, str] = {
    "sweep": "one broad flowing cream stroke curving across the lower form",
    "field": "one rounded cream side field on the lower form",
    "cloud": "one cloud-like cream field with a few broad rounded swells on the lower form",
    "bank": "one low rolling cream bank along the bottom of the lower form",
    "pool": "one rounded cream edge pool on the lower form",
    "inlet": "one rounded cream inlet rising a little way into the lower form",
    "wrap": "one curved cream corner wrap on the lower form",
}
EXPRESSIONS = {
    "friendly": "small oval eyes, relaxed brows, small closed smile",
    "curious": "small oval eyes, one brow slightly raised, small closed mouth",
    "focused": "small oval eyes, brows gently lowered, small closed mouth; calm, not angry",
    "pleased": "small softly closed eyes, relaxed brows, small closed smile",
}
