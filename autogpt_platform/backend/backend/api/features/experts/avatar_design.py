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

SHAPES = {
    "pebble": "broad irregular rounded pebble head with uneven gentle slopes",
    "slab": "very wide low rounded rectangular slab head, nearly twice as wide as tall",
    "wedge": "tall sloping wedge head with very blunt rounded corners",
    "dome": "low asymmetric dome head with an arched crown and rounded flat bottom",
    "capsule": "tall narrow upright capsule head with softly flattened rounded ends",
    "bean": "upright kidney bean head with a deep smooth inward curve on one side",
    "saddle": "wide cushion head with a smooth shallow dip in its top edge; no ears",
    "kite": "rounded diamond head with four extremely blunt corners and a wide middle",
    "fan": "broad fan head flaring toward a convex top and narrowing at the bottom",
    "arch": "upright arch head with one high domed shoulder and one lower shoulder",
    "shield": "rounded shield head with broad shoulders tapering to a blunt bottom",
    "crescent": "wide crescent pebble head, shallow concave top, convex bottom, blunt ends; no horns",
}
BASES = {
    "compact": "compact rounded base, narrower than the head, stable flat footprint",
    "wide": "wide low oval base spreading beyond the head contact point, stable flat footprint",
    "tall": "taller narrow rounded tapered base, head about 55 percent of total figure height",
}
TILTS = {
    "level": "head upright and nearly level, with natural asymmetry",
    "left": "head tilted gently left about ten degrees while touching the base",
    "right": "head tilted gently right about ten degrees while touching the base",
}
INLAYS = {
    "sweep": "broad flowing S-shaped cream sweep with rounded ends",
    "pool": "rounded cream pool with an uneven organic boundary",
    "curl": "broad cream curl, no thin piping",
    "patch": "rounded irregular patch, like a smooth pebble inset",
    "cap": "rounded cream cap following the outer top edge",
    "teardrop": "soft cream teardrop with a blunt rounded tip",
}
ACCENT_PLACEMENTS = {
    "body": "BODY ONLY; keep the head entirely main color",
    "head": "HEAD ONLY; keep the body entirely main color",
    "both": "BOTH head and body; distribute accents across both parts",
}
