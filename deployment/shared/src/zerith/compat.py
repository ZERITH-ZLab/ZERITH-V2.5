"""Read historical identifiers without rewriting source metadata or hashes."""

LEGACY_ARCHITECTURE = "fam-wam-action20-linear-g4d-teacher-v5.2"
LEGACY_POLICY_FORMAT = "fam-wam-policy-v5.2-action20-linear-g4d-teacher"


def canonical_identifier(value):
    if not isinstance(value, str):
        return value
    aliases = {
        LEGACY_ARCHITECTURE: "zerith-action20-linear-g4d-teacher",
        LEGACY_POLICY_FORMAT: "zerith-policy-action20-linear-g4d-teacher",
        "fam-wam-v5-cfs-asset-receipt-v1": "zerith-asset-receipt-v1",
        "fam-wam-v5.2-node-local-assets-v1": "zerith-node-local-assets-v1",
        "mcap_to_fam_wam": "mcap_to_zerith",
    }
    if value in aliases:
        return aliases[value]
    for old, new in (
        ("zerith-va-", "zerith-"),
        ("zvla-", "zerith-"),
        ("zvla_", "zerith_"),
        ("zwam-", "zerith-"),
        ("zwam_", "zerith_"),
        ("fam-wam-", "zerith-"),
        ("fam_wam_raw/", "zerith/"),
        ("fam_wam_raw_", "zerith_raw_"),
    ):
        if value.startswith(old):
            return new + value[len(old) :]
    return value


def same_identifier(actual, expected):
    return canonical_identifier(actual) == canonical_identifier(expected)
