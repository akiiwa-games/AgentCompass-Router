"""Verified fast contracts, independent of a model's inferred capability scores.

Sources and billing semantics: ../references/fast-modes.md.
No family-name inference: a model is fast-capable only where its catalog row declares `fast_capable: true`, and only
on the harness of its `api_family` (the CLI whose fast launch setting is implemented here).
"""

# A fast-mode variant is its own candidate in a ranking and carries its base bundle's id plus this suffix, so a bundle id names
# exactly one variant: the plain bundle id the base one, `<id>+fast` the fast one.
ID_SUFFIX = '+fast'


def variant_id(bundle_id):
    return f'{bundle_id}{ID_SUFFIX}'


# Harnesses with an implemented fast launch setting (the Codex service tier): a CLI protocol fact. Claude Code's
# fastMode spends paid usage credits outside the subscription, which the product does not use (owner, 2026-11-24): its
# launches always pass it off.
HARNESSES = ('codex',)


def _row(model, model_info):
    if not isinstance(model_info, dict):
        raise ValueError(f'fast mode for {model!r} needs its catalog row (account_fleet.model_spec), not {model_info!r}')
    return model_info


def supported(harness, model, model_info):
    """`model_info` is the model's effective catalog row (account_fleet.model_spec)."""
    model_info = _row(model, model_info)
    return (harness in HARNESSES and model_info.get('fast_capable') is True
            and str(model_info.get('api_family') or '').strip().lower() == harness)


def refusal(harness, model, model_info):
    """Why `model` has no fast mode `harness` can launch (the one wording every stage uses), else None. A route named with
    --fast that has none launches in its plain mode with this reason (`not_applied`, owner 2026-11-24); a stored fast
    route (an older decision or worker) is refused with it. A missing price or speed datum is not a refusal: that is
    the economy's data error (candidate_eval)."""
    if supported(harness, model, model_info):
        return None
    return f'{model} has no verified fast mode on {harness}'


def not_applied(reason):
    """The note of a route named with --fast that runs in its plain mode, as if --fast had not been given."""
    return f'--fast was not applied: {reason}; the route runs in its plain mode.'


def price_gap(model, model_info):
    """The data error of a fast mode whose price or speed has no cited public source (the model's effective, overlaid
    catalog row), else None. Not a refusal: a named fast route launches with it (candidate_eval)."""
    fast_info = ((_row(model, model_info).get('five_axis') or {}).get('fast_mode')) or {}
    if fast_info.get('status') == 'ok':
        return None
    return (f"fast_mode has no cited public pricing/speed for {model} "
            f"({fast_info.get('reason') or fast_info.get('status') or 'missing'})")


def validate(harness, model, model_info):
    reason = refusal(harness, model, model_info)
    if reason:
        raise ValueError(reason)

