"""Exact base-instruction matching for released LIBERO-Plus prompt adapters."""

import re


def libero_plus_prompt(prompt, base_instructions, suffix_regex):
    for base in sorted(base_instructions, key=len, reverse=True):
        if prompt == base:
            return base
        if prompt.startswith(base + " ") and re.fullmatch(suffix_regex, prompt[len(base) :].strip()):
            return base
    # Keep unknown language variants; never guess a task from a partial match.
    return prompt
