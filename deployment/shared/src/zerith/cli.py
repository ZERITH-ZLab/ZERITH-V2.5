"""Command-line tools for ZERITH 2.5 inference and robot control."""

import argparse
from importlib import import_module

from . import __version__

COMMANDS = {
    "serve": ("zerith.server", "Run the GPU inference server"),
    "robot": ("zerith.h1.runtime", "Check or run the H1 controller; offline by default"),
    "replay": ("zerith.replay", "Run inference on recorded observations"),
    "prepare": ("zerith.artifact", "Prepare a policy bundle from an exported artifact"),
    "doctor": ("zerith.doctor", "Check local configuration and dependency files without hardware"),
    "source-hash": ("zerith.source", "Fingerprint a reviewed external source package"),
}


def main(argv=None):
    """Dispatch to a component without importing its optional dependencies early."""
    parser = argparse.ArgumentParser(
        prog="zerith",
        description=__doc__,
        epilog="Run 'zerith COMMAND --help' for command options.",
    )
    parser.add_argument(
        "--version", action="version", version=f"ZERITH 2.5 Deployment {__version__}"
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, (_, description) in COMMANDS.items():
        commands.add_parser(name, help=description, add_help=False)
    args, remaining = parser.parse_known_args(argv)
    if args.command is None:
        if remaining:
            parser.error("unrecognized arguments: " + " ".join(remaining))
        parser.print_help()
        return
    module, _ = COMMANDS[args.command]
    return import_module(module).main(remaining, prog=f"{parser.prog} {args.command}")
