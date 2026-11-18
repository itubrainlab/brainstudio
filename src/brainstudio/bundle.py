"""Copy brAInStudio functions, with everything they use, into a stand-alone script.

Exported scripts (:mod:`brainstudio.script`) must run without brAInStudio installed
and still compute exactly what the app computes. This module reads the package's
source, follows every function, class and constant a script uses through to its own
dependencies, and returns their code unchanged — except that module prefixes are
dropped (``E.trigger_codes`` → ``trigger_codes``) since everything lives in one file,
and the few names defined in two modules get the module's name in front.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

PKG = "brainstudio"
ROOT = Path(__file__).parent
#: the modules scripts may use (not the app, the demo or the script writer itself)
MODULES = ("model", "stats", "events", "gaze", "eeg", "preprocessing", "epoching",
           "bandpower", "workflow", "io", "io.markers", "io.multi", "io.bids", "io.xdf")


@dataclass
class _Def:
    module: str
    name: str
    node: ast.AST
    index: int
    deps: set = field(default_factory=set)        # {(module, name)}
    externals: set = field(default_factory=set)   # external names used
    aliases: dict = field(default_factory=dict)   # local alias → ("mod", m) | ("obj", m, n)
    drop_lines: set = field(default_factory=set)  # line numbers of relative imports


class _Module:
    def __init__(self, name: str):
        self.name = name
        path = ROOT / (name.replace(".", "/") + ".py")
        if not path.exists():
            path = ROOT / name.replace(".", "/") / "__init__.py"
        self.is_pkg = path.name == "__init__.py"
        self.source = path.read_text()
        self.lines = self.source.splitlines()
        self.tree = ast.parse(self.source)
        self.aliases: dict = {}     # name → ("mod", module) | ("obj", module, name)
        self.externals: dict = {}   # bound name → import statement source
        self.defs: dict[str, _Def] = {}
        self._scan()

    # ------------------------------------------------------------------ parsing
    def resolve_relative(self, node: ast.ImportFrom) -> str | None:
        """Module a relative (or brainstudio-absolute) import refers to, else None."""
        if node.level == 0:
            mod = node.module or ""
            return mod[len(PKG) + 1:] if mod.startswith(PKG + ".") else None
        parts = self.name.split(".")
        pkg = parts if self.is_pkg else parts[:-1]
        base = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
        return ".".join([*base, node.module] if node.module else base)

    def import_aliases(self, node: ast.ImportFrom) -> dict:
        base = self.resolve_relative(node)
        out = {}
        if base is None:
            return out
        for a in node.names:
            sub = f"{base}.{a.name}" if base else a.name
            local = a.asname or a.name
            if sub in MODULES or (ROOT / sub.replace(".", "/")).is_dir():
                out[local] = ("mod", sub)
            else:
                out[local] = ("obj", base, a.name)
        return out

    def _scan(self):
        for i, node in enumerate(self.tree.body):
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                continue
            if isinstance(node, ast.ImportFrom) and self.resolve_relative(node) is not None:
                self.aliases.update(self.import_aliases(node))
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                src = ast.get_source_segment(self.source, node)
                for a in node.names:
                    bound = (a.asname or a.name).split(".")[0]
                    self.externals[bound] = src if not a.asname and "." in a.name and \
                        isinstance(node, ast.Import) else _single_import(node, a)
                continue
            for name in _bound_names(node):
                self.defs[name] = _Def(self.name, name, node, i)


def _single_import(node, alias) -> str:
    if isinstance(node, ast.Import):
        return f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else "")
    return f"from {node.module} import {alias.name}" + \
        (f" as {alias.asname}" if alias.asname else "")


def _bound_names(node) -> list[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        out = []
        for t in node.targets:
            out += [n.id for n in ast.walk(t) if isinstance(n, ast.Name)]
        return out
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    if isinstance(node, ast.Try):  # optional imports
        return [n.asname or n.name for b in node.body for n in getattr(b, "names", [])] + \
            [t.id for h in node.handlers for b in h.body if isinstance(b, ast.Assign)
             for t in b.targets if isinstance(t, ast.Name)]
    return []


class Bundle:
    """The package's definitions, with their dependencies."""

    def __init__(self):
        self.mods = {m: _Module(m) for m in MODULES}
        for m in self.mods.values():
            for d in m.defs.values():
                self._deps(m, d)

    def resolve(self, module: str, name: str) -> tuple[str, str]:
        """Where ``module.name`` is defined (follows re-exports)."""
        seen = set()
        while (module, name) not in seen:
            seen.add((module, name))
            m = self.mods[module]
            if name in m.defs:
                return module, name
            a = m.aliases.get(name)
            if a and a[0] == "obj":
                module, name = a[1], a[2]
                continue
            break
        raise KeyError(f"{module}.{name} is not defined in brainstudio")

    def _deps(self, m: _Module, d: _Def):
        local = dict(m.aliases)
        for node in ast.walk(d.node):
            if isinstance(node, ast.ImportFrom) and m.resolve_relative(node) is not None \
                    and node is not d.node:
                local.update(m.import_aliases(node))
                d.drop_lines.update(range(node.lineno, node.end_lineno + 1))
        d.aliases = local
        for node in ast.walk(d.node):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                a = local.get(node.value.id)
                if a and a[0] == "mod":
                    d.deps.add((a[1], node.attr))
            if isinstance(node, ast.Name):
                n = node.id
                if n in m.defs and n != d.name:
                    d.deps.add((m.name, n))
                elif n in local and local[n][0] == "obj":
                    d.deps.add((local[n][1], local[n][2]))
                elif n in m.externals:
                    d.externals.add((m.name, n))

    # ------------------------------------------------------------------ closure
    def closure(self, roots, done: set) -> list[tuple[str, str]]:
        """Definitions needed by ``roots`` (``"module.name"``) and not in ``done``,
        dependencies first."""
        order, seen = [], set(done)

        def visit(key):
            key = self.resolve(*key)
            if key in seen:
                return
            seen.add(key)
            for dep in sorted(self.mods[key[0]].defs[key[1]].deps):
                try:
                    visit(self.resolve(*dep))
                except KeyError:  # e.g. an attribute of something that is not ours
                    continue
            order.append(key)

        for r in roots:
            mod, _, name = r.rpartition(".")
            visit((mod, name))
        return order

    def externals(self, keys) -> list[str]:
        out = []
        for k in keys:
            for mod, n in sorted(self.mods[k[0]].defs[k[1]].externals):
                src = self.mods[mod].externals[n]
                if src not in out:
                    out.append(src)
        return out

    # ------------------------------------------------------------------ source
    def names(self, keys) -> dict:
        """Final name of each definition: its own, or ``module_name`` on a clash."""
        by_name: dict = {}
        for k in keys:
            by_name.setdefault(k[1], []).append(k)
        out = {}
        for name, ks in by_name.items():
            for k in ks:
                out[k] = name if len(ks) == 1 else f"{k[0].split('.')[-1]}_{name}"
        return out

    def source(self, key, final: dict) -> str:
        m = self.mods[key[0]]
        d = m.defs[key[1]]
        node = d.node
        start = min([node.lineno] + [x.lineno for x in getattr(node, "decorator_list", [])])
        lines = [ln for i, ln in enumerate(m.lines[start - 1: node.end_lineno], start)
                 if i not in d.drop_lines]
        text = "\n".join(lines)
        # E.foo → foo (module aliases), renamed clashes, from-imported aliases
        for alias, a in d.aliases.items():
            if a[0] == "mod":
                def sub(mt, mod=a[1]):
                    try:
                        k = self.resolve(mod, mt.group(1))
                    except KeyError:
                        return mt.group(0)
                    return final.get(k, mt.group(1))
                text = re.sub(rf"(?<![\w.]){re.escape(alias)}\.(\w+)", sub, text)
            else:
                try:
                    k = self.resolve(a[1], a[2])
                except KeyError:
                    continue
                if final.get(k, alias) != alias:
                    text = re.sub(rf"(?<![\w.]){re.escape(alias)}\b", final[k], text)
        for (mod, name) in d.deps:
            try:
                k = self.resolve(mod, name)
            except KeyError:
                continue
            if mod == key[0] and final.get(k, name) != name:
                text = re.sub(rf"(?<![\w.]){re.escape(name)}\b", final[k], text)
        if final.get(key, key[1]) != key[1]:
            text = re.sub(rf"(?<![\w.]){re.escape(key[1])}\b", final[key], text, count=1)
        return _fix_empty_bodies(text)


def _fix_empty_bodies(text: str) -> str:
    """A body left empty by removing its imports gets ``pass``."""
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        return re.sub(r"(\n(\s*)(?:try|else|if [^\n]*|def [^\n]*):\n)(?=\2\S|\Z)",
                      r"\1\2    pass\n", text)
