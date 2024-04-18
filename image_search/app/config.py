"""Dynaconf settings definition"""
from dynaconf import Dynaconf, Validator

settings = Dynaconf(
    envvar_prefix="T2I_SEARCH",
    settings_files=['settings.yaml', '.secrets.yaml'],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
