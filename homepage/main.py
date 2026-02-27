from __future__ import annotations


def define_env(env):
    """Load site metadata from external YAML and expose template variables."""
    settings = env.variables.get("site_settings", {})
    if not isinstance(settings, dict):
        settings = {}

    site = settings.get("site", {})
    if not isinstance(site, dict):
        site = {}

    repo_versions = settings.get("repo_versions", {})
    if not isinstance(repo_versions, dict):
        repo_versions = {}

    title = site.get("title")
    if isinstance(title, str) and title.strip():
        env.conf["site_name"] = title
        env.variables["site_name"] = title

    description = site.get("description")
    if isinstance(description, str) and description.strip():
        env.conf["site_description"] = description

    env.variables["site"] = site
    env.variables["ver"] = repo_versions
    env.variables["repo_versions"] = repo_versions
