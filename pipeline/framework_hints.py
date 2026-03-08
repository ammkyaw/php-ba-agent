"""
pipeline/framework_hints.py — Framework-specific prompt hints for BA agents.

Provides context strings injected into Stage 5 and Stage 6 prompts to
tailor BA documentation for Laravel, Symfony, WordPress, CodeIgniter, and raw PHP.

Usage:
    from pipeline.framework_hints import get_hints
    hints = get_hints(ctx.code_map.framework)
    # hints.ac_template, hints.srs_note, hints.brd_note, hints.qa_focus
"""

from __future__ import annotations
from dataclasses import dataclass
from context import Framework


@dataclass(frozen=True)
class FrameworkHints:
    """Prompt fragments tailored to a specific PHP framework."""
    name:          str    # Human-readable framework name
    brd_note:      str    # BRD-level context (architecture, deployment model)
    srs_note:      str    # SRS-level context (routing, ORM, auth mechanism)
    ac_template:   str    # AC-specific guidance (naming conventions, test hooks)
    story_note:    str    # User story guidance (MVC roles, CLI artisan, etc.)
    qa_focus:      str    # What QA should pay extra attention to


_HINTS: dict[Framework, FrameworkHints] = {

    Framework.LARAVEL: FrameworkHints(
        name        = "Laravel",
        brd_note    = (
            "This is a Laravel application. Business requirements should reference "
            "Eloquent models as entities, Routes (web.php/api.php) as feature entry points, "
            "and Policies/Gates as the authorization model."
        ),
        srs_note    = (
            "Laravel specifics to include in functional requirements: "
            "- Routes defined in routes/web.php or routes/api.php (reference actual route names/URIs)\n"
            "- Controllers in app/Http/Controllers/ handle HTTP logic\n"
            "- Eloquent models in app/Models/ represent DB entities\n"
            "- Form Requests (app/Http/Requests/) define input validation rules\n"
            "- Middleware (app/Http/Middleware/) handles auth, rate limiting\n"
            "- Migrations in database/migrations/ define the DB schema\n"
            "Reference actual class names (e.g., UserController@store, LoginRequest) in FR entries."
        ),
        ac_template = (
            "Write Acceptance Criteria using Laravel-specific conventions:\n"
            "- Reference route names (e.g., POST /auth/login) not just page filenames\n"
            "- Auth checks: 'Given the user is unauthenticated, When they access /dashboard, "
            "Then they are redirected to /login via the auth middleware'\n"
            "- Validation failures: reference Form Request rules (e.g., 'email must be unique in users table')\n"
            "- Use 'artisan' CLI where relevant (migrations, seeding test data)"
        ),
        story_note  = (
            "Laravel user story conventions:\n"
            "- Actors map to Laravel Auth guards (web, api, admin)\n"
            "- Reference Blade views as UI surfaces (e.g., 'resources/views/dashboard.blade.php')\n"
            "- API stories should note whether the endpoint is RESTful (Laravel Resource routes)"
        ),
        qa_focus    = (
            "Laravel-specific QA checks:\n"
            "- Verify all routes are named and referenced consistently across BRD, SRS, AC, and stories\n"
            "- Check that Eloquent model names match DB table names (pluralisation convention)\n"
            "- Confirm middleware names (auth, verified, throttle) are cited in access control AC\n"
            "- Ensure migration column names match Form Request validation field names"
        ),
    ),

    Framework.SYMFONY: FrameworkHints(
        name        = "Symfony",
        brd_note    = (
            "This is a Symfony application. Business requirements should reference "
            "Doctrine Entities as data entities, Controller actions as feature entry points, "
            "and Voters/Security annotations as the authorization model."
        ),
        srs_note    = (
            "Symfony specifics to include in functional requirements:\n"
            "- Routes defined via annotations (#[Route]) or YAML in config/routes/\n"
            "- Controllers in src/Controller/ using AbstractController\n"
            "- Doctrine Entities in src/Entity/ with Repository in src/Repository/\n"
            "- Forms defined in src/Form/ (reference actual FormType class names)\n"
            "- Security in config/packages/security.yaml (firewalls, access_control)\n"
            "- Services in src/Service/ for business logic\n"
            "Reference actual class names (e.g., App\\Controller\\UserController::register) in FRs."
        ),
        ac_template = (
            "Write Acceptance Criteria using Symfony conventions:\n"
            "- Reference route names from #[Route(name:)] annotations\n"
            "- Auth: 'Given the ROLE_USER firewall is active, When...'\n"
            "- Form validation: reference Symfony Constraints (NotBlank, Email, UniqueEntity)\n"
            "- Use 'bin/console' for setup steps (doctrine:schema:update, fixtures:load)"
        ),
        story_note  = (
            "Symfony user story conventions:\n"
            "- Actors map to Symfony Security roles (ROLE_USER, ROLE_ADMIN)\n"
            "- Reference Twig templates as UI surfaces\n"
            "- API stories note if using API Platform (serialization groups, operations)"
        ),
        qa_focus    = (
            "Symfony-specific QA checks:\n"
            "- Verify Security roles cited in AC match those in security.yaml\n"
            "- Check Doctrine Entity field names match form field names in AC\n"
            "- Confirm route names are consistent between SRS and AC\n"
            "- Ensure Repository method names referenced in SRS actually exist in Entity context"
        ),
    ),

    Framework.WORDPRESS: FrameworkHints(
        name        = "WordPress",
        brd_note    = (
            "This is a WordPress application (plugin or theme). Business requirements should "
            "reference Post Types and Taxonomies as data entities, hooks (actions/filters) "
            "as integration points, and WordPress Roles/Capabilities as the auth model."
        ),
        srs_note    = (
            "WordPress specifics to include in functional requirements:\n"
            "- Custom Post Types registered via register_post_type()\n"
            "- Custom fields via ACF or get_post_meta()\n"
            "- Admin pages registered via add_menu_page() / add_submenu_page()\n"
            "- AJAX handlers via wp_ajax_{action} and wp_ajax_nopriv_{action} hooks\n"
            "- REST API endpoints via register_rest_route()\n"
            "- Capabilities: check current_user_can() calls for access control rules"
        ),
        ac_template = (
            "Write Acceptance Criteria using WordPress conventions:\n"
            "- Reference WP Admin screens by menu path (e.g., Posts > Add New)\n"
            "- Auth: 'Given the user has the editor capability, When...'\n"
            "- Shortcodes: 'Given [shortcode_name] is placed on a Page, When...'\n"
            "- Use WP-CLI for setup steps where applicable (wp post create, wp user create)"
        ),
        story_note  = (
            "WordPress user story conventions:\n"
            "- Actors map to WP Roles (Administrator, Editor, Author, Subscriber)\n"
            "- Distinguish front-end (theme) stories from back-end (admin/plugin) stories\n"
            "- Note multisite implications if relevant"
        ),
        qa_focus    = (
            "WordPress-specific QA checks:\n"
            "- Verify WP Roles/Capabilities cited in AC match actual current_user_can() calls\n"
            "- Check that AJAX nonce verification is mentioned in security-related AC\n"
            "- Confirm Custom Post Type slugs are consistent across all documents\n"
            "- Ensure hook names (add_action, add_filter) referenced in SRS are accurate"
        ),
    ),

    Framework.CODEIGNITER: FrameworkHints(
        name        = "CodeIgniter",
        brd_note    = (
            "This is a CodeIgniter application. Business requirements should reference "
            "Models as data entities, Controllers as feature entry points, "
            "and CI Session/Ion Auth as the authentication model."
        ),
        srs_note    = (
            "CodeIgniter specifics to include in functional requirements:\n"
            "- Routes defined in app/Config/Routes.php (CI4) or application/config/routes.php (CI3)\n"
            "- Controllers in app/Controllers/ extending BaseController\n"
            "- Models in app/Models/ using CI Query Builder\n"
            "- Views in app/Views/ as the UI layer\n"
            "- Libraries/Helpers for cross-cutting concerns\n"
            "Reference actual class names (e.g., App\\Controllers\\Auth::login) in FRs."
        ),
        ac_template = (
            "Write Acceptance Criteria using CodeIgniter conventions:\n"
            "- Reference URI segments as route paths (e.g., /auth/login, /dashboard/orders)\n"
            "- Auth: 'Given the session user_id is not set, When the user accesses /dashboard, "
            "Then the redirect() helper sends them to /auth/login'\n"
            "- Validation: reference CI Validation rules (required, valid_email, is_unique)"
        ),
        story_note  = (
            "CodeIgniter user story conventions:\n"
            "- Reference Controller/method pairs as action endpoints\n"
            "- Note CI3 vs CI4 differences if the codebase mixes versions"
        ),
        qa_focus    = (
            "CodeIgniter-specific QA checks:\n"
            "- Verify session-based auth checks are cited in all protected-page AC\n"
            "- Check Model method names match what controllers call in SRS FRs\n"
            "- Confirm URI routing is consistent between SRS and AC documents"
        ),
    ),

    Framework.RAW_PHP: FrameworkHints(
        name        = "Raw PHP",
        brd_note    = (
            "This is a raw PHP application (no framework). Business requirements should reference "
            "PHP files directly as feature entry points, SQL queries as data operations, "
            "and $_SESSION variables as the authentication model."
        ),
        srs_note    = (
            "Raw PHP specifics to include in functional requirements:\n"
            "- Entry points are individual .php files (reference actual filenames)\n"
            "- Database access is direct PDO or mysqli — reference actual SQL table names\n"
            "- Auth is typically $_SESSION-based — cite actual session key names\n"
            "- Form data comes from $_POST/$_GET — reference actual field names\n"
            "- Page flow via header('Location: ...') redirects\n"
            "Include the actual PHP filenames, POST field names, and table names in every FR."
        ),
        ac_template = (
            "Write Acceptance Criteria using raw PHP conventions:\n"
            "- Reference actual .php filenames as pages (e.g., 'When the user submits login.php')\n"
            "- Auth: 'Given $_SESSION[\"user_id\"] is not set, When the user accesses rent.php, "
            "Then they are redirected to login.php via header()'\n"
            "- Validation: cite actual $_POST field names being validated\n"
            "- DB state: 'Then a row is inserted into the [table_name] table with fields [x, y, z]'"
        ),
        story_note  = (
            "Raw PHP user story conventions:\n"
            "- Reference actual .php filenames as pages, not abstract route names\n"
            "- Note direct SQL table interactions rather than ORM abstractions\n"
            "- Session key names are important technical details worth capturing in stories"
        ),
        qa_focus    = (
            "Raw PHP-specific QA checks:\n"
            "- Verify all .php filenames in AC match actual files in the codebase\n"
            "- Check that $_POST field names in AC match actual form field names in SRS\n"
            "- Confirm table names are consistent across BRD data requirements, SRS DB interface, and AC\n"
            "- Ensure session key names are cited consistently when auth checks appear in AC"
        ),
    ),

    Framework.UNKNOWN: FrameworkHints(
        name        = "Unknown PHP Framework",
        brd_note    = "Framework could not be detected. Use filenames and table names as-is.",
        srs_note    = "Reference actual filenames and table names observed in the codebase.",
        ac_template = "Use concrete page names and field names from the domain model.",
        story_note  = "Reference pages and tables directly from the domain model evidence.",
        qa_focus    = "Check that page names, table names, and field names are consistent across all docs.",
    ),
}


def get_hints(framework) -> FrameworkHints:
    """
    Return framework-specific prompt hints for the given framework enum value.
    Falls back to UNKNOWN hints if the framework is not recognised.

    Args:
        framework: A Framework enum value (from context.Framework)

    Returns:
        FrameworkHints dataclass with prompt fragment strings.
    """
    if isinstance(framework, str):
        try:
            framework = Framework(framework)
        except ValueError:
            framework = Framework.UNKNOWN
    return _HINTS.get(framework, _HINTS[Framework.UNKNOWN])
