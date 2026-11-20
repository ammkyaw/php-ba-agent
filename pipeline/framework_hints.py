"""
pipeline/framework_hints.py — Framework-specific prompt hints for BA agents.

Provides context strings injected into Stage 5 and Stage 6 prompts to
tailor BA documentation for all supported frameworks (PHP, TypeScript, Java).

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

    # ── TypeScript / JavaScript frameworks ────────────────────────────────────

    Framework.NEXTJS: FrameworkHints(
        name        = "Next.js (App Router)",
        brd_note    = (
            "This is a Next.js application using the App Router (src/app/ directory). "
            "Business requirements should reference React Server Components and Client "
            "Components as the UI layer, Next.js Server Actions or API Route Handlers "
            "(route.ts) as the server-side logic layer, and the detected data store "
            "(see Tech Stack block) as the persistence layer. "
            "Do NOT describe this as a PHP, monolithic, or legacy backend."
        ),
        srs_note    = (
            "Next.js App Router specifics to include in functional requirements:\n"
            "- Pages live in src/app/**/page.tsx — reference by their URL path, not filename\n"
            "- API endpoints are src/app/**/route.ts exporting GET/POST/PUT/PATCH/DELETE\n"
            "- Server Actions are async functions marked 'use server' — reference by function name\n"
            "- Client Components are marked 'use client' — these run in the browser\n"
            "- Layouts (layout.tsx) wrap pages — note shared navigation and auth guards here\n"
            "- Middleware (middleware.ts) intercepts requests — cite for auth/redirect rules\n"
            "Reference actual file paths and exported function names in FR entries."
        ),
        ac_template = (
            "Write Acceptance Criteria using Next.js App Router conventions:\n"
            "- Routes are URL paths matching src/app structure (e.g., GET /api/tasks)\n"
            "- Auth checks: 'Given the user is unauthenticated, When they access /dashboard, "
            "Then middleware redirects them to /login'\n"
            "- Server Actions: 'When the form is submitted, Then createTask() server action "
            "is invoked and persists data to [detected store]'\n"
            "- Use 'use client' / 'use server' boundaries where test behaviour differs"
        ),
        story_note  = (
            "Next.js user story conventions:\n"
            "- Actors map to authenticated/unauthenticated states (reference auth provider)\n"
            "- Reference page paths as UI surfaces (e.g., '/dashboard', '/projects/[id]')\n"
            "- API stories should note whether the endpoint is a Route Handler or Server Action\n"
            "- Dynamic segments ([id], [slug]) should be noted in page references"
        ),
        qa_focus    = (
            "Next.js-specific QA checks:\n"
            "- Verify route paths match actual src/app directory structure\n"
            "- Confirm Server Action function names cited in AC match source code\n"
            "- Check that auth middleware rules are consistently cited across BRD, SRS, and AC\n"
            "- Ensure data store type (NoSQL/relational) is consistent across all documents"
        ),
    ),

    Framework.REACT: FrameworkHints(
        name        = "React (SPA)",
        brd_note    = (
            "This is a React single-page application. Business requirements should reference "
            "React components as UI surfaces, client-side routing (React Router or similar) "
            "as navigation, and the detected API/backend as the data layer. "
            "Do NOT describe this as a server-rendered or PHP application."
        ),
        srs_note    = (
            "React SPA specifics:\n"
            "- Routes are client-side (React Router): reference path patterns not filenames\n"
            "- Data fetching uses hooks (useEffect, React Query, SWR) — note the API endpoint\n"
            "- Forms use controlled components or React Hook Form — reference field names\n"
            "- Auth state is typically held in Context or a state manager (Redux, Zustand)"
        ),
        ac_template = (
            "Write Acceptance Criteria using React SPA conventions:\n"
            "- Given: auth state in Context/store\n"
            "- When: user action triggers component event handler\n"
            "- Then: API call result updates UI state and component re-renders"
        ),
        story_note  = (
            "React user story conventions:\n"
            "- Reference route paths and component names as UI surfaces\n"
            "- Note whether the story requires an API call or is purely client-side"
        ),
        qa_focus    = (
            "React SPA QA checks:\n"
            "- Verify route paths match React Router configuration\n"
            "- Confirm API endpoint URLs cited in AC are consistent with the backend\n"
            "- Check that auth guard components are consistently referenced"
        ),
    ),

    Framework.NESTJS: FrameworkHints(
        name        = "NestJS",
        brd_note    = (
            "This is a NestJS application — a structured Node.js backend framework. "
            "Business requirements should reference Controllers as API entry points, "
            "Services as business logic, and the detected ORM/database as persistence. "
            "Do NOT describe this as a PHP or monolithic application."
        ),
        srs_note    = (
            "NestJS specifics:\n"
            "- Controllers handle HTTP routes — reference @Controller() and @Get/@Post/@Put etc.\n"
            "- Services encapsulate business logic — reference service class and method names\n"
            "- DTOs (Data Transfer Objects) define input schemas — reference DTO class names\n"
            "- Guards (@UseGuards) handle authentication — cite guard names in auth FRs\n"
            "- Modules group related controllers/services — reference module name as feature boundary"
        ),
        ac_template = (
            "Write Acceptance Criteria using NestJS conventions:\n"
            "- Routes: HTTP method + path from @Controller + @Get/@Post decorators\n"
            "- Validation: DTO class-validator rules (e.g., @IsEmail(), @IsNotEmpty())\n"
            "- Auth: @UseGuards(JwtAuthGuard) — cite the guard name"
        ),
        story_note  = (
            "NestJS user story conventions:\n"
            "- Reference module names as feature boundaries\n"
            "- API stories should cite the controller method and HTTP verb"
        ),
        qa_focus    = (
            "NestJS QA checks:\n"
            "- Verify HTTP verbs and paths match @Controller/@Get/@Post decorator values\n"
            "- Confirm DTO field names match validation rules cited in AC\n"
            "- Check Guard names are consistent across auth-related criteria"
        ),
    ),

    Framework.EXPRESS: FrameworkHints(
        name        = "Express.js",
        brd_note    = (
            "This is an Express.js Node.js backend. Business requirements should reference "
            "route handlers as feature entry points and middleware as cross-cutting concerns. "
            "Do NOT describe this as a PHP application."
        ),
        srs_note    = (
            "Express specifics:\n"
            "- Routes defined via app.get/post/put/delete or Router instances\n"
            "- Middleware (app.use) handles auth, validation, logging\n"
            "- Reference actual route path strings and handler function names"
        ),
        ac_template = (
            "Write AC using Express conventions:\n"
            "- Reference HTTP method + path (e.g., POST /api/users)\n"
            "- Cite middleware by name for auth and validation steps"
        ),
        story_note  = "Reference route paths and middleware names directly.",
        qa_focus    = (
            "Express QA checks:\n"
            "- Verify route paths are consistent across all documents\n"
            "- Confirm middleware order is correctly described in auth flows"
        ),
    ),

    Framework.FASTIFY: FrameworkHints(
        name        = "Fastify",
        brd_note    = (
            "This is a Fastify Node.js backend. Reference route schemas and hooks "
            "as the primary technical constructs. Do NOT describe this as PHP."
        ),
        srs_note    = (
            "Fastify specifics:\n"
            "- Routes registered via fastify.get/post/put/delete with JSON schema validation\n"
            "- Hooks (onRequest, preHandler) handle auth and preprocessing\n"
            "- Plugins encapsulate feature modules"
        ),
        ac_template = "Reference Fastify route paths, JSON schema field names, and hook names in AC.",
        story_note  = "Reference plugin/route boundaries as feature surfaces.",
        qa_focus    = "Verify JSON schema field names match AC validation rules.",
    ),

    Framework.NUXTJS: FrameworkHints(
        name        = "Nuxt.js",
        brd_note    = (
            "This is a Nuxt.js application (Vue-based full-stack framework). "
            "Business requirements should reference pages (pages/ directory), "
            "server API routes (server/api/), and composables as the primary constructs. "
            "Do NOT describe this as a PHP application."
        ),
        srs_note    = (
            "Nuxt specifics:\n"
            "- Pages in pages/ use file-system routing — reference by URL path\n"
            "- Server routes in server/api/ — reference as API endpoints\n"
            "- Composables (useXxx) encapsulate reusable reactive logic\n"
            "- Middleware in middleware/ handles route guards"
        ),
        ac_template = (
            "Write AC using Nuxt conventions:\n"
            "- Reference page paths from the pages/ directory\n"
            "- Server API routes: method + path under /api/"
        ),
        story_note  = "Reference page paths and composable names as feature surfaces.",
        qa_focus    = "Verify page paths and API routes are consistent across documents.",
    ),

    Framework.VUE: FrameworkHints(
        name        = "Vue.js (SPA)",
        brd_note    = (
            "This is a Vue.js single-page application. Reference Vue components and "
            "Vue Router paths as UI surfaces. Do NOT describe this as a PHP application."
        ),
        srs_note    = (
            "Vue SPA specifics:\n"
            "- Routes defined in Vue Router — reference by path and component name\n"
            "- Pinia/Vuex stores manage shared state — reference store module names\n"
            "- Components in src/components/ — reference by component name"
        ),
        ac_template = "Reference Vue Router paths and component names in AC.",
        story_note  = "Reference route paths and store names as feature surfaces.",
        qa_focus    = "Verify Vue Router paths are consistent across all documents.",
    ),

    # ── Java frameworks ────────────────────────────────────────────────────────

    Framework.SPRING_BOOT: FrameworkHints(
        name        = "Spring Boot",
        brd_note    = (
            "This is a Spring Boot application. Business requirements should reference "
            "@RestController endpoints as API entry points, @Service classes as business logic, "
            "and JPA/Hibernate entities as the data model."
        ),
        srs_note    = (
            "Spring Boot specifics:\n"
            "- REST endpoints: @GetMapping/@PostMapping etc. on @RestController classes\n"
            "- Business logic in @Service classes — reference class and method names\n"
            "- JPA entities in @Entity classes — reference entity and field names\n"
            "- Security via Spring Security — cite SecurityFilterChain rules for auth FRs"
        ),
        ac_template = (
            "Write AC using Spring Boot conventions:\n"
            "- HTTP method + path from @RequestMapping values\n"
            "- @Valid DTO fields for validation rules\n"
            "- Spring Security roles for auth preconditions"
        ),
        story_note  = "Reference @RestController paths and @Entity names as feature surfaces.",
        qa_focus    = (
            "Spring Boot QA checks:\n"
            "- Verify @RequestMapping paths match routes cited in AC\n"
            "- Confirm @Entity field names match validation rules"
        ),
    ),

    Framework.UNKNOWN: FrameworkHints(
        name        = "Unknown Framework",
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
