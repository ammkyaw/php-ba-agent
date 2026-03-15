<?php
/**
 * parse_project.php — PHP-BA Agent: nikic/php-parser bridge
 *
 * Walks a PHP project directory, parses every .php file with nikic/php-parser,
 * and emits a single JSON payload to stdout that Stage 1 (Python) ingests.
 *
 * Usage:
 *   php parse_project.php /path/to/project [--php-version=8.1]
 *
 * Requirements:
 *   composer require nikic/php-parser
 *
 * Output schema (JSON):
 * {
 *   "php_version":  "8.1",
 *   "framework":    "laravel"|"symfony"|"codeigniter"|"wordpress"|"raw_php"|"unknown",
 *   "total_files":  42,
 *   "total_lines":  8310,
 *   "classes":      [ ClassInfo, ... ],
 *   "routes":       [ RouteInfo, ... ],
 *   "models":       [ ModelInfo, ... ],
 *   "controllers":  [ ControllerInfo, ... ],
 *   "services":     [ ServiceInfo, ... ],
 *   "db_schema":    [ MigrationInfo, ... ],
 *   "config_files": [ "config/app.php", ... ],
 *   "errors":       [ { "file": "...", "message": "..." }, ... ],
 *   "call_graph":   [ { "caller": "fn_a", "callee": "fn_b", "file": "...", "line": N }, ... ],
 *   "form_fields":  [ { "file": "...", "action": "...", "method": "POST", "fields": [...] }, ... ],
 *   "service_deps": [ { "class": "...", "dep_class": "...", "dep_var": "...", "file": "..." }, ... ],
 *   "env_vars":     [ { "key": "APP_KEY", "default": null, "file": "...", "line": N }, ... ]
 * }
 */

declare(strict_types=1);

// ─── Bootstrap ───────────────────────────────────────────────────────────────

$autoloadPaths = [
    __DIR__ . '/vendor/autoload.php',           // local vendor
    __DIR__ . '/../vendor/autoload.php',        // one level up
    __DIR__ . '/../../vendor/autoload.php',     // two levels up
];
$autoloaded = false;
foreach ($autoloadPaths as $path) {
    if (file_exists($path)) {
        require_once $path;
        $autoloaded = true;
        break;
    }
}
if (!$autoloaded) {
    fwrite(STDERR, "ERROR: vendor/autoload.php not found. Run: composer require nikic/php-parser\n");
    exit(1);
}

use PhpParser\Error;
use PhpParser\Node;
use PhpParser\NodeTraverser;
use PhpParser\NodeVisitor\NameResolver;
use PhpParser\NodeVisitorAbstract;
use PhpParser\ParserFactory;
use PhpParser\PhpVersion;
use PhpParser\Comment\Doc;

// ─── CLI Arguments ────────────────────────────────────────────────────────────

$opts = getopt('', ['php-version:', 'cache-file:', 'workers:', 'worker-files:']);
$args = array_values(array_filter($argv, fn($a) => !str_starts_with($a, '--') && $a !== $argv[0]));

$phpVersionStr  = $opts['php-version']  ?? '8.1';
$cacheFile      = $opts['cache-file']   ?? null;   // path to JSON hash cache
$workerCount    = isset($opts['workers']) ? max(1, (int)$opts['workers']) : 1;
$workerFilesArg = $opts['worker-files'] ?? null;   // JSON file listing paths to parse (worker mode)

// ── Worker mode: parse only the files listed in --worker-files ────────────────
// Called internally by the parallel dispatcher. Outputs JSON on stdout.
if ($workerFilesArg !== null) {
    if (!file_exists($workerFilesArg)) {
        fwrite(STDERR, "ERROR: --worker-files file not found: {$workerFilesArg}\n");
        exit(1);
    }
    $workerFilePaths = json_decode(file_get_contents($workerFilesArg), true);
    if (!is_array($workerFilePaths)) {
        fwrite(STDERR, "ERROR: --worker-files must be a JSON array of file paths\n");
        exit(1);
    }
    // Project path is still required for relative path calculation
    if (empty($args)) {
        fwrite(STDERR, "Usage: php parse_project.php /path/to/project --worker-files=list.json\n");
        exit(1);
    }
    $projectPath = rtrim(realpath($args[0]), '/');
    if (!$projectPath || !is_dir($projectPath)) {
        fwrite(STDERR, "ERROR: Project path not found: {$args[0]}\n");
        exit(1);
    }
    // Run in worker mode — defined after the parser is set up below
    define('WORKER_MODE', true);
    define('WORKER_FILES', $workerFilePaths);
} else {
    define('WORKER_MODE', false);
    define('WORKER_FILES', []);

    if (empty($args)) {
        fwrite(STDERR, "Usage: php parse_project.php /path/to/project [--php-version=8.1] [--workers=4] [--cache-file=/path/cache.json]\n");
        exit(1);
    }

    $projectPath = rtrim(realpath($args[0]), '/');
    if (!$projectPath || !is_dir($projectPath)) {
        fwrite(STDERR, "ERROR: Project path not found or not a directory: {$args[0]}\n");
        exit(1);
    }
}

// ─── Parser Setup ─────────────────────────────────────────────────────────────

$parserFactory = new ParserFactory();

// Build a version-aware parser
try {
    $phpVersion = PhpVersion::fromString($phpVersionStr);
    $parser = $parserFactory->createForVersion($phpVersion);
} catch (\Throwable $e) {
    // Fall back to latest if version string is unrecognised
    $parser = $parserFactory->createForNewestSupportedVersion();
    $phpVersionStr = 'latest';
}

// ─── Shared Result Buckets ────────────────────────────────────────────────────

$result = [
    'php_version'  => $phpVersionStr,
    'framework'    => 'unknown',
    'total_files'  => 0,
    'total_lines'  => 0,
    'classes'      => [],
    'routes'       => [],
    'models'       => [],
    'controllers'  => [],
    'services'     => [],
    'db_schema'    => [],
    'config_files' => [],
    'errors'       => [],
    // Procedural (raw_php) extractions
    'functions'       => [],   // all defined functions
    'includes'        => [],   // include/require calls
    'sql_queries'     => [],   // direct mysqli/PDO queries
    'redirects'       => [],   // header("Location: ...") calls
    'globals'         => [],   // global variable usages
    'html_pages'      => [],   // files that mix PHP + HTML (page-level entry points)
    'superglobals'    => [],   // $_POST/$_SESSION/$_GET reads per file+line
    // ── New extractions ──────────────────────────────────────────────────────
    'call_graph'      => [],   // function-to-function call edges (caller → callee)
    'form_fields'     => [],   // HTML <form> fields extracted from PHP-mixed files
    'service_deps'    => [],   // constructor-injected dependencies per class
    'env_vars'        => [],   // env('KEY') / getenv('KEY') / $_ENV['KEY'] usages
    'auth_signals'    => [],   // password_verify/Auth::attempt/session auth patterns
    'http_endpoints'  => [],   // files/methods that accept HTTP requests
    'table_columns'   => [],   // table+column defs from SQL DDL + Eloquent fillable
];

// ─── Framework Detection ──────────────────────────────────────────────────────

function detectFramework(string $projectPath): string
{
    $indicators = [
        'laravel'     => ['artisan', 'app/Http/Controllers', 'bootstrap/app.php'],
        'symfony'     => ['bin/console', 'config/bundles.php', 'src/Kernel.php'],
        'codeigniter' => ['system/CodeIgniter.php', 'application/config/config.php', 'spark'],
        'wordpress'   => ['wp-config.php', 'wp-includes/functions.php', 'wp-login.php'],
    ];

    foreach ($indicators as $framework => $paths) {
        $hits = 0;
        foreach ($paths as $path) {
            if (file_exists($projectPath . '/' . $path)) {
                $hits++;
            }
        }
        if ($hits >= 2) {
            return $framework;
        }
    }

    // Single-file indicators
    if (file_exists($projectPath . '/artisan'))            return 'laravel';
    if (file_exists($projectPath . '/bin/console'))        return 'symfony';
    if (file_exists($projectPath . '/wp-config.php'))      return 'wordpress';
    if (file_exists($projectPath . '/spark'))              return 'codeigniter';

    // Check composer.json for framework dependency
    $composerFile = $projectPath . '/composer.json';
    if (file_exists($composerFile)) {
        $composer = json_decode(file_get_contents($composerFile), true) ?? [];
        $require  = array_merge($composer['require'] ?? [], $composer['require-dev'] ?? []);
        $keys     = array_keys($require);
        foreach ($keys as $pkg) {
            if (str_contains($pkg, 'laravel/framework'))  return 'laravel';
            if (str_contains($pkg, 'symfony/framework'))  return 'symfony';
            if (str_contains($pkg, 'codeigniter4'))       return 'codeigniter';
            if (str_contains($pkg, 'wordpress'))          return 'wordpress';
        }
    }

    return 'raw_php';
}

$result['framework'] = detectFramework($projectPath);

// ─── Config File Collection ───────────────────────────────────────────────────

function collectConfigFiles(string $projectPath): array
{
    $configs = [];
    $configDirs = ['config', 'configuration', 'app/config', 'src/config', 'conf'];

    foreach ($configDirs as $dir) {
        $fullDir = $projectPath . '/' . $dir;
        if (!is_dir($fullDir)) continue;
        $it = new RecursiveIteratorIterator(new RecursiveDirectoryIterator($fullDir));
        foreach ($it as $file) {
            if ($file->isFile() && $file->getExtension() === 'php') {
                $configs[] = str_replace($projectPath . '/', '', $file->getPathname());
            }
        }
    }

    // Root-level config files
    foreach (['wp-config.php', '.env.php', 'config.php', 'settings.php'] as $f) {
        if (file_exists($projectPath . '/' . $f)) {
            $configs[] = $f;
        }
    }

    return array_unique($configs);
}

$result['config_files'] = collectConfigFiles($projectPath);

// ─── AST Visitor ─────────────────────────────────────────────────────────────

/**
 * Walks each parsed AST and extracts classes, methods, properties, docblocks,
 * route registrations, and Eloquent/Doctrine model metadata.
 */
class ProjectVisitor extends NodeVisitorAbstract
{
    /** @var array<string, mixed> */
    public array $classes = [];

    /** @var array<string, mixed> */
    public array $routes = [];

    private string  $currentFile;
    private string  $framework;
    private ?string $currentNamespace = null;
    private ?string $currentClass     = null;
    private array   $routeGroupStack  = [];  // stack of ['prefix'=>..., 'middleware'=>...]

    public function __construct(string $currentFile, string $framework)
    {
        $this->currentFile = $currentFile;
        $this->framework   = $framework;
    }

    // ── Namespace tracking ───────────────────────────────────────────────────

    public function enterNode(Node $node): ?int
    {
        if ($node instanceof Node\Stmt\Namespace_) {
            $this->currentNamespace = $node->name ? $node->name->toString() : null;
        }

        if ($node instanceof Node\Stmt\Class_) {
            $this->handleClass($node);
        }

        if ($node instanceof Node\Stmt\Interface_) {
            $this->handleInterface($node);
        }

        if ($node instanceof Node\Stmt\Trait_) {
            $this->handleTrait($node);
        }

        if ($node instanceof Node\Expr\StaticCall || $node instanceof Node\Expr\MethodCall) {
            $this->handleRouteCall($node);
        }

        return null;
    }

    public function leaveNode(Node $node): void
    {
        if ($node instanceof Node\Stmt\Class_) {
            $this->currentClass = null;
        }
    }

    // ── Class Extraction ─────────────────────────────────────────────────────

    private function handleClass(Node\Stmt\Class_ $node): void
    {
        if (!$node->name) return; // anonymous class

        $fqn = $this->currentNamespace
            ? $this->currentNamespace . '\\' . $node->name->toString()
            : $node->name->toString();

        $this->currentClass = $fqn;

        $classInfo = [
            'name'        => $node->name->toString(),
            'fqn'         => $fqn,
            'namespace'   => $this->currentNamespace,
            'file'        => $this->currentFile,
            'line'        => $node->getStartLine(),
            'is_abstract' => $node->isAbstract(),
            'is_final'    => $node->isFinal(),
            'is_readonly' => method_exists($node, 'isReadonly') && $node->isReadonly(),
            'extends'     => $node->extends ? $node->extends->toString() : null,
            'implements'  => array_map(fn($i) => $i->toString(), $node->implements),
            'traits'      => $this->extractUsedTraits($node),
            'docblock'    => $this->extractDocblock($node),
            'methods'     => $this->extractMethods($node),
            'properties'  => $this->extractProperties($node),
            'constants'   => $this->extractClassConstants($node),
            'type'        => $this->inferClassType($node, $fqn),
        ];

        $this->classes[$fqn] = $classInfo;
    }

    private function handleInterface(Node\Stmt\Interface_ $node): void
    {
        $fqn = $this->currentNamespace
            ? $this->currentNamespace . '\\' . $node->name->toString()
            : $node->name->toString();

        $this->classes[$fqn] = [
            'name'      => $node->name->toString(),
            'fqn'       => $fqn,
            'namespace' => $this->currentNamespace,
            'file'      => $this->currentFile,
            'line'      => $node->getStartLine(),
            'type'      => 'interface',
            'docblock'  => $this->extractDocblock($node),
            'extends'   => array_map(fn($e) => $e->toString(), $node->extends),
            'methods'   => $this->extractMethods($node),
        ];
    }

    private function handleTrait(Node\Stmt\Trait_ $node): void
    {
        $fqn = $this->currentNamespace
            ? $this->currentNamespace . '\\' . $node->name->toString()
            : $node->name->toString();

        $this->classes[$fqn] = [
            'name'       => $node->name->toString(),
            'fqn'        => $fqn,
            'namespace'  => $this->currentNamespace,
            'file'       => $this->currentFile,
            'line'       => $node->getStartLine(),
            'type'       => 'trait',
            'docblock'   => $this->extractDocblock($node),
            'methods'    => $this->extractMethods($node),
            'properties' => $this->extractProperties($node),
        ];
    }

    // ── Method Extraction ────────────────────────────────────────────────────

    private function extractMethods(Node\Stmt\ClassLike $node): array
    {
        $methods = [];
        foreach ($node->stmts as $stmt) {
            if (!($stmt instanceof Node\Stmt\ClassMethod)) continue;

            $params = [];
            foreach ($stmt->params as $param) {
                $params[] = [
                    'name'    => '$' . $param->var->name,
                    'type'    => $param->type ? $this->typeToString($param->type) : null,
                    'default' => $param->default ? $this->nodeToString($param->default) : null,
                    'promoted'=> (bool)($param->flags & Node\Stmt\Class_::MODIFIER_PUBLIC
                                      | $param->flags & Node\Stmt\Class_::MODIFIER_PROTECTED
                                      | $param->flags & Node\Stmt\Class_::MODIFIER_PRIVATE),
                ];
            }

            $methods[] = [
                'name'        => $stmt->name->toString(),
                'visibility'  => $this->visibility($stmt->flags),
                'is_static'   => (bool)($stmt->flags & Node\Stmt\Class_::MODIFIER_STATIC),
                'is_abstract' => (bool)($stmt->flags & Node\Stmt\Class_::MODIFIER_ABSTRACT),
                'return_type' => $stmt->returnType ? $this->typeToString($stmt->returnType) : null,
                'params'      => $params,
                'docblock'    => $this->extractDocblock($stmt),
                'line'        => $stmt->getStartLine(),
            ];
        }
        return $methods;
    }

    // ── Property Extraction ──────────────────────────────────────────────────

    private function extractProperties(Node\Stmt\ClassLike $node): array
    {
        $properties = [];
        foreach ($node->stmts as $stmt) {
            if (!($stmt instanceof Node\Stmt\Property)) continue;
            foreach ($stmt->props as $prop) {
                $properties[] = [
                    'name'       => '$' . $prop->name->toString(),
                    'visibility' => $this->visibility($stmt->flags),
                    'is_static'  => (bool)($stmt->flags & Node\Stmt\Class_::MODIFIER_STATIC),
                    'type'       => $stmt->type ? $this->typeToString($stmt->type) : null,
                    'docblock'   => $this->extractDocblock($stmt),
                    'line'       => $stmt->getStartLine(),
                ];
            }
        }
        return $properties;
    }

    // ── Constants ────────────────────────────────────────────────────────────

    private function extractClassConstants(Node\Stmt\ClassLike $node): array
    {
        $constants = [];
        foreach ($node->stmts as $stmt) {
            if (!($stmt instanceof Node\Stmt\ClassConst)) continue;
            foreach ($stmt->consts as $const) {
                $constants[] = [
                    'name'       => $const->name->toString(),
                    'visibility' => $this->visibility($stmt->flags),
                    'value'      => $this->nodeToString($const->value),
                ];
            }
        }
        return $constants;
    }

    // ── Trait Use ────────────────────────────────────────────────────────────

    private function extractUsedTraits(Node\Stmt\Class_ $node): array
    {
        $traits = [];
        foreach ($node->stmts as $stmt) {
            if ($stmt instanceof Node\Stmt\TraitUse) {
                foreach ($stmt->traits as $trait) {
                    $traits[] = $trait->toString();
                }
            }
        }
        return $traits;
    }

    // ── Route Detection ──────────────────────────────────────────────────────

    /**
     * Detects all Laravel route registration patterns:
     *
     *   HTTP verbs:  Route::get/post/put/patch/delete/options/any/match
     *   Resources:   Route::resource('photos', PhotoController::class)
     *                Route::apiResource('photos', PhotoController::class)
     *   Groups:      Route::group(['prefix'=>'api','middleware'=>'auth'], ...)
     *                Route::prefix('admin')->group(...)
     *                Route::middleware('auth')->group(...)
     *   Chained:     Route::prefix('api')->middleware('auth')->group(...)
     *
     * Symfony annotation/attribute routes are handled by extractSymfonyRoutes().
     *
     * Route groups are stored with type='group' and expand the prefix/middleware
     * context for child routes discovered in later passes (best-effort; full
     * static resolution of closures is not attempted here).
     */
    private function handleRouteCall(Node $node): void
    {
        // ── Static calls: Route::xxx(...) ────────────────────────────────────
        if ($node instanceof Node\Expr\StaticCall) {
            if (!($node->class instanceof Node\Name)) return;
            $className  = strtolower($node->class->toString());
            $methodName = $node->name instanceof Node\Identifier ? $node->name->toString() : null;
            if (!in_array($className, ['route', 'router'], true)) return;
            if (!$methodName) return;

            // ── Route::resource / Route::apiResource ──────────────────────
            if (in_array($methodName, ['resource', 'apiResource'], true)) {
                $name    = $this->extractStringArg($node->args, 0);
                $handler = $this->extractStringArg($node->args, 1)
                        ?? $this->extractClassConstArg($node->args, 1);
                $only    = $this->extractRouteOnlyExcept($node->args, 'only');
                $except  = $this->extractRouteOnlyExcept($node->args, 'except');

                // Generate all RESTful routes for this resource
                $resourceRoutes = [
                    ['GET',    "/{$name}",              'index'],
                    ['POST',   "/{$name}",              'store'],
                    ['GET',    "/{$name}/create",       'create'],
                    ['GET',    "/{$name}/{id}",         'show'],
                    ['PUT',    "/{$name}/{id}",         'update'],
                    ['PATCH',  "/{$name}/{id}",         'update'],
                    ['DELETE', "/{$name}/{id}",         'destroy'],
                    ['GET',    "/{$name}/{id}/edit",    'edit'],
                ];
                // apiResource skips create/edit (HTML form) routes
                if ($methodName === 'apiResource') {
                    $resourceRoutes = array_filter($resourceRoutes,
                        fn($r) => !in_array($r[2], ['create','edit'], true));
                }

                foreach ($resourceRoutes as [$verb, $path, $action]) {
                    if ($only  && !in_array($action, $only,   true)) continue;
                    if ($except &&  in_array($action, $except, true)) continue;
                    $this->routes[] = [
                        'method'        => $verb,
                        'path'          => $name ? $path : '(dynamic)',
                        'handler'       => $handler ? "{$handler}@{$action}" : "(closure)@{$action}",
                        'resource_name' => $name,
                        'resource_type' => $methodName,
                        'file'          => $this->currentFile,
                        'line'          => $node->getStartLine(),
                        'source'        => 'laravel_resource',
                    ];
                }
                return;
            }

            // ── Route::group(['prefix'=>..., 'middleware'=>...], closure) ──
            if ($methodName === 'group') {
                $attrs = $this->extractRouteGroupAttrs($node->args, 0);
                $this->routes[] = array_merge($attrs, [
                    'method'  => 'GROUP',
                    'path'    => $attrs['prefix'] ?? '(group)',
                    'handler' => '(closure)',
                    'file'    => $this->currentFile,
                    'line'    => $node->getStartLine(),
                    'source'  => 'laravel_group',
                ]);
                return;
            }

            // ── Route::prefix('admin') — returns route group builder ──────
            // We record it as a GROUP node; chained ->group() picks up the prefix
            if ($methodName === 'prefix') {
                $prefix = $this->extractStringArg($node->args, 0);
                $this->routeGroupStack[] = ['prefix' => $prefix];
                // Don't add a route entry — the chained ->group() will do it
                return;
            }

            // ── Route::middleware('auth') ─────────────────────────────────
            if ($methodName === 'middleware') {
                $mw = $this->extractStringOrArrayArg($node->args, 0);
                $this->routeGroupStack[] = ['middleware' => $mw];
                return;
            }

            // ── HTTP verb routes: Route::get/post/etc. ────────────────────
            $httpVerbs = ['get','post','put','patch','delete','options','any','match'];
            if (!in_array($methodName, $httpVerbs, true)) return;

            $path    = $this->extractStringArg($node->args, 0);
            $handler = $this->extractRouteHandler($node->args);

            if ($path !== null || $handler !== null) {
                $this->routes[] = array_merge($this->currentGroupContext(), [
                    'method'  => strtoupper($methodName),
                    'path'    => $path ?? '(dynamic)',
                    'handler' => $handler,
                    'file'    => $this->currentFile,
                    'line'    => $node->getStartLine(),
                    'source'  => 'laravel_static',
                ]);
            }
        }

        // ── Method calls: $router->get(...)  OR chained ->group() ────────────
        if ($node instanceof Node\Expr\MethodCall) {
            $methodName = $node->name instanceof Node\Identifier ? $node->name->toString() : null;
            if (!$methodName) return;

            // Chained group() — apply any stacked prefix/middleware context
            if ($methodName === 'group') {
                $attrs = $this->extractRouteGroupAttrs($node->args, 0);
                // Merge stacked context from prefix()/middleware() calls
                $stacked = array_pop($this->routeGroupStack) ?? [];
                $merged  = array_merge($stacked, $attrs);
                $this->routes[] = array_merge($merged, [
                    'method'  => 'GROUP',
                    'path'    => $merged['prefix'] ?? '(group)',
                    'handler' => '(closure)',
                    'file'    => $this->currentFile,
                    'line'    => $node->getStartLine(),
                    'source'  => 'laravel_group',
                ]);
                return;
            }

            // Chained middleware() — push onto stack
            if ($methodName === 'middleware') {
                $mw = $this->extractStringOrArrayArg($node->args, 0);
                if (!empty($this->routeGroupStack)) {
                    $this->routeGroupStack[array_key_last($this->routeGroupStack)]['middleware'] = $mw;
                } else {
                    $this->routeGroupStack[] = ['middleware' => $mw];
                }
                return;
            }

            // Chained prefix()
            if ($methodName === 'prefix') {
                $prefix = $this->extractStringArg($node->args, 0);
                if (!empty($this->routeGroupStack)) {
                    $this->routeGroupStack[array_key_last($this->routeGroupStack)]['prefix'] = $prefix;
                } else {
                    $this->routeGroupStack[] = ['prefix' => $prefix];
                }
                return;
            }

            // HTTP verbs on $router instance
            $httpVerbs = ['get','post','put','patch','delete','options','any','match','resource','apiResource'];
            if (!in_array($methodName, $httpVerbs, true)) return;

            $path    = $this->extractStringArg($node->args, 0);
            $handler = $this->extractRouteHandler($node->args);

            if ($path !== null || $handler !== null) {
                $this->routes[] = array_merge($this->currentGroupContext(), [
                    'method'  => strtoupper($methodName),
                    'path'    => $path ?? '(dynamic)',
                    'handler' => $handler,
                    'file'    => $this->currentFile,
                    'line'    => $node->getStartLine(),
                    'source'  => 'laravel_fluent',
                ]);
            }
        }
    }

    /** Returns the current prefix/middleware context from the group stack. */
    private function currentGroupContext(): array
    {
        if (empty($this->routeGroupStack)) return [];
        $ctx = array_merge(...$this->routeGroupStack);
        return array_filter([
            'prefix'     => $ctx['prefix']     ?? null,
            'middleware' => $ctx['middleware']  ?? null,
        ]);
    }

    /**
     * Extract 'only' or 'except' option from a resource route's options array.
     * Route::resource('photos', PhotoController::class, ['only' => ['index','show']])
     */
    private function extractRouteOnlyExcept(array $args, string $key): ?array
    {
        if (!isset($args[2])) return null;
        $val = $args[2]->value ?? $args[2];
        if ($val instanceof Node\Arg) $val = $val->value;
        if (!($val instanceof Node\Expr\Array_)) return null;

        foreach ($val->items as $item) {
            if (!$item || !$item->key) continue;
            $itemKey = $item->key;
            if ($itemKey instanceof Node\Scalar\String_ && $itemKey->value === $key) {
                $arrVal = $item->value;
                if ($arrVal instanceof Node\Expr\Array_) {
                    return array_filter(array_map(function($i) {
                        $v = $i->value ?? null;
                        return ($v instanceof Node\Scalar\String_) ? $v->value : null;
                    }, $arrVal->items));
                }
            }
        }
        return null;
    }

    /**
     * Extract the attributes from a Route::group(['prefix'=>..., 'middleware'=>...], ...)
     * options array as a flat associative array.
     */
    private function extractRouteGroupAttrs(array $args, int $index): array
    {
        $attrs = [];
        if (!isset($args[$index])) return $attrs;

        $val = $args[$index]->value ?? $args[$index];
        if ($val instanceof Node\Arg) $val = $val->value;
        if (!($val instanceof Node\Expr\Array_)) return $attrs;

        foreach ($val->items as $item) {
            if (!$item || !($item->key instanceof Node\Scalar\String_)) continue;
            $k = $item->key->value;
            $v = $item->value;

            if ($k === 'prefix' && $v instanceof Node\Scalar\String_) {
                $attrs['prefix'] = $v->value;
            } elseif ($k === 'middleware') {
                $attrs['middleware'] = $this->nodeToStringOrArray($v);
            } elseif ($k === 'namespace' && $v instanceof Node\Scalar\String_) {
                $attrs['namespace'] = $v->value;
            } elseif ($k === 'name' && $v instanceof Node\Scalar\String_) {
                $attrs['name_prefix'] = $v->value;
            } elseif ($k === 'as' && $v instanceof Node\Scalar\String_) {
                $attrs['name_prefix'] = $v->value;
            }
        }
        return $attrs;
    }

    /** Extracts a string or array of strings from a call argument. */
    private function extractStringOrArrayArg(array $args, int $index): array|string|null
    {
        if (!isset($args[$index])) return null;
        $val = $args[$index]->value ?? $args[$index];
        if ($val instanceof Node\Arg) $val = $val->value;

        if ($val instanceof Node\Scalar\String_) return $val->value;
        if ($val instanceof Node\Expr\Array_) {
            return array_values(array_filter(array_map(function($item) {
                $v = $item->value ?? null;
                return ($v instanceof Node\Scalar\String_) ? $v->value : null;
            }, $val->items)));
        }
        return null;
    }

    /** Convert a node to string or array of strings (for middleware arrays). */
    private function nodeToStringOrArray(Node $node): array|string|null
    {
        if ($node instanceof Node\Scalar\String_) return $node->value;
        if ($node instanceof Node\Expr\Array_) {
            return array_values(array_filter(array_map(function($item) {
                $v = $item->value ?? null;
                return ($v instanceof Node\Scalar\String_) ? $v->value : null;
            }, $node->items)));
        }
        return null;
    }

    /** Extract first arg when it is a ::class constant fetch (e.g. PhotoController::class) */
    private function extractClassConstArg(array $args, int $index): ?string
    {
        if (!isset($args[$index])) return null;
        $val = $args[$index]->value ?? $args[$index];
        if ($val instanceof Node\Arg) $val = $val->value;
        if ($val instanceof Node\Expr\ClassConstFetch) {
            return $val->class->toString();
        }
        return null;
    }

    // ── Class Type Inference ─────────────────────────────────────────────────

    private function inferClassType(Node\Stmt\Class_ $node, string $fqn): string
    {
        $name    = strtolower($node->name ? $node->name->toString() : '');
        $extends = $node->extends ? strtolower($node->extends->toString()) : '';
        $impls   = array_map(fn($i) => strtolower($i->toString()), $node->implements);

        // Controller
        if (str_ends_with($name, 'controller') || str_contains($extends, 'controller')) {
            return 'controller';
        }

        // Model
        if (
            str_contains($extends, 'model') ||
            str_contains($extends, 'eloquent') ||
            str_contains($extends, 'entity')
        ) {
            return 'model';
        }

        // Service
        if (str_ends_with($name, 'service') || str_ends_with($name, 'manager')) {
            return 'service';
        }

        // Repository
        if (str_ends_with($name, 'repository') || str_ends_with($name, 'repo')) {
            return 'repository';
        }

        // Middleware
        if (str_ends_with($name, 'middleware') || in_array('middleware', $impls, true)) {
            return 'middleware';
        }

        // Command
        if (str_ends_with($name, 'command') || str_contains($extends, 'command')) {
            return 'command';
        }

        // Event / Listener
        if (str_ends_with($name, 'event'))    return 'event';
        if (str_ends_with($name, 'listener')) return 'listener';
        if (str_ends_with($name, 'job'))      return 'job';
        if (str_ends_with($name, 'policy'))   return 'policy';
        if (str_ends_with($name, 'request'))  return 'form_request';
        if (str_ends_with($name, 'resource')) return 'api_resource';
        if (str_ends_with($name, 'factory'))  return 'factory';
        if (str_ends_with($name, 'seeder'))   return 'seeder';
        if (str_ends_with($name, 'migration'))return 'migration';

        return 'class';
    }

    // ── Docblock Extraction ──────────────────────────────────────────────────

    private function extractDocblock(Node $node): ?array
    {
        $docComment = $node->getDocComment();
        if (!$docComment instanceof Doc) return null;

        $raw  = $docComment->getText();
        $text = $this->cleanDocblock($raw);

        return [
            'raw'         => $raw,
            'summary'     => $this->extractDocSummary($text),
            'params'      => $this->extractDocParams($raw),
            'return'      => $this->extractDocReturn($raw),
            'tags'        => $this->extractDocTags($raw),
            'annotations' => $this->extractAnnotations($raw),
        ];
    }

    private function cleanDocblock(string $raw): string
    {
        $lines = explode("\n", $raw);
        $clean = [];
        foreach ($lines as $line) {
            $line = trim($line, " \t\r\n*\/");
            if ($line !== '') $clean[] = $line;
        }
        return implode(' ', $clean);
    }

    private function extractDocSummary(string $text): string
    {
        // Summary is everything before the first @tag or blank-line separator
        if (preg_match('/^([^@]+?)(?=\s*@|\z)/s', $text, $m)) {
            return trim($m[1]);
        }
        return trim($text);
    }

    private function extractDocParams(string $raw): array
    {
        $params = [];
        preg_match_all('/@param\s+(\S+)\s+(\$\S+)(?:\s+(.+))?/', $raw, $matches, PREG_SET_ORDER);
        foreach ($matches as $m) {
            $params[] = [
                'type' => $m[1],
                'name' => $m[2],
                'desc' => isset($m[3]) ? trim($m[3]) : '',
            ];
        }
        return $params;
    }

    private function extractDocReturn(string $raw): ?string
    {
        if (preg_match('/@return\s+(\S+)(?:\s+(.+))?/', $raw, $m)) {
            return $m[1] . (isset($m[2]) ? ' ' . trim($m[2]) : '');
        }
        return null;
    }

    private function extractDocTags(string $raw): array
    {
        $tags = [];
        preg_match_all('/@(\w+)(?:\s+([^\n@]*))?/', $raw, $matches, PREG_SET_ORDER);
        foreach ($matches as $m) {
            $tag = $m[1];
            if (in_array($tag, ['param', 'return'])) continue; // already extracted
            $tags[$tag][] = isset($m[2]) ? trim($m[2]) : '';
        }
        return $tags;
    }

    /** Extracts Doctrine/custom annotation-style tags like @ORM\Entity, @Route */
    private function extractAnnotations(string $raw): array
    {
        $annotations = [];
        preg_match_all('/@([A-Z][A-Za-z\\\\]+)\s*(?:\(([^)]*)\))?/', $raw, $matches, PREG_SET_ORDER);
        foreach ($matches as $m) {
            $annotations[] = [
                'name'   => $m[1],
                'params' => isset($m[2]) ? trim($m[2]) : '',
            ];
        }

        // PHP 8 attribute-style (captured as strings for now — real attributes
        // are extracted from Node\Attribute nodes when available)
        preg_match_all('/#\[([^\]]+)\]/', $raw, $attrMatches, PREG_SET_ORDER);
        foreach ($attrMatches as $m) {
            $annotations[] = ['name' => trim($m[1]), 'params' => '', 'style' => 'php8_attribute'];
        }

        return $annotations;
    }

    // ── PHP 8 Attribute Support ──────────────────────────────────────────────

    /**
     * Extracts PHP 8.x #[Attribute] nodes from class/method nodes.
     * These are used for Symfony #[Route(...)] and similar.
     */
    public function extractPhp8Attributes(Node $node): array
    {
        $attrs = [];
        if (!property_exists($node, 'attrGroups')) return $attrs;
        foreach ($node->attrGroups as $group) {
            foreach ($group->attrs as $attr) {
                $args = [];
                foreach ($attr->args as $arg) {
                    $key        = $arg->name ? $arg->name->toString() : null;
                    $val        = $this->nodeToString($arg->value);
                    $args[$key ?? count($args)] = $val;
                }
                $attrs[] = [
                    'name' => $attr->name->toString(),
                    'args' => $args,
                ];
            }
        }
        return $attrs;
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private function visibility(int $flags): string
    {
        if ($flags & Node\Stmt\Class_::MODIFIER_PRIVATE)   return 'private';
        if ($flags & Node\Stmt\Class_::MODIFIER_PROTECTED) return 'protected';
        return 'public'; // default
    }

    private function typeToString(Node $type): string
    {
        if ($type instanceof Node\Identifier)        return $type->toString();
        if ($type instanceof Node\Name)              return $type->toString();
        if ($type instanceof Node\NullableType)      return '?' . $this->typeToString($type->type);
        if ($type instanceof Node\UnionType) {
            return implode('|', array_map(fn($t) => $this->typeToString($t), $type->types));
        }
        if ($type instanceof Node\IntersectionType) {
            return implode('&', array_map(fn($t) => $this->typeToString($t), $type->types));
        }
        return 'mixed';
    }

    private function nodeToString(Node $node): string
    {
        if ($node instanceof Node\Scalar\String_)  return $node->value;
        if ($node instanceof Node\Scalar\Int_)     return (string)$node->value;
        if ($node instanceof Node\Scalar\Float_)   return (string)$node->value;
        if ($node instanceof Node\Expr\ConstFetch) return $node->name->toString();
        if ($node instanceof Node\Expr\Array_)     return '[...]';
        if ($node instanceof Node\Expr\ClassConstFetch) {
            return $node->class->toString() . '::' . $node->name->toString();
        }
        return '...';
    }

    private function extractStringArg(array $args, int $index): ?string
    {
        if (!isset($args[$index])) return null;
        $val = $args[$index]->value ?? $args[$index];
        if ($val instanceof Node\Arg) $val = $val->value;
        if ($val instanceof Node\Scalar\String_) return $val->value;
        if ($val instanceof Node\Scalar\InterpolatedString) return '(dynamic)';
        return null;
    }

    private function extractRouteHandler(array $args): ?string
    {
        if (!isset($args[1])) return null;
        $val = $args[1]->value ?? $args[1];
        if ($val instanceof Node\Arg) $val = $val->value;

        // String controller: 'UserController@index'
        if ($val instanceof Node\Scalar\String_) return $val->value;

        // Array: [UserController::class, 'index']
        if ($val instanceof Node\Expr\Array_ && count($val->items) === 2) {
            $class  = $this->nodeToString($val->items[0]->value);
            $method = $this->nodeToString($val->items[1]->value);
            return $class . '@' . $method;
        }

        // Closure — just mark it
        if ($val instanceof Node\Expr\Closure) return '(closure)';

        return null;
    }
}

// ─── Procedural PHP Visitor ───────────────────────────────────────────────────

/**
 * Extracts everything meaningful from raw/procedural PHP files:
 *   - Function definitions (name, params, docblock, line)
 *   - Function calls with caller context
 *   - Direct SQL queries via mysqli_query/PDO/pg_query with table + operation
 *   - header("Location: ...") redirects
 *   - include/require dependencies
 *   - Global variable usages
 *   - Session and cookie operations
 *   - Whether the file is an HTML entry-point page
 *
 * This runs on ALL files regardless of framework so that raw_php projects
 * are fully mapped even when they have zero classes.
 */
class ProceduralVisitor extends NodeVisitorAbstract
{
    public array $functions  = [];
    public array $calls      = [];
    public array $callEdges  = [];   // call graph edges: {caller, callee, file, line}
    public array $sqlQueries = [];
    public array $redirects  = [];
    public array $includes   = [];
    public array $globals    = [];
    public array $envVars    = [];   // getenv / $_ENV / env() usages
    public bool  $isHtmlPage = false;

    private string  $currentFile;
    private ?string $currentFunction = null;

    /**
     * Variable assignment table: tracks $var = "SQL string" assignments
     * so they can be resolved when passed to mysqli_query($conn, $var).
     * Scoped per-file; keyed by variable name (without $).
     * e.g. ['sql' => 'SELECT * FROM users', 'query' => 'INSERT INTO ...']
     */
    private array $varSqlMap = [];

    /**
     * Superglobal accesses: tracks $_POST/$_SESSION/$_GET/$_FILES/$_COOKIE reads.
     * Each entry: ['var' => '$_POST', 'key' => 'email', 'file' => '...', 'line' => N]
     */
    public array $superglobalReads = [];

    // Superglobals we care about for BA purposes
    private const SUPERGLOBALS = ['_POST', '_GET', '_SESSION', '_FILES', '_COOKIE', '_REQUEST', '_ENV', '_SERVER'];

    // Functions that indicate this file is an HTML entry-point page
    private const PAGE_INDICATORS = [
        'header', 'echo', 'print', 'printf', 'ob_start',
        'session_start', 'htmlspecialchars', 'htmlentities',
    ];

    // All direct DB call signatures we recognise
    private const SQL_FUNCTIONS = [
        'mysqli_query', 'mysqli_multi_query', 'mysqli_real_query',
        'mysql_query',
        'pg_query', 'pg_execute',
        'sqlite_query',
        'mssql_query',
    ];

    private const SESSION_FUNCTIONS = [
        'session_start', 'session_destroy', 'session_regenerate_id',
        'session_unset',
    ];

    public function __construct(string $currentFile)
    {
        $this->currentFile = $currentFile;
    }

    // ── Traversal ────────────────────────────────────────────────────────────

    public function enterNode(Node $node): ?int
    {
        // Track current function scope
        if ($node instanceof Node\Stmt\Function_) {
            $this->handleFunctionDef($node);
        }

        // Function calls
        if ($node instanceof Node\Expr\FuncCall && $node->name instanceof Node\Name) {
            $this->handleFuncCall($node);
        }

        // Method calls on objects: $pdo->query(...), $stmt->execute(...)
        if ($node instanceof Node\Expr\MethodCall) {
            $this->handleMethodCall($node);
        }

        // include / require / include_once / require_once
        if ($node instanceof Node\Expr\Include_) {
            $this->handleInclude($node);
        }

        // global $var declarations
        if ($node instanceof Node\Stmt\Global_) {
            $this->handleGlobal($node);
        }

        // Variable assignment: $sql = "SELECT ..." — store for later resolution
        // Handles both simple assign ($sql = "...") and assign-in-expression (if($r = ...))
        if ($node instanceof Node\Expr\Assign) {
            $this->trackVarAssignment($node);
        }

        // Superglobal reads: $_POST['field'], $_SESSION['user'], etc.
        if ($node instanceof Node\Expr\ArrayDimFetch) {
            $this->trackSuperglobalRead($node);
        }

        // Inline HTML mixed with PHP — marks a file as a page-level entry point
        if ($node instanceof Node\Stmt\InlineHTML) {
            $this->isHtmlPage = true;
        }

        return null;
    }

    public function leaveNode(Node $node): void
    {
        if ($node instanceof Node\Stmt\Function_) {
            $this->currentFunction = null;
        }
    }

    // ── Function Definition ──────────────────────────────────────────────────

    private function handleFunctionDef(Node\Stmt\Function_ $node): void
    {
        $name = $node->name->toString();
        $this->currentFunction = $name;

        $params = [];
        foreach ($node->params as $param) {
            $params[] = [
                'name'    => '$' . $param->var->name,
                'type'    => $param->type ? $this->typeToString($param->type) : null,
                'default' => $param->default ? $this->scalarToString($param->default) : null,
            ];
        }

        $this->functions[$name] = [
            'name'        => $name,
            'file'        => $this->currentFile,
            'line'        => $node->getStartLine(),
            'params'      => $params,
            'return_type' => $node->returnType ? $this->typeToString($node->returnType) : null,
            'docblock'    => $this->extractDocSummary($node),
        ];
    }

    // ── Function Call ────────────────────────────────────────────────────────

    private function handleFuncCall(Node\Expr\FuncCall $node): void
    {
        $callee = $node->name->toString();
        $caller = $this->currentFunction ?? 'GLOBAL_SCRIPT';

        // Page indicator
        if (in_array($callee, self::PAGE_INDICATORS, true)) {
            $this->isHtmlPage = true;
        }

        // header("Location: ...") redirect
        if ($callee === 'header' && !empty($node->args[0])) {
            $this->handleHeaderCall($node, $caller);
        }

        // Direct SQL via procedural functions
        if (in_array($callee, self::SQL_FUNCTIONS, true)) {
            $this->handleProceduralSql($node, $callee, $caller);
        }

        // Session tracking
        if (in_array($callee, self::SESSION_FUNCTIONS, true)) {
            $this->calls[] = [
                'type'   => 'session',
                'callee' => $callee,
                'caller' => $caller,
                'file'   => $this->currentFile,
                'line'   => $node->getStartLine(),
            ];
        }

    // General function call record
    $this->calls[] = [
            'type'   => 'function_call',
            'callee' => $callee,
            'caller' => $caller,
            'file'   => $this->currentFile,
            'line'   => $node->getStartLine(),
        ];

        // ── Call Graph edge (skip built-ins and trivial noise) ────────────────
        // We only record user-defined function calls. Built-ins (echo, isset, etc.)
        // are filtered in the post-processing step using the known functions list.
        $this->callEdges[] = [
            'caller' => $caller,
            'callee' => $callee,
            'file'   => $this->currentFile,
            'line'   => $node->getStartLine(),
        ];

        // ── Env var extraction: getenv('KEY') ─────────────────────────────────
        if ($callee === 'getenv' && !empty($node->args[0])) {
            $arg = $node->args[0]->value ?? $node->args[0];
            if ($arg instanceof Node\Arg) $arg = $arg->value;
            if ($arg instanceof Node\Scalar\String_) {
                $this->envVars[] = [
                    'key'     => $arg->value,
                    'source'  => 'getenv',
                    'default' => null,
                    'file'    => $this->currentFile,
                    'line'    => $node->getStartLine(),
                ];
            }
        }
    }

    // ── OOP Method Call (PDO / mysqli OO) ────────────────────────────────────

    private function handleMethodCall(Node\Expr\MethodCall $node): void
    {
        $method = $node->name instanceof Node\Identifier ? $node->name->toString() : null;
        if (!$method) return;

        $caller = $this->currentFunction ?? 'GLOBAL_SCRIPT';

        // $pdo->query($sql) / $pdo->prepare($sql) / $pdo->exec($sql)
        // Also resolves variable args via varSqlMap
        if (in_array($method, ['query', 'prepare', 'exec'], true) && !empty($node->args[0])) {
            $sqlArg = $node->args[0]->value ?? $node->args[0];
            if ($sqlArg instanceof Node\Arg) $sqlArg = $sqlArg->value;
            $sqlInfo = $this->parseSqlString($sqlArg);
            if ($sqlInfo) {
                $this->sqlQueries[] = array_merge($sqlInfo, [
                    'caller'   => $caller,
                    'via'      => 'pdo_method',
                    'method'   => $method,
                    'file'     => $this->currentFile,
                    'line'     => $node->getStartLine(),
                ]);
            }
        }
    }

    // ── header() Redirect ────────────────────────────────────────────────────

    private function handleHeaderCall(Node\Expr\FuncCall $node, string $caller): void
    {
        $arg = $node->args[0]->value ?? $node->args[0];
        if ($arg instanceof Node\Arg) $arg = $arg->value;
        if (!($arg instanceof Node\Scalar\String_)) return;

        $headerVal = $arg->value;
        if (!str_contains(strtolower($headerVal), 'location:')) return;

        $target = trim(preg_replace('/^location:\s*/i', '', $headerVal));

        $this->redirects[] = [
            'target' => $target,
            'caller' => $caller,
            'file'   => $this->currentFile,
            'line'   => $node->getStartLine(),
        ];
    }

    // ── Procedural SQL ───────────────────────────────────────────────────────

    private function handleProceduralSql(Node\Expr\FuncCall $node, string $callee, string $caller): void
    {
        // mysqli_query($conn, $sql) — SQL is arg index 1
        // mysql_query($sql)         — SQL is arg index 0
        $sqlArgIndex = str_starts_with($callee, 'mysqli_') ? 1 : 0;

        if (empty($node->args[$sqlArgIndex])) return;
        $sqlArg = $node->args[$sqlArgIndex]->value ?? $node->args[$sqlArgIndex];
        if ($sqlArg instanceof Node\Arg) $sqlArg = $sqlArg->value;

        // extractSqlFromNode handles: string literals, interpolated strings,
        // concat expressions, AND variable references ($sql, $query, $q, etc.)
        $sqlInfo = $this->parseSqlString($sqlArg);
        if ($sqlInfo) {
            $this->sqlQueries[] = array_merge($sqlInfo, [
                'caller'  => $caller,
                'via'     => $callee,
                'file'    => $this->currentFile,
                'line'    => $node->getStartLine(),
            ]);
        }
    }

    // ── include / require ────────────────────────────────────────────────────

    private function handleInclude(Node\Expr\Include_ $node): void
    {
        $typeMap = [
            Node\Expr\Include_::TYPE_INCLUDE      => 'include',
            Node\Expr\Include_::TYPE_INCLUDE_ONCE => 'include_once',
            Node\Expr\Include_::TYPE_REQUIRE      => 'require',
            Node\Expr\Include_::TYPE_REQUIRE_ONCE => 'require_once',
        ];

        $includeType = $typeMap[$node->type] ?? 'include';
        $target      = null;

        if ($node->expr instanceof Node\Scalar\String_) {
            $target = $node->expr->value;
        } elseif ($node->expr instanceof Node\Expr\BinaryOp\Concat) {
            // e.g. dirname(__FILE__) . '/header.php'  — capture what we can
            $target = $this->concatToString($node->expr);
        }

        $this->includes[] = [
            'type'   => $includeType,
            'target' => $target ?? '(dynamic)',
            'caller' => $this->currentFunction ?? 'GLOBAL_SCRIPT',
            'file'   => $this->currentFile,
            'line'   => $node->getStartLine(),
        ];
    }

    // ── Global Variables ─────────────────────────────────────────────────────

    private function handleGlobal(Node\Stmt\Global_ $node): void
    {
        foreach ($node->vars as $var) {
            if ($var instanceof Node\Expr\Variable && is_string($var->name)) {
                $this->globals[] = [
                    'var'    => '$' . $var->name,
                    'scope'  => $this->currentFunction ?? 'GLOBAL_SCRIPT',
                    'file'   => $this->currentFile,
                    'line'   => $node->getStartLine(),
                ];
            }
        }
    }

    // ── Variable Assignment Tracker ─────────────────────────────────────────

    /**
     * When we see $sql = "SELECT ..." or $query = "INSERT ...", store the
     * resolved SQL string in $varSqlMap keyed by variable name.
     * This lets handleProceduralSql() resolve $var → SQL at the call site.
     */
    private function trackVarAssignment(Node\Expr\Assign $node): void
    {
        // LHS must be a simple variable: $sql, $query, $q, $sq, etc.
        if (!($node->var instanceof Node\Expr\Variable)) return;
        if (!is_string($node->var->name)) return;

        $varName = $node->var->name;
        $sql     = $this->extractSqlFromNode($node->expr);

        if ($sql !== null) {
            $this->varSqlMap[$varName] = $sql;
        }
    }

    /**
     * Try to extract a SQL string from any node type:
     *   - String literal:      "SELECT * FROM users"
     *   - Interpolated string: "SELECT * FROM users WHERE id='$id'"
     *   - Concat expression:   "SELECT " . $cols . " FROM users"
     *   - Variable reference:  $sql (look up in varSqlMap)
     *
     * Returns the raw SQL string, or null if it cannot be resolved.
     */
    private function extractSqlFromNode(Node $node): ?string
    {
        // Plain string literal
        if ($node instanceof Node\Scalar\String_) {
            return $node->value;
        }

        // Interpolated string: "SELECT * FROM t WHERE id='$x'"
        // nikic v5: InterpolatedString — reconstruct the static parts
        if ($node instanceof Node\Scalar\InterpolatedString) {
            return $this->interpolatedToString($node);
        }

        // nikic v4 compatibility: Encapsed
        if ($node instanceof Node\Scalar\Encapsed) {
            return $this->interpolatedToString($node);
        }

        // Concatenation: "SELECT " . $cols . " FROM users"
        if ($node instanceof Node\Expr\BinaryOp\Concat) {
            return $this->concatToString($node);
        }

        // Variable reference: resolve from our map
        if ($node instanceof Node\Expr\Variable && is_string($node->name)) {
            return $this->varSqlMap[$node->name] ?? null;
        }

        return null;
    }

    /**
     * Reconstruct the static text of an interpolated string node.
     * Variable parts (like $id) are replaced with '?' placeholders so the
     * SQL structure (SELECT/FROM/WHERE) is preserved for table extraction.
     */
    private function interpolatedToString(Node $node): string
    {
        $parts = [];
        $items = $node->parts ?? [];
        foreach ($items as $part) {
            if ($part instanceof Node\Scalar\String_) {
                $parts[] = $part->value;
            } elseif ($part instanceof Node\InterpolatedStringPart) {
                // nikic v5 static part inside interpolated string
                $parts[] = $part->value;
            } else {
                // Variable or expression interpolation — use placeholder
                $parts[] = '?';
            }
        }
        return implode('', $parts);
    }

    // ── SQL Parsing Helpers ───────────────────────────────────────────────────

    /**
     * Given an AST node that might be a SQL string (or string concat),
     * extracts operation (SELECT/INSERT/UPDATE/DELETE) and table name.
     * Now also resolves variable references via varSqlMap.
     * Returns null if the node is not a recognisable SQL literal.
     */
    private function parseSqlString(Node $node): ?array
    {
        $sql = $this->extractSqlFromNode($node);

        if (!$sql) return null;

        if (!preg_match('/\b(SELECT|INSERT|UPDATE|DELETE|REPLACE|TRUNCATE|CREATE|DROP|ALTER)\b/i', $sql, $opMatch)) {
            return null;
        }

        $operation = strtoupper($opMatch[1]);

        // Extract table name from common SQL patterns
        $table = 'UNKNOWN';
        $patterns = [
            '/\bFROM\s+`?([a-zA-Z0-9_]+)`?/i',
            '/\bINTO\s+`?([a-zA-Z0-9_]+)`?/i',
            '/\bUPDATE\s+`?([a-zA-Z0-9_]+)`?/i',
            '/\bJOIN\s+`?([a-zA-Z0-9_]+)`?/i',
            '/\bTABLE\s+`?([a-zA-Z0-9_]+)`?/i',
        ];
        foreach ($patterns as $pattern) {
            if (preg_match($pattern, $sql, $tMatch)) {
                $table = $tMatch[1];
                break;
            }
        }

        // Capture full SQL (truncated for storage)
        $sqlPreview = strlen($sql) > 200 ? substr($sql, 0, 200) . '...' : $sql;

        return [
            'operation' => $operation,
            'table'     => $table,
            'sql'       => $sqlPreview,
        ];
    }

    private function concatToString(Node\Expr\BinaryOp\Concat $node): string
    {
        $left  = $node->left  instanceof Node\Scalar\String_            ? $node->left->value
               : ($node->left  instanceof Node\Expr\BinaryOp\Concat     ? $this->concatToString($node->left)  : '?');
        $right = $node->right instanceof Node\Scalar\String_            ? $node->right->value
               : ($node->right instanceof Node\Expr\BinaryOp\Concat     ? $this->concatToString($node->right) : '?');
        return $left . $right;
    }

    // ── Misc Helpers ─────────────────────────────────────────────────────────

    private function typeToString(Node $type): string
    {
        if ($type instanceof Node\Identifier)   return $type->toString();
        if ($type instanceof Node\Name)         return $type->toString();
        if ($type instanceof Node\NullableType) return '?' . $this->typeToString($type->type);
        if ($type instanceof Node\UnionType) {
            return implode('|', array_map(fn($t) => $this->typeToString($t), $type->types));
        }
        return 'mixed';
    }

    private function scalarToString(Node $node): string
    {
        if ($node instanceof Node\Scalar\String_)  return $node->value;
        if ($node instanceof Node\Scalar\Int_)     return (string)$node->value;
        if ($node instanceof Node\Scalar\Float_)   return (string)$node->value;
        if ($node instanceof Node\Expr\ConstFetch) return $node->name->toString();
        return '...';
    }

    // ── Superglobal Tracking ─────────────────────────────────────────────────

    /**
     * Detects $_POST['key'], $_GET['key'], $_SESSION['key'] array access nodes.
     * Records both the superglobal name and the accessed key (if it is a string).
     */
    private function trackSuperglobalRead(Node\Expr\ArrayDimFetch $node): void
    {
        // The var part must be a variable whose name is a superglobal
        if (!($node->var instanceof Node\Expr\Variable)) return;
        if (!is_string($node->var->name)) return;

        $varName = $node->var->name; // e.g. "_POST", "_SESSION"
        if (!in_array($varName, self::SUPERGLOBALS, true)) return;

        // Extract the key if it is a string literal
        $key = null;
        if ($node->dim instanceof Node\Scalar\String_) {
            $key = $node->dim->value;
        }

        $this->superglobalReads[] = [
            'var'    => '$' . $varName,
            'key'    => $key,
            'caller' => $this->currentFunction ?? 'GLOBAL_SCRIPT',
            'file'   => $this->currentFile,
            'line'   => $node->getStartLine(),
        ];

        // Capture $_ENV['KEY'] reads as env var references
        if ($varName === '_ENV' && $key !== null) {
            $this->envVars[] = [
                'key'     => $key,
                'source'  => '$_ENV',
                'default' => null,
                'file'    => $this->currentFile,
                'line'    => $node->getStartLine(),
            ];
        }
    }

    private function extractDocSummary(Node $node): ?string
    {
        $doc = $node->getDocComment();
        if (!$doc) return null;
        $lines = explode("\n", $doc->getText());
        foreach ($lines as $line) {
            $line = trim($line, " \t\r\n*\/");
            if ($line !== '' && !str_starts_with($line, '@')) return $line;
        }
        return null;
    }
}

// ─── Migration / DB Schema Extractor ─────────────────────────────────────────

/**
 * Parses Laravel migration files to extract table operations and column
 * definitions so Stage 1 can populate CodeMap::db_schema.
 */
class MigrationVisitor extends NodeVisitorAbstract
{
    public array $schema = [];
    private string $file;

    public function __construct(string $file)
    {
        $this->file = $file;
    }

    public function enterNode(Node $node): ?int
    {
        // Schema::create('table', function (Blueprint $table) { ... })
        if (
            $node instanceof Node\Expr\StaticCall &&
            $node->class instanceof Node\Name &&
            strtolower($node->class->toString()) === 'schema'
        ) {
            $method = $node->name instanceof Node\Identifier ? $node->name->toString() : null;
            if (!in_array($method, ['create', 'table', 'drop', 'dropIfExists', 'rename'], true)) {
                return null;
            }

            $tableName = null;
            if (!empty($node->args[0])) {
                $arg = $node->args[0]->value ?? $node->args[0];
                if ($arg instanceof Node\Scalar\String_) {
                    $tableName = $arg->value;
                }
            }

            $columns = [];
            // Walk into the closure body to find Blueprint method calls
            if (!empty($node->args[1])) {
                $closure = $node->args[1]->value ?? $node->args[1];
                if ($closure instanceof Node\Arg) $closure = $closure->value;
                if ($closure instanceof Node\Expr\Closure || $closure instanceof Node\Expr\ArrowFunction) {
                    $stmts = $closure->stmts ?? [];
                    $columns = $this->extractColumns($stmts);
                }
            }

            $this->schema[] = [
                'operation' => $method,
                'table'     => $tableName,
                'columns'   => $columns,
                'file'      => $this->file,
                'line'      => $node->getStartLine(),
            ];
        }

        return null;
    }

    private function extractColumns(array $stmts): array
    {
        $columns = [];
        foreach ($stmts as $stmt) {
            // Unwrap expression statements
            $expr = ($stmt instanceof Node\Stmt\Expression) ? $stmt->expr : null;
            if (!$expr) continue;

            // Walk chained method calls: $table->string('name')->nullable()
            $chain = $this->unwindChain($expr);
            if (empty($chain)) continue;

            $base = $chain[0]; // first call = column type
            if (!($base instanceof Node\Expr\MethodCall)) continue;

            $colType = $base->name instanceof Node\Identifier ? $base->name->toString() : null;
            if (!$colType) continue;

            // Skip index/constraint helpers
            $skipTypes = ['primary','unique','index','foreign','timestamps','softDeletes','rememberToken','morphs'];
            if (in_array($colType, $skipTypes, true)) continue;

            $colName = null;
            if (!empty($base->args[0])) {
                $arg = $base->args[0]->value ?? $base->args[0];
                if ($arg instanceof Node\Arg) $arg = $arg->value;
                if ($arg instanceof Node\Scalar\String_) $colName = $arg->value;
            }

            // Collect modifiers from chained calls
            $modifiers = [];
            foreach (array_slice($chain, 1) as $chained) {
                if ($chained instanceof Node\Expr\MethodCall) {
                    $mod = $chained->name instanceof Node\Identifier ? $chained->name->toString() : null;
                    if ($mod) $modifiers[] = $mod;
                }
            }

            $columns[] = [
                'name'      => $colName,
                'type'      => $colType,
                'modifiers' => $modifiers,
            ];
        }
        return $columns;
    }

    /** Unwraps $a->b()->c() into [$a->b(), $a->b()->c()] from innermost outward */
    private function unwindChain(Node $node): array
    {
        $chain = [];
        $current = $node;
        while ($current instanceof Node\Expr\MethodCall) {
            array_unshift($chain, $current);
            $current = $current->var;
        }
        return $chain;
    }
}

// ─── Symfony Route Extractor ──────────────────────────────────────────────────

/**
 * Handles Symfony-style PHP 8 attribute routes (#[Route(...)]) and
 * annotation-style @Route in docblocks. Runs as a second pass over classes.
 */
function extractSymfonyRoutes(array &$classes, string $framework): array
{
    if ($framework !== 'symfony') return [];

    $routes = [];
    foreach ($classes as $classInfo) {
        $classPrefix = '';

        // Class-level route prefix from annotations
        if (!empty($classInfo['docblock']['annotations'])) {
            foreach ($classInfo['docblock']['annotations'] as $ann) {
                if (str_ends_with($ann['name'], 'Route')) {
                    preg_match('/["\']([^"\']+)["\']/', $ann['params'], $m);
                    $classPrefix = $m[1] ?? '';
                }
            }
        }

        foreach ($classInfo['methods'] ?? [] as $method) {
            foreach ($method['docblock']['annotations'] ?? [] as $ann) {
                if (!str_ends_with($ann['name'], 'Route')) continue;
                preg_match('/["\']([^"\']+)["\']/', $ann['params'], $m);
                $path = ($classPrefix . ($m[1] ?? '')) ?: null;
                if (!$path) continue;

                // Extract HTTP methods from annotation
                preg_match('/methods\s*=\s*[{"\']([^}"\']+)[}"\']/', $ann['params'], $hm);
                $httpMethod = isset($hm[1]) ? strtoupper($hm[1]) : 'ANY';

                $routes[] = [
                    'method'  => $httpMethod,
                    'path'    => $path,
                    'handler' => $classInfo['fqn'] . '::' . $method['name'],
                    'file'    => $classInfo['file'],
                    'line'    => $method['line'],
                    'source'  => 'symfony_annotation',
                ];
            }
        }
    }
    return $routes;
}

// ─── WordPress Hook Extractor ─────────────────────────────────────────────────

/**
 * WordPress doesn't use MVC routes — instead it uses add_action / add_filter.
 * We capture these as pseudo-routes for the BA stage to interpret.
 */
class WordPressHookVisitor extends NodeVisitorAbstract
{
    public array $hooks = [];
    private string $file;

    public function __construct(string $file)
    {
        $this->file = $file;
    }

    public function enterNode(Node $node): ?int
    {
        if (!($node instanceof Node\Expr\FuncCall)) return null;
        if (!($node->name instanceof Node\Name))   return null;

        $fn = $node->name->toString();
        if (!in_array($fn, ['add_action', 'add_filter', 'do_action', 'apply_filters'], true)) {
            return null;
        }

        $hookName = null;
        if (!empty($node->args[0])) {
            $arg = $node->args[0]->value ?? $node->args[0];
            if ($arg instanceof Node\Arg) $arg = $arg->value;
            if ($arg instanceof Node\Scalar\String_) $hookName = $arg->value;
        }

        $callback = null;
        if (!empty($node->args[1])) {
            $arg = $node->args[1]->value ?? $node->args[1];
            if ($arg instanceof Node\Arg) $arg = $arg->value;
            if ($arg instanceof Node\Scalar\String_)   $callback = $arg->value;
            if ($arg instanceof Node\Expr\Array_ && count($arg->items) === 2) {
                $cls = $arg->items[0]->value ?? null;
                $mth = $arg->items[1]->value ?? null;
                if ($cls && $mth) {
                    $clsStr = ($cls instanceof Node\Expr\Variable) ? '$' . $cls->name : '...';
                    $mthStr = ($mth instanceof Node\Scalar\String_) ? $mth->value : '...';
                    $callback = $clsStr . '::' . $mthStr;
                }
            }
        }

        $this->hooks[] = [
            'hook_type' => $fn,
            'hook_name' => $hookName,
            'callback'  => $callback,
            'file'      => $this->file,
            'line'      => $node->getStartLine(),
        ];

        return null;
    }
}

// ─── Form Field Extractor ─────────────────────────────────────────────────────

/**
 * Extracts HTML <form> fields from PHP files that mix PHP and HTML.
 *
 * Detects:
 *   - InlineHTML nodes containing <form ...> and <input name="...">
 *   - echo/print statements rendering <input name="..."> tags
 *   - The form's action attribute and method (GET/POST)
 *
 * Output per form:
 *   { file, action, method, fields: [ {name, type, id}, ... ] }
 */
class FormFieldVisitor extends NodeVisitorAbstract
{
    public array $forms = [];
    private string $file;
    private array $pendingForm = [];

    public function __construct(string $file)
    {
        $this->file = $file;
    }

    public function enterNode(Node $node): ?int
    {
        // Inline HTML blocks: <form action="..." method="POST"> ... <input ...>
        if ($node instanceof Node\Stmt\InlineHTML) {
            $this->parseHtmlChunk($node->value, $node->getStartLine());
        }

        // echo/print rendering HTML: echo "<input name='field' ...>"
        if ($node instanceof Node\Expr\FuncCall || $node instanceof Node\Stmt\Echo_) {
            $this->scanEchoForInputs($node);
        }

        return null;
    }

    private function parseHtmlChunk(string $html, int $startLine): void
    {
        // Extract <form ...> openings
        if (preg_match_all('/<form\b([^>]*)>/i', $html, $formMatches)) {
            foreach ($formMatches[1] as $attrs) {
                $action = '';
                $method = 'GET';
                if (preg_match('/action\s*=\s*["\']([^"\']*)["\']|action\s*=\s*(\S+)/i', $attrs, $am)) {
                    $action = $am[1] ?: $am[2] ?? '';
                }
                if (preg_match('/method\s*=\s*["\']([^"\']*)["\']|method\s*=\s*(\S+)/i', $attrs, $mm)) {
                    $method = strtoupper($mm[1] ?: $mm[2] ?? 'GET');
                }
                $this->pendingForm = [
                    'file'   => $this->file,
                    'action' => $action,
                    'method' => $method,
                    'line'   => $startLine,
                    'fields' => [],
                ];
            }
        }

        // Extract <input ...>, <select ...>, <textarea ...> fields
        $fieldPatterns = [
            '/<input\b([^>]*)>/i',
            '/<select\b([^>]*)>/i',
            '/<textarea\b([^>]*)>/i',
        ];
        foreach ($fieldPatterns as $pat) {
            if (preg_match_all($pat, $html, $inputMatches)) {
                foreach ($inputMatches[1] as $attrs) {
                    $field = $this->parseFieldAttrs($attrs);
                    if ($field['name'] !== null) {
                        if (!empty($this->pendingForm)) {
                            $this->pendingForm['fields'][] = $field;
                        } else {
                            // Field outside an explicit <form> — attach to file-level form
                            $this->pendingForm = [
                                'file'   => $this->file,
                                'action' => '',
                                'method' => 'POST',
                                'line'   => $startLine,
                                'fields' => [$field],
                            ];
                        }
                    }
                }
            }
        }

        // Close form
        if (str_contains(strtolower($html), '</form>') && !empty($this->pendingForm)) {
            if (!empty($this->pendingForm['fields'])) {
                $this->forms[] = $this->pendingForm;
            }
            $this->pendingForm = [];
        }
    }

    private function parseFieldAttrs(string $attrs): array
    {
        $name = null;
        $type = 'text';
        $id   = null;
        if (preg_match('/\bname\s*=\s*["\']([^"\']*)["\']|name\s*=\s*(\S+)/i', $attrs, $m)) {
            $name = trim($m[1] ?: $m[2] ?? '', '"\'');
        }
        if (preg_match('/\btype\s*=\s*["\']([^"\']*)["\']|type\s*=\s*(\S+)/i', $attrs, $m)) {
            $type = strtolower(trim($m[1] ?: $m[2] ?? 'text', '"\''));
        }
        if (preg_match('/\bid\s*=\s*["\']([^"\']*)["\']|id\s*=\s*(\S+)/i', $attrs, $m)) {
            $id = trim($m[1] ?: $m[2] ?? '', '"\'');
        }
        return ['name' => $name, 'type' => $type, 'id' => $id];
    }

    private function scanEchoForInputs(Node $node): void
    {
        // Collect string expressions from echo/print/function calls
        $strs = [];
        if ($node instanceof Node\Stmt\Echo_) {
            foreach ($node->exprs as $expr) {
                if ($expr instanceof Node\Scalar\String_) $strs[] = $expr->value;
                if ($expr instanceof Node\Scalar\InterpolatedString ||
                    $expr instanceof Node\Scalar\Encapsed) {
                    $strs[] = $this->encapsedToText($expr);
                }
            }
        }
        foreach ($strs as $s) {
            if (preg_match('/<input|<form|<select|<textarea/i', $s)) {
                $this->parseHtmlChunk($s, $node->getStartLine());
            }
        }
    }

    private function encapsedToText(Node $node): string
    {
        $out = '';
        foreach ($node->parts ?? [] as $part) {
            if ($part instanceof Node\Scalar\String_)     $out .= $part->value;
            elseif ($part instanceof Node\InterpolatedStringPart) $out .= $part->value;
            else                                                    $out .= '?';
        }
        return $out;
    }
}


// ─── Service Dependency Extractor ─────────────────────────────────────────────

/**
 * Extracts constructor-injected dependencies from OOP classes.
 *
 * Detects type-hinted constructor parameters — the standard DI pattern
 * across Laravel, Symfony, and any PSR-11 container:
 *
 *   class OrderController {
 *       public function __construct(
 *           private OrderService $orderService,
 *           private readonly PaymentGateway $gateway,
 *       ) {}
 *   }
 *
 * Output per dependency:
 *   { class: "OrderController", dep_class: "OrderService",
 *     dep_var: "$orderService", file: "...", line: N }
 *
 * Also captures Laravel Facades and static service locators:
 *   Cache::get(...) → { class: "GLOBAL", dep_class: "Cache", dep_var: null }
 */
class ServiceDependencyVisitor extends NodeVisitorAbstract
{
    public array $deps = [];
    private string $file;
    private ?string $currentClass = null;

    public function __construct(string $file)
    {
        $this->file = $file;
    }

    public function enterNode(Node $node): ?int
    {
        // Track current class context
        if ($node instanceof Node\Stmt\Class_) {
            $this->currentClass = $node->name ? $node->name->toString() : null;
        }

        // Constructor injection: public function __construct(TypeHint $var)
        if ($node instanceof Node\Stmt\ClassMethod) {
            if ($node->name->toString() === '__construct' && $this->currentClass) {
                foreach ($node->params as $param) {
                    // Only capture type-hinted parameters (not scalars like string/int)
                    if (!$param->type) continue;
                    $typeStr = $this->typeToString($param->type);
                    // Skip scalar types — we only want class/interface dependencies
                    if (in_array(strtolower($typeStr),
                        ['string','int','float','bool','array','callable','iterable',
                         'mixed','void','never','null','object'], true)) {
                        continue;
                    }
                    // Skip nullable primitives (?string etc.)
                    if (str_starts_with($typeStr, '?')) {
                        $inner = ltrim($typeStr, '?');
                        if (in_array(strtolower($inner),
                            ['string','int','float','bool','array'], true)) continue;
                    }

                    $varName = '$' . $param->var->name;
                    $this->deps[] = [
                        'class'     => $this->currentClass,
                        'dep_class' => ltrim($typeStr, '?\\'),
                        'dep_var'   => $varName,
                        'file'      => $this->file,
                        'line'      => $node->getStartLine(),
                    ];
                }
            }
        }

        // Laravel Facade static calls: Cache::get(), Auth::user(), etc.
        if ($node instanceof Node\Expr\StaticCall) {
            if ($node->class instanceof Node\Name) {
                $facade = $node->class->toString();
                // Filter: only Laravel-style PascalCase short names (not Schema::, Route:: etc.)
                $knownFacades = [
                    'Auth','Cache','DB','Event','Gate','Hash','Http','Log',
                    'Mail','Notification','Queue','Redis','Session','Storage',
                    'Validator','View','Crypt','Config','Cookie','Broadcast',
                ];
                if (in_array($facade, $knownFacades, true)) {
                    $this->deps[] = [
                        'class'     => $this->currentClass ?? 'GLOBAL',
                        'dep_class' => $facade . ' (Facade)',
                        'dep_var'   => null,
                        'file'      => $this->file,
                        'line'      => $node->getStartLine(),
                    ];
                }
            }
        }

        return null;
    }

    public function leaveNode(Node $node): void
    {
        if ($node instanceof Node\Stmt\Class_) {
            $this->currentClass = null;
        }
    }

    private function typeToString(Node $type): string
    {
        if ($type instanceof Node\Name)            return $type->toString();
        if ($type instanceof Node\Identifier)      return $type->toString();
        if ($type instanceof Node\NullableType)    return '?' . $this->typeToString($type->type);
        if ($type instanceof Node\UnionType) {
            return implode('|', array_map([$this, 'typeToString'], $type->types));
        }
        if ($type instanceof Node\IntersectionType) {
            return implode('&', array_map([$this, 'typeToString'], $type->types));
        }
        return 'mixed';
    }
}


// ─── Laravel env() / Symfony getenv() Visitor ────────────────────────────────

/**
 * Extracts all env('KEY', $default) and getenv('KEY') calls across all files.
 *
 * This complements the getenv() tracking in ProceduralVisitor by handling
 * the Laravel-specific env() helper and dotenv patterns used in config files.
 *
 * Output per env var:
 *   { key: "DB_HOST", default: "localhost", source: "env()", file: "...", line: N }
 */
class EnvVarVisitor extends NodeVisitorAbstract
{
    public array $envVars = [];
    private string $file;

    public function __construct(string $file)
    {
        $this->file = $file;
    }

    public function enterNode(Node $node): ?int
    {
        if (!($node instanceof Node\Expr\FuncCall)) return null;
        if (!($node->name instanceof Node\Name))    return null;

        $fn = strtolower($node->name->toString());

        // Laravel env('KEY', $default) or env('KEY')
        if ($fn === 'env') {
            [$key, $default] = $this->extractKeyDefault($node->args);
            if ($key !== null) {
                $this->envVars[] = [
                    'key'     => $key,
                    'default' => $default,
                    'source'  => 'env()',
                    'file'    => $this->file,
                    'line'    => $node->getStartLine(),
                ];
            }
        }

        // getenv('KEY')
        if ($fn === 'getenv') {
            [$key, ] = $this->extractKeyDefault($node->args);
            if ($key !== null) {
                $this->envVars[] = [
                    'key'     => $key,
                    'default' => null,
                    'source'  => 'getenv()',
                    'file'    => $this->file,
                    'line'    => $node->getStartLine(),
                ];
            }
        }

        return null;
    }

    private function extractKeyDefault(array $args): array
    {
        $key     = null;
        $default = null;

        if (!empty($args[0])) {
            $arg = $args[0]->value ?? $args[0];
            if ($arg instanceof Node\Arg) $arg = $arg->value;
            if ($arg instanceof Node\Scalar\String_) $key = $arg->value;
        }

        if (!empty($args[1])) {
            $arg = $args[1]->value ?? $args[1];
            if ($arg instanceof Node\Arg) $arg = $arg->value;
            if ($arg instanceof Node\Scalar\String_)  $default = $arg->value;
            elseif ($arg instanceof Node\Scalar\LNumber) $default = $arg->value;
            elseif ($arg instanceof Node\Scalar\DNumber) $default = $arg->value;
            elseif ($arg instanceof Node\Expr\ConstFetch) {
                $name = $arg->name->toString();
                $default = match(strtolower($name)) {
                    'true'  => true,
                    'false' => false,
                    'null'  => null,
                    default => $name,
                };
            }
        }

        return [$key, $default];
    }
}


// ─── Authentication Logic Visitor ────────────────────────────────────────────

/**
 * Extracts authentication and authorisation signals from PHP files.
 *
 * Detects all major PHP auth patterns across frameworks:
 *
 *   Raw PHP     : password_verify(), session_start(), $_SESSION['user']
 *   Laravel     : Auth::attempt(), Auth::check(), Auth::user(), Hash::check()
 *                 auth()->attempt(), auth()->check(), middleware('auth')
 *   Symfony     : $this->getUser(), $this->denyAccessUnlessGranted(),
 *                 isGranted(), UserPasswordHasherInterface
 *   WordPress   : wp_check_password(), is_user_logged_in(), current_user_can()
 *
 * Output per signal:
 *   { type: 'attempt'|'check'|'hash_verify'|'session_auth'|'guard'|'logout'|'role_check',
 *     pattern: 'Auth::attempt'|'password_verify'|...,
 *     detail: '...',
 *     file: '...', line: N }
 */
class AuthLogicVisitor extends NodeVisitorAbstract
{
    public array $authSignals = [];

    private string  $currentFile;
    private ?string $currentFunction = null;

    // Laravel/Symfony facades and helpers we recognise as auth-related
    private const LARAVEL_AUTH_STATIC = [
        'Auth'  => ['attempt','check','user','guest','login','logout','id',
                    'loginUsingId','once','viaRemember'],
        'Hash'  => ['check','make','needsRehash'],
        'Gate'  => ['allows','denies','check','any','none','authorize','inspect'],
    ];

    // Raw PHP auth functions
    private const RAW_AUTH_FUNCTIONS = [
        'password_verify'       => 'hash_verify',
        'password_hash'         => 'hash_make',
        'session_start'         => 'session_start',
        'session_destroy'       => 'logout',
        'session_regenerate_id' => 'session_regen',
    ];

    // WordPress auth functions
    private const WP_AUTH_FUNCTIONS = [
        'wp_check_password'     => 'hash_verify',
        'wp_set_password'       => 'hash_make',
        'is_user_logged_in'     => 'check',
        'current_user_can'      => 'role_check',
        'wp_login'              => 'attempt',
        'wp_logout'             => 'logout',
        'auth_redirect'         => 'guard',
    ];

    public function __construct(string $file)
    {
        $this->currentFile = $file;
    }

    public function enterNode(Node $node): ?int
    {
        // Track function scope for context
        if ($node instanceof Node\Stmt\Function_ || $node instanceof Node\Stmt\ClassMethod) {
            $this->currentFunction = $node->name->toString();
        }

        // ── Static calls: Auth::attempt(), Hash::check(), Gate::allows() ────
        if ($node instanceof Node\Expr\StaticCall) {
            if ($node->class instanceof Node\Name) {
                $class  = $node->class->toString();
                $method = $node->name instanceof Node\Identifier
                    ? $node->name->toString() : null;

                if ($method && isset(self::LARAVEL_AUTH_STATIC[$class])) {
                    if (in_array($method, self::LARAVEL_AUTH_STATIC[$class], true)) {
                        $type = match(true) {
                            in_array($method, ['attempt','login','loginUsingId','once'], true) => 'attempt',
                            in_array($method, ['check','guest'],                         true) => 'check',
                            in_array($method, ['logout'],                                true) => 'logout',
                            in_array($method, ['user','id','viaRemember'],               true) => 'get_user',
                            $class === 'Hash'                                                  => 'hash_verify',
                            $class === 'Gate'                                                  => 'role_check',
                            default                                                            => 'auth',
                        };
                        $this->record($type, "{$class}::{$method}", $node->getStartLine());
                    }
                }
            }
        }

        // ── Function calls ───────────────────────────────────────────────────
        if ($node instanceof Node\Expr\FuncCall && $node->name instanceof Node\Name) {
            $fn = $node->name->toString();

            // Raw PHP
            if (isset(self::RAW_AUTH_FUNCTIONS[$fn])) {
                $this->record(self::RAW_AUTH_FUNCTIONS[$fn], $fn, $node->getStartLine());
            }

            // WordPress
            if (isset(self::WP_AUTH_FUNCTIONS[$fn])) {
                $this->record(self::WP_AUTH_FUNCTIONS[$fn], $fn, $node->getStartLine());
            }

            // Laravel auth() helper: auth()->attempt(), auth()->check()
            // Captured at the method-call level below; track the auth() call itself
            if ($fn === 'auth') {
                // Will be completed by the chained MethodCall node
            }
        }

        // ── Method calls: auth()->attempt(), $this->getUser() ───────────────
        if ($node instanceof Node\Expr\MethodCall) {
            $method = $node->name instanceof Node\Identifier
                ? $node->name->toString() : null;
            if (!$method) return null;

            // auth()->attempt() / auth()->check() / auth()->user() etc.
            if ($node->var instanceof Node\Expr\FuncCall &&
                $node->var->name instanceof Node\Name &&
                $node->var->name->toString() === 'auth') {

                $type = match($method) {
                    'attempt'      => 'attempt',
                    'check'        => 'check',
                    'guest'        => 'check',
                    'login'        => 'attempt',
                    'logout'       => 'logout',
                    'user','id'    => 'get_user',
                    default        => 'auth',
                };
                $this->record($type, "auth()->{$method}", $node->getStartLine());
            }

            // Symfony: $this->denyAccessUnlessGranted(), $this->isGranted()
            if ($node->var instanceof Node\Expr\Variable &&
                $node->var->name === 'this') {
                if ($method === 'denyAccessUnlessGranted') {
                    $role = $this->extractStringArg0($node->args);
                    $this->record('role_check', 'denyAccessUnlessGranted',
                        $node->getStartLine(), $role);
                }
                if (in_array($method, ['getUser','getToken'], true)) {
                    $this->record('get_user', "\$this->{$method}",
                        $node->getStartLine());
                }
            }

            // $passwordHasher->verify() — Symfony PasswordHasherInterface
            if ($method === 'verify' || $method === 'isPasswordValid') {
                $this->record('hash_verify', "\$hasher->{$method}",
                    $node->getStartLine());
            }
        }

        // ── $_SESSION['user_id'] / $_SESSION['user'] writes ──────────────────
        if ($node instanceof Node\Expr\Assign) {
            if ($node->var instanceof Node\Expr\ArrayDimFetch) {
                $arrNode = $node->var->var;
                if ($arrNode instanceof Node\Expr\Variable &&
                    is_string($arrNode->name) &&
                    $arrNode->name === '_SESSION') {
                    $key = null;
                    if ($node->var->dim instanceof Node\Scalar\String_) {
                        $key = $node->var->dim->value;
                    }
                    // Only record if key looks auth-related
                    if ($key && preg_match('/user|uid|id|auth|logged|role|admin/i', $key)) {
                        $this->record('session_auth', "\$_SESSION['{$key}'] = ...",
                            $node->getStartLine(), $key);
                    }
                }
            }
        }

        // ── if(isset($_SESSION['user_id'])) auth guards ─────────────────────
        if ($node instanceof Node\Expr\FuncCall &&
            $node->name instanceof Node\Name &&
            $node->name->toString() === 'isset') {
            foreach ($node->args as $arg) {
                $v = $arg->value ?? $arg;
                if ($v instanceof Node\Arg) $v = $v->value;
                if ($v instanceof Node\Expr\ArrayDimFetch &&
                    $v->var instanceof Node\Expr\Variable &&
                    $v->var->name === '_SESSION') {
                    $key = ($v->dim instanceof Node\Scalar\String_)
                        ? $v->dim->value : '?';
                    if (preg_match('/user|uid|id|auth|logged|role|admin/i', $key)) {
                        $this->record('guard', "isset(\$_SESSION['{$key}'])",
                            $node->getStartLine(), $key);
                    }
                }
            }
        }

        return null;
    }

    public function leaveNode(Node $node): void
    {
        if ($node instanceof Node\Stmt\Function_ || $node instanceof Node\Stmt\ClassMethod) {
            $this->currentFunction = null;
        }
    }

    private function record(string $type, string $pattern, int $line, ?string $detail = null): void
    {
        $this->authSignals[] = [
            'type'     => $type,
            'pattern'  => $pattern,
            'detail'   => $detail,
            'function' => $this->currentFunction,
            'file'     => $this->currentFile,
            'line'     => $line,
        ];
    }

    private function extractStringArg0(array $args): ?string
    {
        if (empty($args[0])) return null;
        $v = $args[0]->value ?? $args[0];
        if ($v instanceof Node\Arg) $v = $v->value;
        return ($v instanceof Node\Scalar\String_) ? $v->value : null;
    }
}


// ─── HTTP Entry Point Visitor ─────────────────────────────────────────────────

/**
 * Detects HTTP entry points — files and methods that accept HTTP requests.
 *
 * More precise than the existing html_pages heuristic:
 *
 *   Raw PHP   : Files with $_POST/$_GET + output (echo/header) at top scope
 *   Laravel   : Controller methods with Route registrations or return response()
 *   Symfony   : Controller methods with #[Route] attributes or @Route annotations
 *   Generic   : Methods returning Response|JsonResponse|View, or containing
 *               json_encode + return, or reading $_SERVER['REQUEST_METHOD']
 *
 * Output per entry point:
 *   { file: '...', type: 'page'|'api'|'controller_method'|'form_handler',
 *     accepts: ['GET','POST'], produces: 'html'|'json'|'redirect'|'mixed',
 *     handler: 'ControllerClass::method' or null (for file-level) }
 */
class HttpEntryPointVisitor extends NodeVisitorAbstract
{
    public array $entryPoints = [];

    private string  $currentFile;
    private ?string $currentClass  = null;
    private ?string $currentMethod = null;

    // HTTP method signals per scope
    private array   $scopeSignals  = [];

    public function __construct(string $file)
    {
        $this->currentFile = $file;
    }

    public function enterNode(Node $node): ?int
    {
        if ($node instanceof Node\Stmt\Class_) {
            $this->currentClass = $node->name ? $node->name->toString() : null;
        }

        if ($node instanceof Node\Stmt\ClassMethod) {
            $this->currentMethod = $node->name->toString();
            $this->scopeSignals[$this->scopeKey()] = [];
        }

        // ── REQUEST_METHOD check ─────────────────────────────────────────────
        if ($node instanceof Node\Expr\ArrayDimFetch &&
            $node->var instanceof Node\Expr\Variable &&
            is_string($node->var->name) &&
            $node->var->name === '_SERVER') {
            if ($node->dim instanceof Node\Scalar\String_ &&
                $node->dim->value === 'REQUEST_METHOD') {
                $this->addSignal('request_method_check');
            }
        }

        // ── json_encode return → API endpoint ─────────────────────────────
        if ($node instanceof Node\Expr\FuncCall &&
            $node->name instanceof Node\Name &&
            $node->name->toString() === 'json_encode') {
            $this->addSignal('json_output');
        }

        // ── response()->json() or return response(view(...)) — Laravel ─────
        if ($node instanceof Node\Expr\FuncCall &&
            $node->name instanceof Node\Name) {
            $fn = $node->name->toString();
            if ($fn === 'response')  $this->addSignal('response_helper');
            if ($fn === 'view')      $this->addSignal('view_helper');
            if ($fn === 'redirect')  $this->addSignal('redirect_helper');
        }

        // ── header('Content-Type: application/json') ─────────────────────
        if ($node instanceof Node\Expr\FuncCall &&
            $node->name instanceof Node\Name &&
            $node->name->toString() === 'header') {
            if (!empty($node->args[0])) {
                $v = $node->args[0]->value ?? $node->args[0];
                if ($v instanceof Node\Arg) $v = $v->value;
                if ($v instanceof Node\Scalar\String_) {
                    if (str_contains(strtolower($v->value), 'application/json')) {
                        $this->addSignal('json_header');
                    }
                    if (str_contains(strtolower($v->value), 'location:')) {
                        $this->addSignal('redirect');
                    }
                }
            }
        }

        // ── $_POST / $_GET / $_FILES access ──────────────────────────────
        if ($node instanceof Node\Expr\ArrayDimFetch &&
            $node->var instanceof Node\Expr\Variable &&
            is_string($node->var->name)) {
            if (in_array($node->var->name, ['_POST','_GET','_FILES','_REQUEST'], true)) {
                $this->addSignal('input_' . strtolower($node->var->name));
            }
        }

        // ── echo / print output ───────────────────────────────────────────
        if ($node instanceof Node\Stmt\Echo_ || $node instanceof Node\Stmt\InlineHTML) {
            $this->addSignal('html_output');
        }

        return null;
    }

    public function leaveNode(Node $node): void
    {
        if ($node instanceof Node\Stmt\ClassMethod) {
            $key     = $this->scopeKey();
            $signals = $this->scopeSignals[$key] ?? [];
            if (!empty($signals)) {
                $ep = $this->classifyScope($signals);
                if ($ep) {
                    $this->entryPoints[] = array_merge($ep, [
                        'file'    => $this->currentFile,
                        'handler' => $this->currentClass
                            ? "{$this->currentClass}::{$this->currentMethod}"
                            : $this->currentMethod,
                        'line'    => $node->getStartLine(),
                    ]);
                }
            }
            unset($this->scopeSignals[$key]);
            $this->currentMethod = null;
        }
        if ($node instanceof Node\Stmt\Class_) {
            $this->currentClass = null;
        }
    }

    private function addSignal(string $signal): void
    {
        $key = $this->scopeKey();
        $this->scopeSignals[$key][] = $signal;
        // Also track at file level
        $this->scopeSignals['__file'][] = $signal;
    }

    private function scopeKey(): string
    {
        return $this->currentClass
            ? "{$this->currentClass}::{$this->currentMethod}"
            : '__file';
    }

    private function classifyScope(array $signals): ?array
    {
        $hasInput    = (bool) array_filter($signals, fn($s) => str_starts_with($s, 'input_'));
        $hasOutput   = in_array('html_output', $signals, true);
        $hasJson     = in_array('json_output', $signals, true) ||
                       in_array('json_header', $signals, true);
        $hasResponse = in_array('response_helper', $signals, true) ||
                       in_array('view_helper', $signals, true) ||
                       in_array('redirect_helper', $signals, true);
        $hasRedirect = in_array('redirect', $signals, true) ||
                       in_array('redirect_helper', $signals, true);
        $hasReqCheck = in_array('request_method_check', $signals, true);

        // Must have at least one HTTP signal
        if (!$hasInput && !$hasOutput && !$hasJson && !$hasResponse && !$hasReqCheck) {
            return null;
        }

        // Determine accepted methods
        $accepts = [];
        if (in_array('input__post', $signals, true) ||
            in_array('input__request', $signals, true)) $accepts[] = 'POST';
        if (in_array('input__get', $signals, true))     $accepts[] = 'GET';
        if (empty($accepts))                             $accepts   = ['GET','POST'];

        // Determine output type
        $produces = 'mixed';
        if ($hasJson && !$hasOutput)    $produces = 'json';
        elseif ($hasOutput && !$hasJson) $produces = 'html';
        elseif ($hasRedirect)            $produces = 'redirect';

        // Classify type
        $type = 'page';
        if ($hasJson)                            $type = 'api';
        elseif ($hasInput && $hasOutput)         $type = 'form_handler';
        elseif ($hasResponse && !$hasInput)      $type = 'controller_method';
        elseif ($hasInput && !$hasOutput)        $type = 'form_processor';

        return [
            'type'     => $type,
            'accepts'  => $accepts,
            'produces' => $produces,
        ];
    }
}


// ─── Table/Column Extractor ───────────────────────────────────────────────────

/**
 * Extracts database table and column definitions from:
 *
 *   1. Raw SQL strings (CREATE TABLE, ALTER TABLE) embedded in PHP files
 *   2. Laravel migration Blueprint methods ($table->string('col'), etc.)
 *      — already handled by MigrationVisitor but this catches CREATE TABLE SQL
 *   3. Doctrine entity annotations (@ORM\Column, @ORM\Table)
 *   4. Eloquent $fillable / $guarded / $casts arrays on Model classes
 *
 * Output per table:
 *   { table: 'users', source: 'sql_ddl'|'eloquent_fillable'|'doctrine_orm'|'migration',
 *     columns: [ {name: 'email', type: 'string', nullable: false, ...} ],
 *     file: '...', line: N }
 */
class TableColumnVisitor extends NodeVisitorAbstract
{
    public array $tables = [];

    private string  $currentFile;
    private ?string $currentClass = null;

    public function __construct(string $file)
    {
        $this->currentFile = $file;
    }

    public function enterNode(Node $node): ?int
    {
        if ($node instanceof Node\Stmt\Class_) {
            $this->currentClass = $node->name ? $node->name->toString() : null;
        }

        // ── Eloquent: protected $fillable = ['col1', 'col2'] ────────────────
        if ($node instanceof Node\Stmt\PropertyProperty && $this->currentClass) {
            $propName = $node->name->toString();
            if (in_array($propName, ['fillable', 'guarded', 'casts', 'hidden'], true) &&
                $node->default instanceof Node\Expr\Array_) {
                $cols = $this->extractArrayStrings($node->default);
                if (!empty($cols)) {
                    $this->tables[] = [
                        'table'   => $this->guessTableName($this->currentClass),
                        'source'  => "eloquent_{$propName}",
                        'columns' => array_map(fn($c) => ['name'=>$c,'type'=>'mixed'], $cols),
                        'class'   => $this->currentClass,
                        'file'    => $this->currentFile,
                        'line'    => $node->getStartLine(),
                    ];
                }
            }
        }

        // ── Raw SQL strings: CREATE TABLE / ALTER TABLE ─────────────────────
        if ($node instanceof Node\Expr\Assign &&
            $node->var instanceof Node\Expr\Variable) {
            $sql = $this->extractSqlLiteral($node->expr);
            if ($sql && preg_match('/\bCREATE\s+TABLE\b/i', $sql)) {
                $parsed = $this->parseCreateTable($sql);
                if ($parsed) {
                    $parsed['source'] = 'sql_ddl';
                    $parsed['file']   = $this->currentFile;
                    $parsed['line']   = $node->getStartLine();
                    $this->tables[]   = $parsed;
                }
            }
        }

        // Also catch raw string expressions used directly in queries
        if ($node instanceof Node\Scalar\String_ || $node instanceof Node\Scalar\InterpolatedString) {
            $sql = ($node instanceof Node\Scalar\String_)
                ? $node->value
                : $this->interpolatedToText($node);
            if ($sql && preg_match('/\bCREATE\s+TABLE\b/i', $sql)) {
                $parsed = $this->parseCreateTable($sql);
                if ($parsed) {
                    $parsed['source'] = 'sql_ddl';
                    $parsed['file']   = $this->currentFile;
                    $parsed['line']   = $node->getStartLine();
                    $this->tables[]   = $parsed;
                }
            }
        }

        return null;
    }

    public function leaveNode(Node $node): void
    {
        if ($node instanceof Node\Stmt\Class_) $this->currentClass = null;
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private function extractArrayStrings(Node\Expr\Array_ $node): array
    {
        $out = [];
        foreach ($node->items as $item) {
            if (!$item) continue;
            $v = $item->value;
            // For $casts: ['email' => 'string'] — extract the key
            if ($item->key instanceof Node\Scalar\String_) {
                $out[] = $item->key->value;
            } elseif ($v instanceof Node\Scalar\String_) {
                $out[] = $v->value;
            }
        }
        return $out;
    }

    private function extractSqlLiteral(Node $node): ?string
    {
        if ($node instanceof Node\Scalar\String_) return $node->value;
        if ($node instanceof Node\Scalar\InterpolatedString) return $this->interpolatedToText($node);
        return null;
    }

    private function interpolatedToText(Node $node): string
    {
        $out = '';
        foreach ($node->parts ?? [] as $part) {
            if ($part instanceof Node\Scalar\String_)          $out .= $part->value;
            elseif ($part instanceof Node\InterpolatedStringPart) $out .= $part->value;
            else                                                    $out .= '?';
        }
        return $out;
    }

    /** Parse CREATE TABLE SQL into { table, columns[] } */
    private function parseCreateTable(string $sql): ?array
    {
        // Extract table name
        if (!preg_match('/CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\'"]?(\w+)[`\'"]?\s*\(/is', $sql, $m)) {
            return null;
        }
        $tableName = $m[1];

        // Extract the column definitions between the outermost parens
        $start = strpos($sql, '(');
        $end   = strrpos($sql, ')');
        if ($start === false || $end === false) return null;
        $body  = substr($sql, $start + 1, $end - $start - 1);

        $columns = [];
        // Split on commas that aren't inside parens (for ENUM values etc.)
        $defs = $this->splitColumnDefs($body);
        foreach ($defs as $def) {
            $def = trim($def);
            // Skip constraints
            if (preg_match('/^\s*(PRIMARY|UNIQUE|KEY|INDEX|FOREIGN|CONSTRAINT|CHECK)\b/i', $def)) continue;

            // Parse: `col_name` DATATYPE [modifiers]
            if (!preg_match('/^[`\'"]?(\w+)[`\'"]?\s+(\w+)/i', $def, $cm)) continue;
            $colName  = $cm[1];
            $colType  = strtolower($cm[2]);
            $nullable = stripos($def, 'NOT NULL') === false;
            $default  = null;
            if (preg_match('/DEFAULT\s+["\']?([^,\s"\']+)["\']?/i', $def, $dm)) {
                $default = $dm[1];
            }
            $columns[] = [
                'name'     => $colName,
                'type'     => $colType,
                'nullable' => $nullable,
                'default'  => $default,
            ];
        }

        return ['table' => $tableName, 'columns' => $columns];
    }

    private function splitColumnDefs(string $body): array
    {
        $defs  = [];
        $depth = 0;
        $cur   = '';
        for ($i = 0; $i < strlen($body); $i++) {
            $ch = $body[$i];
            if ($ch === '(') $depth++;
            if ($ch === ')') $depth--;
            if ($ch === ',' && $depth === 0) {
                $defs[] = $cur;
                $cur    = '';
            } else {
                $cur .= $ch;
            }
        }
        if (trim($cur)) $defs[] = $cur;
        return $defs;
    }

    /**
     * Guess table name from a class name using Laravel conventions.
     * e.g. "UserProfile" → "user_profiles", "Order" → "orders"
     */
    private function guessTableName(string $className): string
    {
        // Strip common suffixes
        $name = preg_replace('/(Model|Entity)$/i', '', $className);
        // CamelCase → snake_case
        $snake = strtolower(preg_replace('/([A-Z])/', '_$1', lcfirst($name)));
        // Naive pluralise
        return rtrim($snake, '_') . 's';
    }
}


// ─── File Walker ──────────────────────────────────────────────────────────────

/**
 * Recursively collects all .php files, skipping vendor, node_modules,
 * test directories, and cache folders.
 */
function collectPhpFiles(string $projectPath): array
{
    $skipDirs = [
        'vendor', 'node_modules', '.git', '.svn',
        'cache', 'tmp', 'temp', 'logs', 'storage/logs',
        'bootstrap/cache', 'public/assets', 'public/vendor',
        'test', 'tests', 'spec', 'specs',
    ];

    $files = [];
    $it    = new RecursiveIteratorIterator(
        new RecursiveDirectoryIterator($projectPath, RecursiveDirectoryIterator::SKIP_DOTS),
        RecursiveIteratorIterator::SELF_FIRST
    );

    foreach ($it as $file) {
        if (!$file->isFile() || $file->getExtension() !== 'php') continue;

        $relativePath = str_replace($projectPath . '/', '', $file->getPathname());

        // Check if any skip-dir segment appears in the relative path
        $parts   = explode('/', $relativePath);
        $skipped = false;
        foreach ($parts as $part) {
            if (in_array(strtolower($part), $skipDirs, true)) {
                $skipped = true;
                break;
            }
        }
        if ($skipped) continue;

        $files[] = $file->getPathname();
    }

    sort($files);
    return $files;
}

// ─── File Hash Cache ──────────────────────────────────────────────────────────

/**
 * Load the hash cache from disk.
 * Returns an associative array: { "absolute/path.php" => "sha1hex", ... }
 * Returns an empty array if the cache file doesn't exist or is malformed.
 */
function loadHashCache(?string $cacheFile): array
{
    if (!$cacheFile || !file_exists($cacheFile)) return [];
    $data = @json_decode(@file_get_contents($cacheFile), true);
    return is_array($data) ? $data : [];
}

/**
 * Persist the hash cache to disk (atomic write via temp file + rename).
 * Merges newly computed hashes on top of the existing cache so that
 * entries for files not in this run are preserved.
 */
function saveHashCache(?string $cacheFile, array $newHashes): void
{
    if (!$cacheFile) return;
    $existing = loadHashCache($cacheFile);
    $merged   = array_merge($existing, $newHashes);
    $tmpFile  = $cacheFile . '.tmp.' . getmypid();
    file_put_contents($tmpFile, json_encode($merged, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES));
    rename($tmpFile, $cacheFile);
}

/**
 * SHA1 hash of a file's content — fast, collision-resistant enough for
 * change detection. We use content hash rather than mtime because mtime
 * can be reset by git checkouts or rsync.
 */
function hashFile(string $path): string
{
    return sha1_file($path) ?: 'UNREADABLE';
}

// ─── Parallel Dispatcher ──────────────────────────────────────────────────────

/**
 * Split $phpFiles into $workerCount batches, spawn one PHP subprocess per
 * batch using proc_open(), then merge all partial JSON results.
 *
 * Each worker receives:
 *   php parse_project.php <projectPath> --php-version=X --worker-files=<tmpJson>
 *
 * Returns the merged result array (same shape as a normal single-process run).
 *
 * @param string[] $phpFiles     Full list of files to parse
 * @param string   $projectPath  Project root (for relative path calculation)
 * @param string   $phpBin       PHP binary path (php, php8.1, etc.)
 * @param string   $thisScript   Absolute path to this script file
 * @param string   $phpVersion   PHP version string passed to nikic parser
 * @param int      $workerCount  Number of parallel workers
 * @param array    $baseResult   Skeleton result array to merge into
 */
function runParallel(
    array  $phpFiles,
    string $projectPath,
    string $phpBin,
    string $thisScript,
    string $phpVersion,
    int    $workerCount,
    array  $baseResult
): array {
    $total = count($phpFiles);
    if ($total === 0) return $baseResult;

    // Clamp workers to file count — no point in more workers than files
    $workerCount = min($workerCount, $total);

    // Split file list evenly into batches
    $batches = array_chunk($phpFiles, (int)ceil($total / $workerCount));

    $processes  = [];
    $tmpFiles   = [];

    foreach ($batches as $i => $batch) {
        // Write the batch file list to a temp JSON file
        $tmpList = sys_get_temp_dir() . '/phpba_worker_' . getmypid() . '_' . $i . '.json';
        file_put_contents($tmpList, json_encode($batch));
        $tmpFiles[] = $tmpList;

        $cmd = escapeshellcmd($phpBin)
             . ' ' . escapeshellarg($thisScript)
             . ' ' . escapeshellarg($projectPath)
             . ' --php-version=' . escapeshellarg($phpVersion)
             . ' --worker-files=' . escapeshellarg($tmpList);

        $descriptors = [
            0 => ['pipe', 'r'],   // stdin
            1 => ['pipe', 'w'],   // stdout — JSON output
            2 => ['pipe', 'w'],   // stderr — errors/warnings
        ];

        $proc = proc_open($cmd, $descriptors, $pipes);
        if (!is_resource($proc)) {
            fwrite(STDERR, "WARNING: Failed to spawn worker {$i}, batch will be skipped\n");
            continue;
        }

        fclose($pipes[0]);  // no stdin needed
        $processes[] = ['proc' => $proc, 'pipes' => $pipes, 'batch' => $i];
    }

    // Collect all worker outputs (blocks until each finishes)
    $mergedResult = $baseResult;
    $mergeKeys = [
        'classes','routes','models','controllers','services','db_schema',
        'functions','includes','sql_queries','redirects','globals','html_pages',
        'superglobals','call_graph','form_fields','service_deps','env_vars',
        'auth_signals','http_endpoints','table_columns','errors',
    ];

    foreach ($processes as ['proc' => $proc, 'pipes' => $pipes, 'batch' => $batchIdx]) {
        $stdout = stream_get_contents($pipes[1]);
        $stderr = stream_get_contents($pipes[2]);
        fclose($pipes[1]);
        fclose($pipes[2]);
        $exitCode = proc_close($proc);

        if ($stderr) {
            fwrite(STDERR, "Worker {$batchIdx} stderr: " . substr($stderr, 0, 500) . "\n");
        }

        if ($exitCode !== 0) {
            fwrite(STDERR, "WARNING: Worker {$batchIdx} exited with code {$exitCode}\n");
            continue;
        }

        $partial = @json_decode($stdout, true);
        if (!is_array($partial)) {
            fwrite(STDERR, "WARNING: Worker {$batchIdx} produced invalid JSON: "
                         . substr($stdout, 0, 200) . "\n");
            continue;
        }

        // Merge scalar totals
        $mergedResult['total_lines'] += (int)($partial['total_lines'] ?? 0);

        // Merge array buckets
        foreach ($mergeKeys as $key) {
            if (!empty($partial[$key])) {
                $mergedResult[$key] = array_merge($mergedResult[$key] ?? [], $partial[$key]);
            }
        }
    }

    // Cleanup temp files
    foreach ($tmpFiles as $tmp) {
        @unlink($tmp);
    }

    return $mergedResult;
}

// ─── Main Parsing Loop ────────────────────────────────────────────────────────

$phpFiles = WORKER_MODE ? WORKER_FILES : collectPhpFiles($projectPath);

// ── Load file hash cache (skip unchanged files on re-runs) ────────────────────
$hashCache    = loadHashCache($cacheFile);
$freshHashes  = [];   // hashes computed this run (for files we actually parse)
$cachedResult = null; // will hold merged cached data if we skip any files

// Separate files into: needs-parse vs cache-hits
$filesToParse = [];
$cacheHitData = [];   // raw per-file cache hits aren't stored individually — see note below

foreach ($phpFiles as $filePath) {
    $hash = hashFile($filePath);
    if ($cacheFile && isset($hashCache[$filePath]) && $hashCache[$filePath] === $hash) {
        // File unchanged — skip (result is already in code_map.json from a previous run
        // that was merged into the cache). We track the hit for reporting only.
        $cacheHitData[] = $filePath;
        // Still count lines for the summary total
        $code = @file_get_contents($filePath);
        if ($code !== false) {
            $result['total_lines'] += substr_count($code, "\n") + 1;
        }
    } else {
        $filesToParse[] = $filePath;
        $freshHashes[$filePath] = $hash;  // remember for cache save
    }
}

$cacheHits   = count($cacheHitData);
$filesToParse_count = count($filesToParse);
$result['total_files'] = count($phpFiles);

if ($cacheHits > 0) {
    fwrite(STDERR, "Cache: {$cacheHits} file(s) unchanged (skipped), "
                 . "{$filesToParse_count} file(s) to parse\n");
}

// ── Parallel dispatch (only in main process, not in worker mode) ──────────────
if (!WORKER_MODE && $workerCount > 1 && $filesToParse_count > 0) {
    $phpBin     = PHP_BINARY;  // built-in constant — path to the running PHP binary
    $thisScript = __FILE__;

    fwrite(STDERR, "Parallel: spawning {$workerCount} worker(s) for {$filesToParse_count} files\n");

    $result = runParallel(
        $filesToParse, $projectPath, $phpBin, $thisScript,
        $phpVersionStr, $workerCount, $result
    );

    // Post-processing and emit after parallel merge (same as single-process path below)
    goto POST_PROCESSING;
}

// ── Single-process (sequential) parsing ──────────────────────────────────────
$allClasses    = [];
$allRoutes     = [];
$allSchema     = [];
$wpHooks       = [];
$allFunctions       = [];
$allIncludes        = [];
$allSqlQueries      = [];
$allRedirects       = [];
$allGlobals         = [];
$allHtmlPages       = [];
$allSuperglobals    = [];
$allCallEdges       = [];
$allFormFields      = [];
$allServiceDeps     = [];
$allEnvVars         = [];
$allAuthSignals     = [];
$allHttpEndpoints   = [];
$allTableColumns    = [];

foreach ($filesToParse as $filePath) {
    $code = @file_get_contents($filePath);
    if ($code === false) {
        $result['errors'][] = ['file' => $filePath, 'message' => 'Could not read file'];
        continue;
    }

    // Count lines (cache-skipped files already had their lines counted above)
    $result['total_lines'] += substr_count($code, "\n") + 1;

    // Check if this looks like a migration file
    $isMigration = str_contains($filePath, 'migration') ||
                   str_contains($filePath, 'Migration') ||
                   preg_match('/\d{4}_\d{2}_\d{2}_\d{6}_/', basename($filePath));

    try {
        $ast = $parser->parse($code);
        if ($ast === null) continue;

        // Resolve all names to FQN
        $traverser = new NodeTraverser();
        $traverser->addVisitor(new NameResolver(null, ['replaceNodes' => false]));
        $ast = $traverser->traverse($ast);

        $relPath = str_replace($projectPath . '/', '', $filePath);

        // ── OOP: class / route visitor ───────────────────────────────────────
        $visitor    = new ProjectVisitor($relPath, $result['framework']);
        $traverser2 = new NodeTraverser();
        $traverser2->addVisitor($visitor);
        $traverser2->traverse($ast);

        $allClasses = array_merge($allClasses, array_values($visitor->classes));
        $allRoutes  = array_merge($allRoutes, $visitor->routes);

        // ── Procedural: functions, SQL, redirects, includes ──────────────────
        $procVisitor  = new ProceduralVisitor($relPath);
        $traverser5   = new NodeTraverser();
        $traverser5->addVisitor($procVisitor);
        $traverser5->traverse($ast);

        $allFunctions    = array_merge($allFunctions,    array_values($procVisitor->functions));
        $allIncludes     = array_merge($allIncludes,     $procVisitor->includes);
        $allSqlQueries   = array_merge($allSqlQueries,   $procVisitor->sqlQueries);
        $allRedirects    = array_merge($allRedirects,    $procVisitor->redirects);
        $allGlobals      = array_merge($allGlobals,      $procVisitor->globals);
        $allSuperglobals = array_merge($allSuperglobals, $procVisitor->superglobalReads);
        $allCallEdges    = array_merge($allCallEdges,    $procVisitor->callEdges);
        $allEnvVars      = array_merge($allEnvVars,      $procVisitor->envVars);

        if ($procVisitor->isHtmlPage) {
            $allHtmlPages[] = $relPath;
        }

        // ── Form fields (only for HTML-mixed files) ──────────────────────────
        if ($procVisitor->isHtmlPage) {
            $formVisitor  = new FormFieldVisitor($relPath);
            $traverserF   = new NodeTraverser();
            $traverserF->addVisitor($formVisitor);
            $traverserF->traverse($ast);
            $allFormFields = array_merge($allFormFields, $formVisitor->forms);
        }

        // ── Service dependencies (OOP files only) ────────────────────────────
        if (!empty(array_values($visitor->classes))) {
            $depVisitor   = new ServiceDependencyVisitor($relPath);
            $traverserD   = new NodeTraverser();
            $traverserD->addVisitor($depVisitor);
            $traverserD->traverse($ast);
            $allServiceDeps = array_merge($allServiceDeps, $depVisitor->deps);
        }

        // ── Env vars (config files + any PHP file) ───────────────────────────
        $envVisitor   = new EnvVarVisitor($relPath);
        $traverserE   = new NodeTraverser();
        $traverserE->addVisitor($envVisitor);
        $traverserE->traverse($ast);
        $allEnvVars = array_merge($allEnvVars, $envVisitor->envVars);

        // ── Auth logic signals ───────────────────────────────────────────────
        $authVisitor  = new AuthLogicVisitor($relPath);
        $traverserA   = new NodeTraverser();
        $traverserA->addVisitor($authVisitor);
        $traverserA->traverse($ast);
        $allAuthSignals = array_merge($allAuthSignals, $authVisitor->authSignals);

        // ── HTTP entry points ────────────────────────────────────────────────
        $httpVisitor  = new HttpEntryPointVisitor($relPath);
        $traverserH   = new NodeTraverser();
        $traverserH->addVisitor($httpVisitor);
        $traverserH->traverse($ast);
        $allHttpEndpoints = array_merge($allHttpEndpoints, $httpVisitor->entryPoints);

        // ── Table/column extraction ──────────────────────────────────────────
        $tableVisitor = new TableColumnVisitor($relPath);
        $traverserT   = new NodeTraverser();
        $traverserT->addVisitor($tableVisitor);
        $traverserT->traverse($ast);
        $allTableColumns = array_merge($allTableColumns, $tableVisitor->tables);

        // ── Laravel migrations ───────────────────────────────────────────────
        if ($isMigration) {
            $migVisitor  = new MigrationVisitor($relPath);
            $traverser3  = new NodeTraverser();
            $traverser3->addVisitor($migVisitor);
            $traverser3->traverse($ast);
            $allSchema = array_merge($allSchema, $migVisitor->schema);
        }

        // ── WordPress hooks ──────────────────────────────────────────────────
        if ($result['framework'] === 'wordpress') {
            $wpVisitor  = new WordPressHookVisitor($relPath);
            $traverser4 = new NodeTraverser();
            $traverser4->addVisitor($wpVisitor);
            $traverser4->traverse($ast);
            $wpHooks = array_merge($wpHooks, $wpVisitor->hooks);
        }

    } catch (Error $e) {
        $result['errors'][] = [
            'file'    => str_replace($projectPath . '/', '', $filePath),
            'message' => $e->getMessage(),
        ];
    }
}

// ─── Save hash cache for next run ─────────────────────────────────────────────
// Only save in non-worker mode (workers don't manage the cache directly;
// the main process that spawned them will save after merging).
if (!WORKER_MODE && $cacheFile) {
    saveHashCache($cacheFile, $freshHashes);
}

POST_PROCESSING:

// In parallel mode the per-file accumulators don't exist as locals; they were
// merged into $result by runParallel(). The ??= operator safely initialises
// any that are missing without overwriting ones the loop DID populate.
$allClasses      ??= [];
$allRoutes       ??= [];
$allSchema       ??= [];
$wpHooks         ??= [];
$allFunctions    ??= [];
$allIncludes     ??= [];
$allSqlQueries   ??= [];
$allRedirects    ??= [];
$allGlobals      ??= [];
$allHtmlPages    ??= [];
$allSuperglobals ??= [];
$allCallEdges    ??= [];
$allFormFields   ??= [];
$allServiceDeps  ??= [];
$allEnvVars      ??= [];
$allAuthSignals  ??= [];
$allHttpEndpoints??= [];
$allTableColumns ??= [];

$symfonyRoutes = extractSymfonyRoutes($allClasses, $result['framework']);
$allRoutes     = array_merge($allRoutes, $symfonyRoutes);

// ─── WordPress hooks → unified routes ────────────────────────────────────────
if ($result['framework'] === 'wordpress') {
    foreach ($wpHooks as $hook) {
        $allRoutes[] = [
            'method'  => strtoupper($hook['hook_type']),
            'path'    => $hook['hook_name'],
            'handler' => $hook['callback'],
            'file'    => $hook['file'],
            'line'    => $hook['line'],
            'source'  => 'wordpress_hook',
        ];
    }
}

// ─── Classify OOP classes into CodeMap buckets ────────────────────────────────
foreach ($allClasses as $class) {
    $type = $class['type'] ?? 'class';
    if ($type === 'model')                                     $result['models'][]      = $class;
    elseif ($type === 'controller')                            $result['controllers'][] = $class;
    elseif (in_array($type, ['service', 'repository'], true)) $result['services'][]    = $class;
    else                                                       $result['classes'][]     = $class;
}

// ─── Populate procedural buckets ─────────────────────────────────────────────
$result['routes']      = $allRoutes;
$result['db_schema']   = $allSchema;
$result['functions']    = $allFunctions;
$result['includes']     = $allIncludes;
$result['sql_queries']  = $allSqlQueries;
$result['redirects']    = $allRedirects;
$result['globals']      = $allGlobals;
$result['html_pages']   = $allHtmlPages;
$result['superglobals'] = $allSuperglobals;

// ─── Call Graph: filter to user-defined function edges only ──────────────────
// Build a set of known user-defined function names so we can drop pure built-in edges
$userDefinedFns = array_flip(array_column($allFunctions, 'name'));
$phpBuiltins = [
    'isset','empty','array_key_exists','count','strlen','strpos','str_contains',
    'str_starts_with','str_ends_with','in_array','array_map','array_filter',
    'array_merge','array_push','array_pop','array_keys','array_values',
    'explode','implode','trim','ltrim','rtrim','strtolower','strtoupper',
    'substr','sprintf','printf','echo','print','var_dump','print_r',
    'header','exit','die','intval','floatval','strval','boolval',
    'date','time','mktime','json_encode','json_decode','base64_encode',
    'base64_decode','htmlspecialchars','htmlentities','nl2br','strip_tags',
    'preg_match','preg_replace','preg_split','str_replace','str_pad',
    'number_format','round','ceil','floor','abs','max','min','rand',
    'file_exists','is_dir','is_file','file_get_contents','file_put_contents',
    'ob_start','ob_get_clean','session_start','session_destroy',
    'password_hash','password_verify','md5','sha1','hash',
    'mysqli_connect','mysqli_query','mysqli_fetch_assoc','mysqli_fetch_array',
    'mysqli_num_rows','mysqli_affected_rows','mysqli_close','mysqli_real_escape_string',
    'getenv','putenv','env','require','require_once','include','include_once',
    'class_exists','method_exists','function_exists','property_exists',
    'compact','extract','list','define','defined','constant',
];
$builtinSet = array_flip($phpBuiltins);

// Collect unique edges: only include if caller or callee is user-defined
// This keeps the graph focused on application logic
$callGraphSeen = [];
$callGraph = [];
foreach ($allCallEdges as $edge) {
    $callerIsUser = isset($userDefinedFns[$edge['caller']]) || $edge['caller'] === 'GLOBAL_SCRIPT';
    $calleeIsUser = isset($userDefinedFns[$edge['callee']]);
    $calleeIsBuiltin = isset($builtinSet[$edge['callee']]);

    // Include edge if: callee is user-defined, OR caller is user-defined and calling something noteworthy
    if (!$calleeIsUser && $calleeIsBuiltin) continue;
    if (!$callerIsUser && !$calleeIsUser)  continue;

    // Deduplicate: same caller→callee in same file (keep first occurrence)
    $dedupKey = $edge['caller'] . '→' . $edge['callee'] . '@' . $edge['file'];
    if (isset($callGraphSeen[$dedupKey])) continue;
    $callGraphSeen[$dedupKey] = true;

    $callGraph[] = [
        'caller' => $edge['caller'],
        'callee' => $edge['callee'],
        'file'   => $edge['file'],
        'line'   => $edge['line'],
    ];
}
$result['call_graph'] = $callGraph;

// ─── Form fields: deduplicate by file+action+fields fingerprint ──────────────
$formSeen = [];
$uniqueForms = [];
foreach ($allFormFields as $form) {
    $fp = $form['file'] . '|' . $form['action'] . '|' . implode(',', array_column($form['fields'], 'name'));
    if (isset($formSeen[$fp])) continue;
    $formSeen[$fp] = true;
    $uniqueForms[] = $form;
}
$result['form_fields'] = $uniqueForms;

// ─── Service deps: deduplicate ───────────────────────────────────────────────
$depSeen = [];
$uniqueDeps = [];
foreach ($allServiceDeps as $dep) {
    $fp = $dep['class'] . '|' . $dep['dep_class'] . '|' . $dep['file'];
    if (isset($depSeen[$fp])) continue;
    $depSeen[$fp] = true;
    $uniqueDeps[] = $dep;
}
$result['service_deps'] = $uniqueDeps;

// ─── Env vars: deduplicate by key+file ───────────────────────────────────────
$envSeen = [];
$uniqueEnv = [];
foreach ($allEnvVars as $ev) {
    $fp = $ev['key'] . '|' . $ev['file'];
    if (isset($envSeen[$fp])) continue;
    $envSeen[$fp] = true;
    $uniqueEnv[] = $ev;
}
// Sort by key for readability
usort($uniqueEnv, fn($a, $b) => strcmp($a['key'], $b['key']));
$result['env_vars'] = $uniqueEnv;

// ─── Auth signals: deduplicate by type+pattern+file ──────────────────────────
$authSeen   = [];
$uniqueAuth = [];
foreach ($allAuthSignals as $sig) {
    $fp = $sig['type'] . '|' . $sig['pattern'] . '|' . $sig['file'];
    if (isset($authSeen[$fp])) continue;
    $authSeen[$fp] = true;
    $uniqueAuth[]  = $sig;
}
// Sort by file then line so BA output reads naturally
usort($uniqueAuth, fn($a, $b) => $a['file'] <=> $b['file'] ?: $a['line'] <=> $b['line']);
$result['auth_signals'] = $uniqueAuth;

// ─── HTTP endpoints: deduplicate by handler+file ──────────────────────────────
$epSeen      = [];
$uniqueEp    = [];
foreach ($allHttpEndpoints as $ep) {
    $fp = ($ep['handler'] ?? '') . '|' . $ep['file'];
    if (isset($epSeen[$fp])) continue;
    $epSeen[$fp] = true;
    $uniqueEp[]  = $ep;
}
$result['http_endpoints'] = $uniqueEp;

// ─── Table/column definitions: merge with migration data, deduplicate ─────────
// Migration data from MigrationVisitor is already in $result['db_schema'].
// table_columns adds Eloquent fillable + raw SQL DDL — deduplicate by table+source.
$tableSeen   = [];
$uniqueTables = [];
foreach ($allTableColumns as $tbl) {
    $fp = ($tbl['table'] ?? '?') . '|' . ($tbl['source'] ?? '?') . '|' . ($tbl['file'] ?? '?');
    if (isset($tableSeen[$fp])) continue;
    $tableSeen[$fp] = true;
    $uniqueTables[]  = $tbl;
}
// Also lift tables found in MigrationVisitor (already in db_schema) into table_columns
// for a single unified lookup — mark them so consumers know the source
foreach ($result['db_schema'] as $migration) {
    if (empty($migration['table']) || empty($migration['columns'])) continue;
    $fp = $migration['table'] . '|migration|' . ($migration['file'] ?? '?');
    if (isset($tableSeen[$fp])) continue;
    $tableSeen[$fp] = true;
    $uniqueTables[] = [
        'table'   => $migration['table'],
        'source'  => 'migration',
        'columns' => $migration['columns'],
        'file'    => $migration['file'] ?? null,
        'line'    => $migration['line'] ?? null,
    ];
}
usort($uniqueTables, fn($a, $b) => strcmp($a['table'] ?? '', $b['table'] ?? ''));
$result['table_columns'] = $uniqueTables;

// ─── Emit ─────────────────────────────────────────────────────────────────────
echo json_encode($result, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
exit(0);
