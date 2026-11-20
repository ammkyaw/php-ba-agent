import pytest
from pathlib import Path
from pipeline.parsers.typescript_parser import _extract_nextjs_routes

def test_extract_nextjs_app_route():
    routes = []
    root = Path("/fake/root")
    src_file = Path("/fake/root/src/app/api/users/route.ts")
    rel = "src/app/api/users/route.ts"
    
    with pytest.MonkeyPatch.context() as m:
        m.setattr("pipeline.parsers.typescript_parser.LanguageParser.safe_read", lambda x: "export async function GET() {}")
        _extract_nextjs_routes(src_file, root, rel, routes)
        
    assert len(routes) == 1
    assert routes[0]["method"] == "GET"
    assert routes[0]["path"] == "/api/users"
    assert routes[0]["file"] == rel
    assert routes[0]["dir"] == "src/app/api/users"
    assert routes[0]["kind"] == "nextjs_app_api"

def test_extract_nextjs_app_route_no_methods():
    routes = []
    root = Path("/fake/root")
    src_file = Path("/fake/root/src/app/api/status/route.tsx")
    rel = "src/app/api/status/route.tsx"
    
    with pytest.MonkeyPatch.context() as m:
        m.setattr("pipeline.parsers.typescript_parser.LanguageParser.safe_read", lambda x: "const dummy = true;")
        _extract_nextjs_routes(src_file, root, rel, routes)
        
    assert len(routes) == 1
    assert routes[0]["method"] == "ANY"
    assert routes[0]["path"] == "/api/status"
    assert routes[0]["file"] == rel
    assert routes[0]["dir"] == "src/app/api/status"
    assert routes[0]["kind"] == "nextjs_app_api"

def test_extract_nextjs_dynamic_segments():
    routes = []
    root = Path("/fake/root")
    src_file = Path("/fake/root/src/app/projects/[id]/route.jsx")
    rel = "src/app/projects/[id]/route.jsx"
    
    with pytest.MonkeyPatch.context() as m:
        m.setattr("pipeline.parsers.typescript_parser.LanguageParser.safe_read", lambda x: "export function POST() {}")
        _extract_nextjs_routes(src_file, root, rel, routes)
        
    assert len(routes) == 1
    assert routes[0]["method"] == "POST"
    assert routes[0]["path"] == "/projects/:id"
    assert routes[0]["file"] == rel
    assert routes[0]["dir"] == "src/app/projects/[id]"
    assert routes[0]["kind"] == "nextjs_app_api"

@pytest.mark.parametrize("filename", ["page.tsx", "page.ts", "page.jsx", "page.js"])
def test_extract_nextjs_app_page(filename):
    routes = []
    root = Path("/fake/root")
    src_file = Path(f"/fake/root/src/app/dashboard/{filename}")
    rel = f"src/app/dashboard/{filename}"
    
    _extract_nextjs_routes(src_file, root, rel, routes)
    
    assert len(routes) == 1
    assert routes[0]["method"] == "GET"
    assert routes[0]["path"] == "/dashboard"
    assert routes[0]["file"] == rel
    assert routes[0]["dir"] == "src/app/dashboard"
    assert routes[0]["kind"] == "nextjs_page"
