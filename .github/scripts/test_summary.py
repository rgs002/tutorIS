import xml.etree.ElementTree as ET
import os

def parse_junit(file_path):
    if not os.path.exists(file_path):
        return None
    
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    summary = {
        "total": 0, "passed": 0, "failed": 0, "skipped": 0, "time": 0.0
    }
    
    tests = []
    
    # Handle multiple testsuites or single testsuite
    suites = root.findall('testsuite') if root.tag == 'testsuites' else [root]
    
    for suite in suites:
        summary["total"] += int(suite.get('tests', 0))
        summary["failed"] += int(suite.get('failures', 0)) + int(suite.get('errors', 0))
        summary["skipped"] += int(suite.get('skipped', 0))
        summary["time"] += float(suite.get('time', 0))
        
        for case in suite.findall('testcase'):
            name = case.get('name')
            classname = case.get('classname')
            time = case.get('time')
            status = "✅ passed"
            error_msg = ""
            
            failure = case.find('failure')
            if failure is not None:
                status = "❌ failed"
                error_msg = failure.text
            
            error = case.find('error')
            if error is not None:
                status = "❌ failed"
                error_msg = error.text
                
            skipped = case.find('skipped')
            if skipped is not None:
                status = "⏭️ skipped"
                
            tests.append({
                "name": name,
                "classname": classname,
                "time": time,
                "status": status,
                "error": error_msg
            })
            
    summary["passed"] = summary["total"] - summary["failed"] - summary["skipped"]
    return summary, tests

def generate_markdown(summary, tests):
    md = "## 🧪 Resultados de Tests\n\n"
    
    # Summary Table
    md += "| ✅ Passed | ❌ Failed | ⏭️ Skipped | Total | ⏱️ Duration |\n"
    md += "|---|---|---|---|---|\n"
    md += f"| {summary['passed']} | {summary['failed']} | {summary['skipped']} | {summary['total']} | {summary['time']:.2f}s |\n\n"
    
    # Detailed Table
    md += "### 📋 Detalle de Ejecución\n\n"
    md += "| Status | Module | Test | Time |\n"
    md += "|---|---|---|---|\n"
    
    # Group by classname
    tests.sort(key=lambda x: x['classname'])
    for test in tests:
        md += f"| {test['status']} | `{test['classname']}` | {test['name']} | {test['time']}s |\n"
        
    # Failed tests details
    failed_tests = [t for t in tests if "failed" in t['status']]
    if failed_tests:
        md += "\n### ❌ Detalles de Errores\n\n"
        for t in failed_tests:
            md += f"<details><summary><b>{t['classname']}.{t['name']}</b></summary>\n\n"
            md += f"```text\n{t['error']}\n```\n\n"
            md += "</details>\n"
            
    return md

if __name__ == "__main__":
    result = parse_junit("test-results.xml")
    if result:
        summary, tests = result
        markdown = generate_markdown(summary, tests)
        with open("test-summary.md", "w") as f:
            f.write(markdown)
    else:
        with open("test-summary.md", "w") as f:
            f.write("## 🧪 Resultados de Tests\n\nNo se encontraron resultados de tests.")
