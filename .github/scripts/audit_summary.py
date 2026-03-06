import json
import os
from datetime import datetime

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            try:
                return json.load(f)
            except:
                return None
    return None

def generate_summary():
    pip_audit = load_json('pip-audit.json')
    bandit = load_json('bandit.json')
    trivy = load_json('trivy.json')
    
    md = "## 🛡️ Resumen de Auditoría de Seguridad\n\n"
    md += f"| Fecha | Rama | Run |\n"
    md += f"|---|---|---|\n"
    md += f"| {datetime.now().strftime('%Y-%m-%d %H:%M')} | {os.environ.get('GITHUB_REF_NAME', 'unknown')} | {os.environ.get('GITHUB_RUN_NUMBER', 'unknown')} |\n\n"

    # Pip-audit
    md += "### 📦 pip-audit (Vulnerabilidades en dependencias)\n\n"
    if pip_audit:
        # pip-audit output format varies, but usually it's a list or has a dependencies key
        vulnerabilities = []
        if isinstance(pip_audit, list):
            for entry in pip_audit:
                if 'vulns' in entry:
                    for v in entry['vulns']:
                        vulnerabilities.append({
                            "pkg": entry['name'],
                            "version": entry['version'],
                            "id": v.get('id'),
                            "fix": v.get('fix_versions', ['N/A'])[0]
                        })
        
        if vulnerabilities:
            md += "| Paquete | Versión | ID | Fix |\n"
            md += "|---|---|---|---|\n"
            for v in vulnerabilities:
                md += f"| {v['pkg']} | {v['version']} | {v['id']} | {v['fix']} |\n"
        else:
            md += "✅ No se encontraron vulnerabilidades en dependencias.\n"
    else:
        md += "⚠️ No se pudo cargar el reporte de pip-audit.\n"

    # Bandit
    md += "\n### 🔍 Bandit (Análisis estático de código)\n\n"
    if bandit and 'results' in bandit:
        if bandit['results']:
            md += "| Archivo | Línea | Severidad | Confianza | Descripción |\n"
            md += "|---|---|---|---|---|\n"
            for r in bandit['results']:
                md += f"| `{r['filename']}` | {r['line_number']} | {r['issue_severity']} | {r['issue_confidence']} | {r['issue_text']} |\n"
        else:
            md += "✅ No se encontraron problemas de seguridad en el código.\n"
    else:
        md += "⚠️ No se pudo cargar el reporte de Bandit.\n"

    # Trivy
    md += "\n### 🛡️ Trivy (Escaneo de archivos y dependencias)\n\n"
    if trivy and 'Results' in trivy:
        vulnerabilities = []
        for res in trivy['Results']:
            if 'Vulnerabilities' in res:
                for v in res['Vulnerabilities']:
                    vulnerabilities.append({
                        "pkg": v.get('PkgName'),
                        "version": v.get('InstalledVersion'),
                        "severity": v.get('Severity'),
                        "id": v.get('VulnerabilityID'),
                        "fix": v.get('FixedVersion', 'N/A')
                    })
        
        if vulnerabilities:
            md += "| Paquete | Versión | Severidad | ID | Fix |\n"
            md += "|---|---|---|---|---|\n"
            for v in vulnerabilities:
                md += f"| {v['pkg']} | {v['version']} | {v['severity']} | {v['id']} | {v['fix']} |\n"
        else:
            md += "✅ No se encontraron vulnerabilidades con Trivy.\n"
    else:
        md += "⚠️ No se pudo cargar el reporte de Trivy.\n"

    with open('audit-summary.md', 'w') as f:
        f.write(md)

if __name__ == "__main__":
    generate_summary()
