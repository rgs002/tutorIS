import json
import os
import sys

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            try:
                return json.load(f)
            except:
                return None
    return None

def check_gate():
    bandit = load_json('bandit.json')
    trivy = load_json('trivy.json')
    
    fail_on = os.environ.get('SECURITY_FAIL_ON', 'CRITICAL')
    severity_scores = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'UNDEFINED': 0}
    threshold = severity_scores.get(fail_on, 4)
    
    failed = False
    
    # Check Bandit
    if bandit and 'results' in bandit:
        for r in bandit['results']:
            if severity_scores.get(r['issue_severity'], 0) >= threshold:
                print(f"❌ Bandit: {r['issue_severity']} issue found in {r['filename']}:{r['line_number']}")
                if severity_scores.get(r['issue_severity'], 0) >= severity_scores['CRITICAL']:
                    failed = True

    # Check Trivy
    if trivy and 'Results' in trivy:
        for res in trivy['Results']:
            if 'Vulnerabilities' in res:
                for v in res['Vulnerabilities']:
                    if severity_scores.get(v['Severity'], 0) >= threshold:
                        print(f"❌ Trivy: {v['Severity']} vulnerability {v['VulnerabilityID']} in {v.get('PkgName')}")
                        if severity_scores.get(v['Severity'], 0) >= severity_scores['CRITICAL']:
                            failed = True

    if failed:
        print("\n💥 Security gate failed. Please fix CRITICAL vulnerabilities.")
        sys.exit(1)
    else:
        print("\n✅ Security gate passed (or only non-blocking issues found).")

if __name__ == "__main__":
    check_gate()
