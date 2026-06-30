#!/usr/bin/env py
"""Test the analytics dashboard API."""

import urllib.request
import json

print("Testing Provenance Guard Analytics Dashboard...")
print("=" * 60)

# Test 1: Submit some content
print("\n1. Submitting test content...")
submissions = [
    {"text": "This is a formally written text with sophisticated vocabulary and complex structures.", "creator_id": "formal-user"},
    {"text": "Hey, I'm really happy about this! It's so cool, don't you think?", "creator_id": "casual-user"},
    {"text": "The implementation demonstrates various analytical capabilities.", "creator_id": "technical-user"},
]

for sub in submissions:
    payload = json.dumps(sub).encode()
    req = urllib.request.Request(
        'http://localhost:5000/submit',
        data=payload,
        headers={'Content-Type': 'application/json'}
    )
    try:
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        print(f"   ✓ {sub['creator_id']}: confidence = {data['confidence']:.3f}")
    except Exception as e:
        print(f"   ✗ {sub['creator_id']}: {str(e)[:50]}")

# Test 2: Verify creator
print("\n2. Verifying creator...")
payload = json.dumps({"creator_id": "formal-user", "verification_method": "email"}).encode()
req = urllib.request.Request(
    'http://localhost:5000/verify',
    data=payload,
    headers={'Content-Type': 'application/json'}
)
try:
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read())
    print(f"   ✓ {data['creator_id']} verified!")
except Exception as e:
    print(f"   ✗ Verification failed: {str(e)[:50]}")

# Test 3: Fetch analytics
print("\n3. Fetching analytics...")
try:
    resp = urllib.request.urlopen('http://localhost:5000/api/analytics')
    analytics = json.loads(resp.read())
    
    print(f"\n   Dashboard Metrics:")
    print(f"   - Total Submissions: {analytics['total_submissions']}")
    print(f"   - Confidence Distribution:")
    for label, count in analytics['confidence_distribution'].items():
        print(f"     • {label}: {count}")
    print(f"   - Signal Agreement: {analytics['signal_agreement_percentage']}%")
    print(f"   - Creators: {analytics['creators']['verified']}/{analytics['creators']['total']} verified")
    print(f"   - Pending Appeals: {analytics['appeals']['pending']}")
    
    if analytics['signal_statistics']:
        print(f"\n   Signal Statistics:")
        for signal_name, stats in analytics['signal_statistics'].items():
            print(f"   - {signal_name}:")
            print(f"     • Mean: {stats['mean']:.3f}, Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
except Exception as e:
    print(f"   ✗ Analytics fetch failed: {str(e)}")

print("\n" + "=" * 60)
print("✓ Analytics Dashboard Tests Complete!")
print("  → Visit http://localhost:5000/dashboard to see the interactive dashboard")
